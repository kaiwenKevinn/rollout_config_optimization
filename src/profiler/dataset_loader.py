"""
Generic Dataset Loader Module

Provides base classes and implementations for loading different datasets for benchmarking.
Currently supports GPQA-Diamond and AIME 25 datasets.
Supports caching and various prompt formats.
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from pathlib import Path

from datasets import load_dataset, Dataset
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class BaseQuestion:
    """Base class for representing questions from different datasets."""
    question_id: str
    question: str
    subject: str
    
    # Optional fields with defaults
    prompt: str = ""
    input_tokens: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GPQAQuestion:
    """Represents a single GPQA question with multiple choice metadata."""
    # Required fields
    question_id: str
    question: str
    subject: str
    choices: List[str]
    correct_answer: str
    correct_answer_index: int
    
    # Optional fields with defaults
    prompt: str = ""
    input_tokens: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AIME25Question:
    """Represents a single AIME 25 question with mathematical problem metadata."""
    # Required fields
    question_id: str
    question: str
    subject: str
    answer: str  # Integer answer in format "000" to "999"
    problem_type: str  # e.g., "algebra", "geometry", "number_theory", "combinatorics"
    
    # Optional fields with defaults
    prompt: str = ""
    input_tokens: int = 0
    difficulty_level: str = "intermediate"  # "easy", "intermediate", "hard"
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BaseDatasetStats:
    """Base statistics for any dataset."""
    total_questions: int
    subjects: Dict[str, int]
    avg_question_length: float
    min_question_length: int
    max_question_length: int
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GPQADatasetStats(BaseDatasetStats):
    """Statistics specific to GPQA dataset."""
    avg_choices_per_question: float


@dataclass
class AIME25DatasetStats(BaseDatasetStats):
    """Statistics specific to AIME 25 dataset."""
    problem_types: Dict[str, int]
    difficulty_distribution: Dict[str, int]
    avg_answer_length: float


class BaseDatasetLoader:
    """
    Abstract base class for dataset loaders.
    
    Provides common functionality for loading, preprocessing, and caching datasets.
    Subclasses should implement dataset-specific logic.
    """
    
    def __init__(
        self,
        dataset_name: str,
        split: str = "train",
        cache_dir: Optional[str] = None,
        max_samples: Optional[int] = None,
        prompt_style: str = "detailed"
    ):
        """
        Initialize the base dataset loader.
        
        Args:
            dataset_name: Dataset identifier/name
            split: Dataset split to load
            cache_dir: Directory to cache the dataset
            max_samples: Maximum number of samples to load (None for all)
            prompt_style: Style of prompt to generate
        """
        self.dataset_name = dataset_name
        self.split = split
        self.cache_dir = cache_dir
        self.max_samples = max_samples
        self.prompt_style = prompt_style
        
        self._dataset: Optional[Dataset] = None
        self._questions: List[BaseQuestion] = []
        self._stats: Optional[BaseDatasetStats] = None
    
    def load(self) -> List[BaseQuestion]:
        """
        Load and preprocess the dataset.
        
        Returns:
            List of question objects
        """
        raise NotImplementedError("Subclasses must implement load method")
    
    def _process_dataset(self) -> List[BaseQuestion]:
        """Process raw dataset into question objects."""
        raise NotImplementedError("Subclasses must implement _process_dataset method")
    
    def _build_prompt(self, question: str, **kwargs) -> str:
        """Build a formatted prompt for the question."""
        raise NotImplementedError("Subclasses must implement _build_prompt method")
    
    def get_questions(self) -> List[BaseQuestion]:
        """Get loaded questions."""
        if not self._questions:
            self.load()
        return self._questions
    
    def get_prompts(self) -> List[str]:
        """Get list of prompts for all questions."""
        return [q.prompt for q in self.get_questions()]
    
    def get_questions_by_subject(self) -> Dict[str, List[BaseQuestion]]:
        """Group questions by subject/domain."""
        by_subject: Dict[str, List[BaseQuestion]] = {}
        
        for q in self.get_questions():
            if q.subject not in by_subject:
                by_subject[q.subject] = []
            by_subject[q.subject].append(q)
        
        return by_subject
    
    def get_statistics(self) -> BaseDatasetStats:
        """Calculate and return dataset statistics."""
        raise NotImplementedError("Subclasses must implement get_statistics method")
    
    def export_to_jsonl(self, output_path: str) -> None:
        """Export questions to JSONL format."""
        with open(output_path, 'w', encoding='utf-8') as f:
            for q in self.get_questions():
                f.write(json.dumps(q.to_dict(), ensure_ascii=False) + '\n')
        
        logger.info(f"Exported {len(self._questions)} questions to {output_path}")
    
    def _load_from_cache(self, cache_path: Path) -> List[BaseQuestion]:
        """Load processed questions from cache."""
        raise NotImplementedError("Subclasses must implement _load_from_cache method")
    
    def _save_to_cache(self, cache_path: Path) -> None:
        """Save processed questions to cache."""
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'dataset_name': self.dataset_name,
            'split': self.split,
            'questions': [q.to_dict() for q in self._questions]
        }
        
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Cached processed data to: {cache_path}")
    
    def __len__(self) -> int:
        return len(self.get_questions())
    
    def __iter__(self):
        return iter(self.get_questions())
    
    def __getitem__(self, idx: int) -> BaseQuestion:
        return self.get_questions()[idx]


class GPQADatasetLoader(BaseDatasetLoader):
    """
    Loader for GPQA-Diamond dataset.
    
    Handles dataset loading, preprocessing, and caching.
    Supports multiple prompt formats for different evaluation scenarios.
    """
    
    # Prompt template for multiple choice questions
    PROMPT_TEMPLATE = """Answer the following multiple choice question. Think step by step and provide your reasoning, then give your final answer.

Question: {question}

Choices:
A. {choice_a}
B. {choice_b}
C. {choice_c}
D. {choice_d}

"Please show your choice in the answer field with only the choice letter, e.g., "answer": "C"."."""

    SIMPLE_PROMPT_TEMPLATE = """Question: {question}

A. {choice_a}
B. {choice_b}
C. {choice_c}
D. {choice_d}

Answer:"""

    def __init__(
        self,
        dataset_name: str = "Idavidrein/gpqa",
        subset: str = "gpqa_diamond",
        split: str = "train",
        cache_dir: Optional[str] = None,
        max_samples: Optional[int] = None,
        prompt_style: str = "detailed"  # "detailed" or "simple"
    ):
        """
        Initialize the GPQA dataset loader.
        
        Args:
            dataset_name: HuggingFace dataset name
            subset: Dataset subset (gpqa_diamond, gpqa_main, gpqa_extended)
            split: Dataset split to load
            cache_dir: Directory to cache the dataset
            max_samples: Maximum number of samples to load (None for all)
            prompt_style: Style of prompt to generate ("detailed" or "simple")
        """
        super().__init__(dataset_name, split, cache_dir, max_samples, prompt_style)
        self.subset = subset
        self._questions: List[GPQAQuestion] = []
        self._stats: Optional[GPQADatasetStats] = None
        
    def load(self) -> List[GPQAQuestion]:
        """
        Load and preprocess the dataset.
        
        Returns:
            List of GPQAQuestion objects
        """
        logger.info(f"Loading GPQA dataset: {self.dataset_name}/{self.subset}")
        
        # Check for cached data
        if self.cache_dir:
            cache_path = Path(self.cache_dir) / f"{self.subset}_{self.split}_processed.json"
            if cache_path.exists():
                logger.info(f"Loading from cache: {cache_path}")
                return self._load_from_cache(cache_path)
        
        # Load from HuggingFace
        try:
            self._dataset = load_dataset(
                self.dataset_name,
                self.subset,
                split=self.split,
                cache_dir=self.cache_dir,
                trust_remote_code=True
            )
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise
        
        # Process dataset
        self._questions = self._process_dataset()
        
        # Apply max_samples limit
        if self.max_samples is not None and len(self._questions) > self.max_samples:
            self._questions = self._questions[:self.max_samples]
            logger.info(f"Limited to {self.max_samples} samples")
        
        # Cache processed data
        if self.cache_dir:
            self._save_to_cache(cache_path)
        
        logger.info(f"Loaded {len(self._questions)} questions")
        return self._questions
    
    def _process_dataset(self) -> List[GPQAQuestion]:
        """Process raw dataset into GPQAQuestion objects."""
        questions = []
        
        for idx, item in enumerate(tqdm(self._dataset, desc="Processing questions")):
            # Extract question data
            question_text = item.get("Question", item.get("question", ""))
            
            # Extract choices - GPQA uses different column names
            choices = []
            for key in ["Correct Answer", "Incorrect Answer 1", "Incorrect Answer 2", "Incorrect Answer 3"]:
                if key in item:
                    choices.append(item[key])
            
            # If choices not found in expected format, try alternative format
            if not choices:
                for i in range(4):
                    choice_key = f"choice_{i}" if f"choice_{i}" in item else f"Choice {i}"
                    if choice_key in item:
                        choices.append(item[choice_key])
            
            # Ensure we have 4 choices
            while len(choices) < 4:
                choices.append("")
            
            # Get correct answer index (GPQA has correct answer first, then shuffles)
            correct_answer = item.get("Correct Answer", choices[0] if choices else "")
            correct_idx = 0  # In original format, correct answer is first
            
            # Get subject/domain
            subject = item.get("High-level domain", item.get("Subdomain", "Unknown"))
            
            # Build prompt
            prompt = self._build_prompt(question_text, choices)
            
            # Create question object
            gpqa_q = GPQAQuestion(
                question_id=f"gpqa_{self.subset}_{idx}",
                question=question_text,
                choices=choices,
                correct_answer=correct_answer,
                correct_answer_index=correct_idx,
                subject=subject,
                prompt=prompt
            )
            
            questions.append(gpqa_q)
        
        return questions
    
    def _build_prompt(self, question: str, choices: List[str]) -> str:
        """Build a formatted prompt for the question."""
        template = self.PROMPT_TEMPLATE if self.prompt_style == "detailed" else self.SIMPLE_PROMPT_TEMPLATE
        
        # Ensure we have 4 choices
        while len(choices) < 4:
            choices.append("")
        
        return template.format(
            question=question,
            choice_a=choices[0],
            choice_b=choices[1],
            choice_c=choices[2],
            choice_d=choices[3]
        )
    
    def _load_from_cache(self, cache_path: Path) -> List[GPQAQuestion]:
        """Load processed questions from cache."""
        with open(cache_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self._questions = [GPQAQuestion(**q) for q in data['questions']]
        
        if self.max_samples is not None and len(self._questions) > self.max_samples:
            self._questions = self._questions[:self.max_samples]
        
        return self._questions
    
    def _save_to_cache(self, cache_path: Path) -> None:
        """Save processed questions to cache."""
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'dataset_name': self.dataset_name,
            'subset': self.subset,
            'split': self.split,
            'questions': [q.to_dict() for q in self._questions]
        }
        
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Cached processed data to: {cache_path}")
    
    def get_questions(self) -> List[GPQAQuestion]:
        """Get loaded questions."""
        if not self._questions:
            self.load()
        return self._questions
    
    def get_prompts(self) -> List[str]:
        """Get list of prompts for all questions."""
        return [q.prompt for q in self.get_questions()]
    
    def get_questions_by_subject(self) -> Dict[str, List[GPQAQuestion]]:
        """Group questions by subject/domain."""
        by_subject: Dict[str, List[GPQAQuestion]] = {}
        
        for q in self.get_questions():
            if q.subject not in by_subject:
                by_subject[q.subject] = []
            by_subject[q.subject].append(q)
        
        return by_subject
    
    def get_statistics(self) -> GPQADatasetStats:
        """Calculate and return dataset statistics."""
        if self._stats is not None:
            return self._stats
        
        questions = self.get_questions()
        
        # Count subjects
        subjects: Dict[str, int] = {}
        question_lengths = []
        
        for q in questions:
            subjects[q.subject] = subjects.get(q.subject, 0) + 1
            question_lengths.append(len(q.question))
        
        self._stats = GPQADatasetStats(
            total_questions=len(questions),
            subjects=subjects,
            avg_question_length=sum(question_lengths) / len(question_lengths) if question_lengths else 0,
            min_question_length=min(question_lengths) if question_lengths else 0,
            max_question_length=max(question_lengths) if question_lengths else 0,
            avg_choices_per_question=4.0  # GPQA always has 4 choices
        )
        
        return self._stats
    
    def export_to_jsonl(self, output_path: str) -> None:
        """Export questions to JSONL format."""
        with open(output_path, 'w', encoding='utf-8') as f:
            for q in self.get_questions():
                f.write(json.dumps(q.to_dict(), ensure_ascii=False) + '\n')
        
        logger.info(f"Exported {len(self._questions)} questions to {output_path}")
    
    def __len__(self) -> int:
        return len(self.get_questions())
    
    def __iter__(self):
        return iter(self.get_questions())
    
    def __getitem__(self, idx: int) -> GPQAQuestion:
        return self.get_questions()[idx]




class AIME25DatasetLoader(BaseDatasetLoader):
    """
    Loader for AIME 25 mathematics competition dataset.
    
    Handles loading and preprocessing of AIME mathematical problems.
    Supports various prompt formats for mathematical reasoning tasks.
    """
    
    # Prompt templates for mathematical problems
    PROMPT_TEMPLATE = """Please reason step by step, and put your final answer within \boxed{{}}.

Problem: {question}

Solution:"""
    
    SIMPLE_PROMPT_TEMPLATE = """Problem: {question}

Answer:"""
    
    def __init__(
        self,
        dataset_name: str = "opencompass/AIME2025",
        subset: str = "AIME2025-I",  # AIME25 has two subsets: AIME2025-I and AIME2025-II
        split: str = "test",  # AIME25 only has 'test' split
        cache_dir: Optional[str] = None,
        max_samples: Optional[int] = None,
        prompt_style: str = "detailed"  # "detailed" or "simple"
    ):
        """
        Initialize the AIME 25 dataset loader.
        
        Args:
            dataset_name: HuggingFace dataset name for AIME 25
            subset: Dataset subset ("AIME2025-I" or "AIME2025-II")
            split: Dataset split to load (AIME25 only has 'test' split)
            cache_dir: Directory to cache the dataset
            max_samples: Maximum number of samples to load (None for all)
            prompt_style: Style of prompt to generate ("detailed" or "simple")
        """
        super().__init__(dataset_name, split, cache_dir, max_samples, prompt_style)
        self.subset = subset
        self._questions: List[AIME25Question] = []
        self._stats: Optional[AIME25DatasetStats] = None
    
    def load(self) -> List[AIME25Question]:
        """
        Load and preprocess the AIME 25 dataset.
        
        Returns:
            List of AIME25Question objects
        """
        logger.info(f"Loading AIME 25 dataset: {self.dataset_name}")
        
        # Check for cached data
        if self.cache_dir:
            cache_path = Path(self.cache_dir) / f"aime25_{self.subset}_{self.split}_processed.json"
            if cache_path.exists():
                logger.info(f"Loading from cache: {cache_path}")
                return self._load_from_cache(cache_path)
        
        # Load from HuggingFace
        try:
            self._dataset = load_dataset(
                self.dataset_name,
                self.subset,
                split=self.split,
                cache_dir=self.cache_dir,
                trust_remote_code=True
            )
        except Exception as e:
            logger.error(f"Failed to load AIME 25 dataset: {e}")
            raise
        
        # Process dataset
        self._questions = self._process_dataset()
        
        # Apply max_samples limit
        if self.max_samples is not None and len(self._questions) > self.max_samples:
            self._questions = self._questions[:self.max_samples]
            logger.info(f"Limited to {self.max_samples} samples")
        
        # Cache processed data
        if self.cache_dir:
            self._save_to_cache(cache_path)
        
        logger.info(f"Loaded {len(self._questions)} AIME problems")
        return self._questions
    
    def _process_dataset(self) -> List[AIME25Question]:
        """Process raw AIME 25 dataset into AIME25Question objects."""
        questions = []
        
        for idx, item in enumerate(tqdm(self._dataset, desc="Processing AIME problems")):
            # Extract problem data
            question_text = item.get("problem", item.get("question", ""))
            answer = str(item.get("answer", "")).zfill(3)  # Ensure 3-digit format
            
            # Get problem type/category
            problem_type = item.get("type", item.get("category", "unknown")).lower()
            if problem_type not in ["algebra", "geometry", "number_theory", "combinatorics"]:
                problem_type = "other"
            
            # Get difficulty level if available
            difficulty = item.get("difficulty", "intermediate").lower()
            if difficulty not in ["easy", "intermediate", "hard"]:
                difficulty = "intermediate"
            
            # Get subject (using problem type as subject for AIME)
            subject = problem_type.replace("_", " ").title()
            
            # Build prompt
            prompt = self._build_prompt(question_text)
            
            # Create question object
            aime_q = AIME25Question(
                question_id=f"aime25_{idx}",
                question=question_text,
                answer=answer,
                problem_type=problem_type,
                difficulty_level=difficulty,
                subject=subject,
                prompt=prompt
            )
            
            questions.append(aime_q)
        
        return questions
    
    def _build_prompt(self, question: str) -> str:
        """Build a formatted prompt for the mathematical problem."""
        template = self.PROMPT_TEMPLATE if self.prompt_style == "detailed" else self.SIMPLE_PROMPT_TEMPLATE
        return template.format(question=question)
    
    def _load_from_cache(self, cache_path: Path) -> List[AIME25Question]:
        """Load processed AIME questions from cache."""
        with open(cache_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self._questions = []
        for q_data in data['questions']:
            # Handle both old and new formats
            if 'problem_type' in q_data:
                self._questions.append(AIME25Question(**q_data))
            else:
                # Convert old format if needed
                base_question = BaseQuestion(**{k: v for k, v in q_data.items() 
                                              if k in ['question_id', 'question', 'subject', 'prompt', 'input_tokens']})
                aime_question = AIME25Question(
                    question_id=base_question.question_id,
                    question=base_question.question,
                    subject=base_question.subject,
                    prompt=base_question.prompt,
                    input_tokens=base_question.input_tokens,
                    answer=q_data.get('answer', '000'),
                    problem_type=q_data.get('problem_type', 'other'),
                    difficulty_level=q_data.get('difficulty_level', 'intermediate')
                )
                self._questions.append(aime_question)
        
        if self.max_samples is not None and len(self._questions) > self.max_samples:
            self._questions = self._questions[:self.max_samples]
        
        return self._questions
    
    def _save_to_cache(self, cache_path: Path) -> None:
        """Save processed AIME questions to cache."""
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'dataset_name': self.dataset_name,
            'subset': self.subset,
            'split': self.split,
            'questions': [q.to_dict() for q in self._questions]
        }
        
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Cached AIME processed data to: {cache_path}")
    
    def get_questions(self) -> List[AIME25Question]:
        """Get loaded AIME questions."""
        if not self._questions:
            self.load()
        return self._questions
    
    def get_questions_by_problem_type(self) -> Dict[str, List[AIME25Question]]:
        """Group questions by problem type."""
        by_type: Dict[str, List[AIME25Question]] = {}
        
        for q in self.get_questions():
            ptype = q.problem_type
            if ptype not in by_type:
                by_type[ptype] = []
            by_type[ptype].append(q)
        
        return by_type
    
    def get_questions_by_difficulty(self) -> Dict[str, List[AIME25Question]]:
        """Group questions by difficulty level."""
        by_difficulty: Dict[str, List[AIME25Question]] = {}
        
        for q in self.get_questions():
            diff = q.difficulty_level
            if diff not in by_difficulty:
                by_difficulty[diff] = []
            by_difficulty[diff].append(q)
        
        return by_difficulty
    
    def get_statistics(self) -> AIME25DatasetStats:
        """Calculate and return AIME 25 dataset statistics."""
        if self._stats is not None:
            return self._stats
        
        questions = self.get_questions()
        
        # Count subjects and problem types
        subjects: Dict[str, int] = {}
        problem_types: Dict[str, int] = {}
        difficulty_dist: Dict[str, int] = {}
        question_lengths = []
        answer_lengths = []
        
        for q in questions:
            subjects[q.subject] = subjects.get(q.subject, 0) + 1
            problem_types[q.problem_type] = problem_types.get(q.problem_type, 0) + 1
            difficulty_dist[q.difficulty_level] = difficulty_dist.get(q.difficulty_level, 0) + 1
            question_lengths.append(len(q.question))
            answer_lengths.append(len(q.answer))
        
        self._stats = AIME25DatasetStats(
            total_questions=len(questions),
            subjects=subjects,
            problem_types=problem_types,
            difficulty_distribution=difficulty_dist,
            avg_question_length=sum(question_lengths) / len(question_lengths) if question_lengths else 0,
            min_question_length=min(question_lengths) if question_lengths else 0,
            max_question_length=max(question_lengths) if question_lengths else 0,
            avg_answer_length=sum(answer_lengths) / len(answer_lengths) if answer_lengths else 0
        )
        
        return self._stats
    
    def __getitem__(self, idx: int) -> AIME25Question:
        return self.get_questions()[idx]


def create_dataset_loader(dataset_type: str, **kwargs) -> BaseDatasetLoader:
    """
    Factory function to create appropriate dataset loader based on type.
    
    Args:
        dataset_type: Type of dataset ("gpqa" or "aime25")
        **kwargs: Arguments to pass to the loader constructor
    
    Returns:
        Appropriate dataset loader instance
    """
    if dataset_type.lower() == "gpqa":
        return GPQADatasetLoader(**kwargs)
    elif dataset_type.lower() == "aime25":
        return AIME25DatasetLoader(**kwargs)
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}. Supported types: gpqa, aime25")


def load_dataset_by_type(config: Dict[str, Any]) -> BaseDatasetLoader:
    """
    Convenience function to load dataset based on configuration.
    
    Args:
        config: Configuration dictionary with dataset settings
        
    Returns:
        Loaded dataset loader instance
    """
    dataset_config = config.get('dataset', {})
    dataset_type = dataset_config.get('type', 'gpqa')  # default to gpqa for backward compatibility
    
    # Prepare arguments based on dataset type
    loader_kwargs = {
        'cache_dir': dataset_config.get('cache_dir'),
        'max_samples': dataset_config.get('max_samples'),
        'prompt_style': dataset_config.get('prompt_style', 'detailed')
    }
    
    if dataset_type.lower() == "gpqa":
        loader_kwargs.update({
            'dataset_name': dataset_config.get('name', 'Idavidrein/gpqa'),
            'subset': dataset_config.get('subset', 'gpqa_diamond'),
            'split': dataset_config.get('split', 'train')
        })
    elif dataset_type.lower() == "aime25":
        loader_kwargs.update({
            'dataset_name': dataset_config.get('name', 'opencompass/AIME2025'),
            'subset': dataset_config.get('subset', 'AIME2025-I'),
            'split': dataset_config.get('split', 'test')
        })
    
    loader = create_dataset_loader(dataset_type, **loader_kwargs)
    loader.load()
    return loader
