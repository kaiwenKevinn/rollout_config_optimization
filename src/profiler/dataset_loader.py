"""
GPQA Dataset Loader Module

Loads and preprocesses the GPQA-Diamond dataset for benchmarking.
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
class GPQAQuestion:
    """Represents a single GPQA question with metadata."""
    question_id: str
    question: str
    choices: List[str]
    correct_answer: str
    correct_answer_index: int
    subject: str
    prompt: str = ""
    input_tokens: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GPQADatasetStats:
    """Statistics about the loaded dataset."""
    total_questions: int
    subjects: Dict[str, int]
    avg_question_length: float
    min_question_length: int
    max_question_length: int
    avg_choices_per_question: float


class GPQADatasetLoader:
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
        self.dataset_name = dataset_name
        self.subset = subset
        self.split = split
        self.cache_dir = cache_dir
        self.max_samples = max_samples
        self.prompt_style = prompt_style
        
        self._dataset: Optional[Dataset] = None
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


def load_gpqa_dataset(config: Dict[str, Any]) -> GPQADatasetLoader:
    """
    Convenience function to load GPQA dataset from config.
    
    Args:
        config: Configuration dictionary with dataset settings
        
    Returns:
        Loaded GPQADatasetLoader instance
    """
    dataset_config = config.get('dataset', {})
    
    loader = GPQADatasetLoader(
        dataset_name=dataset_config.get('name', 'Idavidrein/gpqa'),
        subset=dataset_config.get('subset', 'gpqa_diamond'),
        split=dataset_config.get('split', 'train'),
        cache_dir=dataset_config.get('cache_dir'),
        max_samples=dataset_config.get('max_samples'),
        prompt_style=dataset_config.get('prompt_style', 'detailed')
    )
    
    loader.load()
    return loader
