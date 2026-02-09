"""
Sequence Profiler Module

Analyzes sequence lengths using model tokenizer and categorizes
sequences for intelligent routing decisions.
"""

import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from collections import defaultdict
from enum import Enum

import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer

from .dataset_loader import GPQAQuestion, GPQADatasetLoader

logger = logging.getLogger(__name__)


class SequenceCategory(Enum):
    """Categories for sequence lengths."""
    SHORT = "short"
    MEDIUM = "medium"
    LONG = "long"
    EXTRA_LONG = "extra_long"


@dataclass
class SequenceInfo:
    """
    Information about a single sequence.
    
    Contains both estimated (pre-inference) and actual (post-inference) metrics.
    The actual values are populated after inference completion.
    """
    question_id: str
    prompt: str
    input_tokens: int
    estimated_output_tokens: int
    total_estimated_tokens: int
    category: str  # Based on input tokens (pre-inference)
    subject: str
    
    # Post-inference actual metrics (populated after generation)
    actual_output_tokens: Optional[int] = None
    actual_total_tokens: Optional[int] = None
    actual_category: Optional[str] = None  # Based on actual total tokens
    is_completed: bool = False  # Flag indicating inference completion
    generation_time_ms: Optional[float] = None  # Time taken for generation
    
    def update_with_actual_output(
        self,
        actual_output_tokens: int,
        generation_time_ms: Optional[float] = None,
        categorizer: Optional['SequenceProfiler'] = None
    ) -> 'SequenceInfo':
        """
        Update sequence info with actual inference results.
        
        Args:
            actual_output_tokens: Actual number of output tokens generated
            generation_time_ms: Time taken for generation in milliseconds
            categorizer: SequenceProfiler instance to recategorize based on total length
            
        Returns:
            Self for method chaining
        """
        self.actual_output_tokens = actual_output_tokens
        self.actual_total_tokens = self.input_tokens + actual_output_tokens
        self.is_completed = True
        self.generation_time_ms = generation_time_ms
        
        # Recategorize based on actual total tokens if categorizer provided
        if categorizer is not None:
            self.actual_category = categorizer.categorize_by_total(
                self.actual_total_tokens
            ).value
        
        return self
    
    def get_routing_category(self) -> str:
        """
        Get the category to use for routing decisions.
        
        Returns actual_category if available (post-inference),
        otherwise returns the estimated category (pre-inference).
        """
        return self.actual_category if self.actual_category else self.category
    
    def get_total_tokens(self) -> int:
        """
        Get total tokens for routing decisions.
        
        Returns actual_total_tokens if available (post-inference),
        otherwise returns total_estimated_tokens (pre-inference).
        """
        return self.actual_total_tokens if self.actual_total_tokens else self.total_estimated_tokens


@dataclass
class SequenceDistribution:
    """Distribution statistics for sequences."""
    total_sequences: int
    by_category: Dict[str, int]  # Based on input tokens
    by_subject: Dict[str, int]
    
    # Token statistics
    input_token_stats: Dict[str, float]  # min, max, mean, std, p50, p90, p99
    estimated_output_stats: Dict[str, float]
    
    # Category thresholds used
    thresholds: Dict[str, int]
    
    # Post-inference statistics (optional, populated after inference)
    actual_output_stats: Optional[Dict[str, float]] = None
    actual_total_stats: Optional[Dict[str, float]] = None
    by_actual_category: Optional[Dict[str, int]] = None  # Based on actual total tokens
    completed_count: int = 0


class SequenceProfiler:
    """
    Profiles sequences for token length analysis and categorization.
    
    Uses the model's tokenizer to accurately count input tokens and
    estimates output tokens based on configurable heuristics.
    
    Supports two-phase profiling:
    1. Pre-inference: Estimates based on input token count
    2. Post-inference: Actual categorization based on total sequence length
       (input + output tokens)
    """
    
    # Default thresholds for input-only categorization (pre-inference)
    DEFAULT_THRESHOLDS = {
        'short': 256,
        'medium': 512,
        'long': 1024
    }
    
    # Default thresholds for total sequence length categorization (post-inference)
    # These are typically larger since they include output tokens
    DEFAULT_TOTAL_THRESHOLDS = {
        'short': 1000,      # Total tokens <= 1000 (更适合AIME数据集)
        'medium': 3000,     # Total tokens <= 3000
        'long': 8000        # Total tokens <= 8000
    }
    
    # Estimated output tokens based on question complexity
    OUTPUT_TOKEN_ESTIMATES = {
        'short': 128,   # Simple questions
        'medium': 256,  # Moderate complexity
        'long': 384,    # Complex questions requiring detailed reasoning
        'extra_long': 512
    }
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-32B",
        thresholds: Optional[Dict[str, int]] = None,
        total_thresholds: Optional[Dict[str, int]] = None,
        trust_remote_code: bool = True,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize the sequence profiler.
        
        Args:
            model_name: Name or path of the model for tokenizer
            thresholds: Custom thresholds for input-only categorization (pre-inference)
            total_thresholds: Custom thresholds for total sequence categorization (post-inference)
            trust_remote_code: Whether to trust remote code for tokenizer
            cache_dir: Cache directory for tokenizer
        """
        self.model_name = model_name
        self.thresholds = thresholds or self.DEFAULT_THRESHOLDS
        self.total_thresholds = total_thresholds or self.DEFAULT_TOTAL_THRESHOLDS
        self.trust_remote_code = trust_remote_code
        self.cache_dir = cache_dir
        
        self._tokenizer: Optional[AutoTokenizer] = None
        self._sequences: List[SequenceInfo] = []
        self._sequences_by_id: Dict[str, SequenceInfo] = {}  # Index for fast lookup
        self._distribution: Optional[SequenceDistribution] = None
    
    @property
    def tokenizer(self) -> AutoTokenizer:
        """Lazy load tokenizer."""
        if self._tokenizer is None:
            logger.info(f"Loading tokenizer: {self.model_name}")
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=self.trust_remote_code,
                cache_dir=self.cache_dir
            )
        return self._tokenizer
    
    def profile_questions(
        self,
        questions: List[GPQAQuestion],
        show_progress: bool = True
    ) -> List[SequenceInfo]:
        """
        Profile a list of questions for sequence length analysis.
        
        Args:
            questions: List of GPQAQuestion objects
            show_progress: Whether to show progress bar
            
        Returns:
            List of SequenceInfo objects
        """
        logger.info(f"Profiling {len(questions)} questions")
        
        self._sequences = []
        iterator = tqdm(questions, desc="Profiling sequences") if show_progress else questions
        
        for question in iterator:
            seq_info = self._profile_single(question)
            self._sequences.append(seq_info)
            self._sequences_by_id[seq_info.question_id] = seq_info
            
            # Update question object with token count
            question.input_tokens = seq_info.input_tokens
        
        # Calculate distribution
        self._distribution = self._calculate_distribution()
        
        logger.info(f"Profiled {len(self._sequences)} sequences")
        return self._sequences
    
    def _profile_single(self, question: GPQAQuestion) -> SequenceInfo:
        """Profile a single question."""
        # Tokenize the prompt
        tokens = self.tokenizer.encode(question.prompt, add_special_tokens=True)
        input_tokens = len(tokens)
        
        # Categorize based on input length
        category = self._categorize(input_tokens)
        
        # Estimate output tokens
        estimated_output = self.OUTPUT_TOKEN_ESTIMATES.get(category.value, 256)
        
        return SequenceInfo(
            question_id=question.question_id,
            prompt=question.prompt,
            input_tokens=input_tokens,
            estimated_output_tokens=estimated_output,
            total_estimated_tokens=input_tokens + estimated_output,
            category=category.value,
            subject=question.subject
        )
    
    def _categorize(self, token_count: int) -> SequenceCategory:
        """Categorize sequence based on input token count (pre-inference)."""
        if token_count <= self.thresholds['short']:
            return SequenceCategory.SHORT
        elif token_count <= self.thresholds['medium']:
            return SequenceCategory.MEDIUM
        elif token_count <= self.thresholds['long']:
            return SequenceCategory.LONG
        else:
            return SequenceCategory.EXTRA_LONG
    
    def categorize_by_total(self, total_tokens: int) -> SequenceCategory:
        """
        Categorize sequence based on total token count (post-inference).
        
        This method uses total_thresholds which are typically larger
        since they account for both input and output tokens.
        
        Args:
            total_tokens: Total tokens (input + output)
            
        Returns:
            SequenceCategory based on total length
        """
        if total_tokens <= self.total_thresholds['short']:
            return SequenceCategory.SHORT
        elif total_tokens <= self.total_thresholds['medium']:
            return SequenceCategory.MEDIUM
        elif total_tokens <= self.total_thresholds['long']:
            return SequenceCategory.LONG
        else:
            return SequenceCategory.EXTRA_LONG
    
    def _calculate_distribution(self) -> SequenceDistribution:
        """Calculate distribution statistics including actual results if available."""
        if not self._sequences:
            raise ValueError("No sequences profiled yet")
        
        # Count by category (input-based)
        by_category: Dict[str, int] = defaultdict(int)
        by_subject: Dict[str, int] = defaultdict(int)
        by_actual_category: Dict[str, int] = defaultdict(int)
        input_tokens = []
        estimated_output_tokens = []
        actual_output_tokens = []
        actual_total_tokens = []
        completed_count = 0
        
        for seq in self._sequences:
            by_category[seq.category] += 1
            by_subject[seq.subject] += 1
            input_tokens.append(seq.input_tokens)
            estimated_output_tokens.append(seq.estimated_output_tokens)
            
            # Collect actual data if available
            if seq.is_completed:
                completed_count += 1
                if seq.actual_output_tokens is not None:
                    actual_output_tokens.append(seq.actual_output_tokens)
                if seq.actual_total_tokens is not None:
                    actual_total_tokens.append(seq.actual_total_tokens)
                if seq.actual_category:
                    by_actual_category[seq.actual_category] += 1
        
        # Calculate statistics
        input_stats = self._calculate_stats(input_tokens)
        estimated_output_stats = self._calculate_stats(estimated_output_tokens)
        
        # Calculate actual statistics if we have completed sequences
        actual_output_stats = None
        actual_total_stats = None
        if actual_output_tokens:
            actual_output_stats = self._calculate_stats(actual_output_tokens)
        if actual_total_tokens:
            actual_total_stats = self._calculate_stats(actual_total_tokens)
        
        return SequenceDistribution(
            total_sequences=len(self._sequences),
            by_category=dict(by_category),
            by_subject=dict(by_subject),
            input_token_stats=input_stats,
            estimated_output_stats=estimated_output_stats,
            thresholds=self.thresholds,
            actual_output_stats=actual_output_stats,
            actual_total_stats=actual_total_stats,
            by_actual_category=dict(by_actual_category) if by_actual_category else None,
            completed_count=completed_count
        )
    
    def _calculate_stats(self, values: List[int]) -> Dict[str, float]:
        """Calculate statistical metrics for a list of values."""
        arr = np.array(values)
        return {
            'min': float(np.min(arr)),
            'max': float(np.max(arr)),
            'mean': float(np.mean(arr)),
            'std': float(np.std(arr)),
            'p50': float(np.percentile(arr, 50)),
            'p90': float(np.percentile(arr, 90)),
            'p99': float(np.percentile(arr, 99))
        }
    
    def get_sequences(self) -> List[SequenceInfo]:
        """Get profiled sequences."""
        return self._sequences
    
    def get_distribution(self) -> SequenceDistribution:
        """Get sequence distribution statistics."""
        if self._distribution is None:
            raise ValueError("No distribution calculated. Run profile_questions first.")
        return self._distribution
    
    def get_sequences_by_category(self) -> Dict[str, List[SequenceInfo]]:
        """Group sequences by category."""
        by_category: Dict[str, List[SequenceInfo]] = defaultdict(list)
        
        for seq in self._sequences:
            by_category[seq.category].append(seq)
        
        return dict(by_category)
    
    def get_sequences_sorted_by_length(
        self, 
        descending: bool = False,
        use_actual: bool = False
    ) -> List[SequenceInfo]:
        """
        Get sequences sorted by token length.
        
        Args:
            descending: Sort in descending order
            use_actual: If True, sort by actual_total_tokens (for completed sequences)
                       If False, sort by input_tokens
        """
        if use_actual:
            # Filter to completed sequences and sort by actual total
            completed = [s for s in self._sequences if s.is_completed]
            return sorted(
                completed,
                key=lambda x: x.actual_total_tokens or 0,
                reverse=descending
            )
        return sorted(
            self._sequences,
            key=lambda x: x.input_tokens,
            reverse=descending
        )
    
    def filter_by_length(
        self,
        min_tokens: Optional[int] = None,
        max_tokens: Optional[int] = None,
        use_actual_total: bool = False
    ) -> List[SequenceInfo]:
        """
        Filter sequences by token length range.
        
        Args:
            min_tokens: Minimum token count
            max_tokens: Maximum token count  
            use_actual_total: If True, filter by actual_total_tokens
                             If False, filter by input_tokens
        """
        filtered = self._sequences
        
        if use_actual_total:
            # Only consider completed sequences
            filtered = [s for s in filtered if s.is_completed]
            if min_tokens is not None:
                filtered = [s for s in filtered if (s.actual_total_tokens or 0) >= min_tokens]
            if max_tokens is not None:
                filtered = [s for s in filtered if (s.actual_total_tokens or 0) <= max_tokens]
        else:
            if min_tokens is not None:
                filtered = [s for s in filtered if s.input_tokens >= min_tokens]
            if max_tokens is not None:
                filtered = [s for s in filtered if s.input_tokens <= max_tokens]
        
        return filtered
    
    def export_profile(self, output_path: str) -> None:
        """Export profiling results to JSON."""
        data = {
            'model_name': self.model_name,
            'thresholds': self.thresholds,
            'total_thresholds': self.total_thresholds,
            'distribution': asdict(self._distribution) if self._distribution else None,
            'sequences': [asdict(s) for s in self._sequences]
        }
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Exported profile to: {output_path}")
    
    def load_profile(self, input_path: str) -> None:
        """Load profiling results from JSON."""
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.thresholds = data.get('thresholds', self.DEFAULT_THRESHOLDS)
        self.total_thresholds = data.get('total_thresholds', self.DEFAULT_TOTAL_THRESHOLDS)
        self._sequences = [SequenceInfo(**s) for s in data.get('sequences', [])]
        
        # Rebuild index
        self._sequences_by_id = {s.question_id: s for s in self._sequences}
        
        if data.get('distribution'):
            self._distribution = SequenceDistribution(**data['distribution'])
        
        logger.info(f"Loaded profile from: {input_path}")
    
    def print_summary(self) -> None:
        """Print a summary of the profiling results."""
        if not self._distribution:
            print("No profiling results available.")
            return
        
        dist = self._distribution
        
        print("\n" + "=" * 60)
        print("Sequence Profiling Summary")
        print("=" * 60)
        print(f"\nTotal Sequences: {dist.total_sequences}")
        print(f"Completed Sequences: {dist.completed_count}")
        
        print(f"\nInput Thresholds: short<={self.thresholds['short']}, "
              f"medium<={self.thresholds['medium']}, "
              f"long<={self.thresholds['long']}")
        print(f"Total Thresholds: short<={self.total_thresholds['short']}, "
              f"medium<={self.total_thresholds['medium']}, "
              f"long<={self.total_thresholds['long']}")
        
        print("\nDistribution by Category (Input-based):")
        for cat, count in sorted(dist.by_category.items()):
            pct = count / dist.total_sequences * 100
            print(f"  {cat:12s}: {count:4d} ({pct:5.1f}%)")
        
        if dist.by_actual_category:
            print("\nDistribution by Actual Category (Total-based):")
            for cat, count in sorted(dist.by_actual_category.items()):
                pct = count / dist.completed_count * 100 if dist.completed_count > 0 else 0
                print(f"  {cat:12s}: {count:4d} ({pct:5.1f}%)")
        
        print("\nInput Token Statistics:")
        stats = dist.input_token_stats
        print(f"  Min:  {stats['min']:.0f}")
        print(f"  Max:  {stats['max']:.0f}")
        print(f"  Mean: {stats['mean']:.1f}")
        print(f"  Std:  {stats['std']:.1f}")
        print(f"  P50:  {stats['p50']:.0f}")
        print(f"  P90:  {stats['p90']:.0f}")
        print(f"  P99:  {stats['p99']:.0f}")
        
        if dist.actual_total_stats:
            print("\nActual Total Token Statistics (Input + Output):")
            stats = dist.actual_total_stats
            print(f"  Min:  {stats['min']:.0f}")
            print(f"  Max:  {stats['max']:.0f}")
            print(f"  Mean: {stats['mean']:.1f}")
            print(f"  Std:  {stats['std']:.1f}")
            print(f"  P50:  {stats['p50']:.0f}")
            print(f"  P90:  {stats['p90']:.0f}")
            print(f"  P99:  {stats['p99']:.0f}")
        
        print("\nDistribution by Subject:")
        for subject, count in sorted(dist.by_subject.items(), key=lambda x: -x[1])[:5]:
            pct = count / dist.total_sequences * 100
            print(f"  {subject[:30]:30s}: {count:4d} ({pct:5.1f}%)")
        
        print("=" * 60)


    def update_sequence_with_result(
        self,
        question_id: str,
        actual_output_tokens: int,
        generation_time_ms: Optional[float] = None
    ) -> Optional[SequenceInfo]:
        """
        Update a sequence with actual inference results.
        
        This method is called after inference completion to record
        the actual output token count and recategorize based on total length.
        
        Args:
            question_id: ID of the question/sequence
            actual_output_tokens: Actual number of output tokens generated
            generation_time_ms: Time taken for generation in milliseconds
            
        Returns:
            Updated SequenceInfo or None if not found
        """
        seq_info = self._sequences_by_id.get(question_id)
        if seq_info is None:
            logger.warning(f"Sequence not found: {question_id}")
            return None
        
        seq_info.update_with_actual_output(
            actual_output_tokens=actual_output_tokens,
            generation_time_ms=generation_time_ms,
            categorizer=self
        )
        
        logger.debug(
            f"Updated sequence {question_id}: "
            f"input={seq_info.input_tokens}, output={actual_output_tokens}, "
            f"total={seq_info.actual_total_tokens}, category={seq_info.actual_category}"
        )
        
        return seq_info
    
    def batch_update_with_results(
        self,
        results: List[Dict[str, Any]]
    ) -> int:
        """
        Batch update sequences with inference results.
        
        Args:
            results: List of dicts with keys:
                - question_id: str
                - output_tokens: int
                - generation_time_ms: Optional[float]
                
        Returns:
            Number of successfully updated sequences
        """
        updated_count = 0
        for result in results:
            seq = self.update_sequence_with_result(
                question_id=result['question_id'],
                actual_output_tokens=result['output_tokens'],
                generation_time_ms=result.get('generation_time_ms')
            )
            if seq is not None:
                updated_count += 1
        
        # Recalculate distribution with actual data
        if updated_count > 0:
            self._distribution = self._calculate_distribution()
        
        logger.info(f"Updated {updated_count}/{len(results)} sequences with actual results")
        return updated_count
    
    def get_sequence_by_id(self, question_id: str) -> Optional[SequenceInfo]:
        """Get a sequence by its question ID."""
        return self._sequences_by_id.get(question_id)
    
    def get_completed_sequences(self) -> List[SequenceInfo]:
        """Get all sequences that have completed inference."""
        return [s for s in self._sequences if s.is_completed]
    
    def load_actual_tokens_from_profiling(self, profiling_dir: str) -> int:
        """
        从profiling目录加载实际的total_tokens数据
        
        Args:
            profiling_dir: profiling结果目录路径
            
        Returns:
            成功更新的序列数量
        """
        try:
            profiling_path = Path(profiling_dir)
            inference_results_file = profiling_path / "inference_results.json"
            
            if not inference_results_file.exists():
                logger.warning(f"Profiling file not found: {inference_results_file}")
                return 0
            
            with open(inference_results_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
            
            updated_count = 0
            for result in results:
                question_id = result.get("question_id")
                actual_total_tokens = result.get("actual_total_tokens")
                actual_output_tokens = result.get("actual_output_tokens")
                generation_time_ms = result.get("generation_time_ms")
                
                if question_id and actual_total_tokens and actual_output_tokens:
                    seq_info = self.get_sequence_by_id(question_id)
                    if seq_info:
                        seq_info.update_with_actual_output(
                            actual_output_tokens=actual_output_tokens,
                            generation_time_ms=generation_time_ms,
                            categorizer=self
                        )
                        updated_count += 1
                        
                        logger.debug(
                            f"Loaded actual tokens for {question_id}: "
                            f"input={seq_info.input_tokens}, output={actual_output_tokens}, "
                            f"total={seq_info.actual_total_tokens}, category={seq_info.actual_category}"
                        )
            
            # 重新计算分布
            if updated_count > 0:
                self._distribution = self._calculate_distribution()
            
            logger.info(f"Loaded actual tokens from profiling for {updated_count} sequences")
            return updated_count
            
        except Exception as e:
            logger.error(f"Error loading actual tokens from profiling: {e}")
            return 0
    
    def get_sequences_by_actual_category(self) -> Dict[str, List[SequenceInfo]]:
        """
        Group completed sequences by their actual category (based on total tokens).
        
        This is the recommended method for routing decisions after inference.
        """
        by_category: Dict[str, List[SequenceInfo]] = defaultdict(list)
        
        for seq in self._sequences:
            if seq.is_completed and seq.actual_category:
                by_category[seq.actual_category].append(seq)
        
        return dict(by_category)
    
    def get_routing_recommendation(self, question_id: str) -> Dict[str, Any]:
        """
        Get routing recommendation for a sequence.
        
        Returns category and recommended TP configuration based on
        actual total length if available, otherwise estimated.
        
        Args:
            question_id: ID of the question/sequence
            
        Returns:
            Dict with routing recommendation
        """
        seq = self._sequences_by_id.get(question_id)
        if seq is None:
            return {'error': 'Sequence not found'}
        
        # Default TP mapping based on category
        tp_mapping = {
            'short': [1, 2],      # Small TP for short sequences
            'medium': [2, 4],     # Medium TP
            'long': [4, 8],       # High TP for long sequences
            'extra_long': [8]     # Maximum TP for extra long
        }
        
        category = seq.get_routing_category()
        total_tokens = seq.get_total_tokens()
        
        return {
            'question_id': question_id,
            'category': category,
            'total_tokens': total_tokens,
            'is_actual': seq.is_completed,
            'recommended_tp': tp_mapping.get(category, [4]),
            'input_tokens': seq.input_tokens,
            'output_tokens': seq.actual_output_tokens if seq.is_completed else seq.estimated_output_tokens
        }


def profile_dataset(
    loader: GPQADatasetLoader,
    config: Dict[str, Any]
) -> Tuple[SequenceProfiler, List[SequenceInfo]]:
    """
    Convenience function to profile a dataset.
    
    Args:
        loader: GPQADatasetLoader instance
        config: Configuration dictionary
        
    Returns:
        Tuple of (SequenceProfiler, List[SequenceInfo])
    """
    model_config = config.get('model', {})
    scheduling_config = config.get('scheduling', {})
    
    profiler = SequenceProfiler(
        model_name=model_config.get('name', 'Qwen/Qwen3-32B'),
        thresholds=scheduling_config.get('length_thresholds'),
        total_thresholds=scheduling_config.get('total_length_thresholds'),
        trust_remote_code=model_config.get('trust_remote_code', True)
    )
    
    sequences = profiler.profile_questions(loader.get_questions())
    
    return profiler, sequences
