"""
Profiling Results Loader
用于加载和解析profiling阶段生成的结果文件
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SequenceStats:
    """序列统计信息"""
    total_sequences: int
    completed_sequences: int
    by_category_input: Dict[str, int]  # 输入token长度分类
    by_category_total: Dict[str, int]  # 总token长度分类（输入+输出）
    input_token_stats: Dict[str, float]  # 输入token统计
    actual_output_stats: Optional[Dict[str, float]]  # 实际输出token统计
    actual_total_stats: Optional[Dict[str, float]]  # 实际总token统计
    thresholds_input: Dict[str, int]  # 输入长度阈值
    thresholds_total: Dict[str, int]  # 总长度阈值


@dataclass
class DatasetStats:
    """数据集统计信息"""
    total_questions: int
    subjects: List[str]
    avg_question_length: float
    min_question_length: int
    max_question_length: int
    problem_types: Optional[Dict[str, int]] = None
    difficulty_distribution: Optional[Dict[str, int]] = None
    avg_answer_length: Optional[float] = None


class ProfilingResultsLoader:
    """Profiling结果加载器"""
    
    def __init__(self, profiling_dir: str):
        """
        初始化加载器
        
        Args:
            profiling_dir: profiling结果目录路径
        """
        self.profiling_dir = Path(profiling_dir)
        self.sequence_distribution_file = self.profiling_dir / "sequence_distribution.json"
        self.dataset_stats_file = self.profiling_dir / "dataset_stats.json"
        self.inference_results_file = self.profiling_dir / "inference_results.json"
        
        # 检查必要文件是否存在
        self._validate_files()
    
    def _validate_files(self):
        """验证必要的结果文件是否存在"""
        required_files = [self.sequence_distribution_file, self.dataset_stats_file]
        for file_path in required_files:
            if not file_path.exists():
                raise FileNotFoundError(f"Required profiling file not found: {file_path}")
    
    def load_sequence_stats(self) -> SequenceStats:
        """加载序列统计信息"""
        with open(self.sequence_distribution_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return SequenceStats(
            total_sequences=data["total_sequences"],
            completed_sequences=data["completed_sequences"],
            by_category_input=data["by_category_input_based"],
            by_category_total=data.get("by_category_total_based", {}),
            input_token_stats=data["input_token_stats"],
            actual_output_stats=data.get("actual_output_stats"),
            actual_total_stats=data.get("actual_total_stats"),
            thresholds_input=data["thresholds_input"],
            thresholds_total=data["thresholds_total"]
        )
    
    def load_dataset_stats(self) -> DatasetStats:
        """加载数据集统计信息"""
        with open(self.dataset_stats_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return DatasetStats(
            total_questions=data["total_questions"],
            subjects=data["subjects"],
            avg_question_length=data["avg_question_length"],
            min_question_length=data["min_question_length"],
            max_question_length=data["max_question_length"],
            problem_types=data.get("problem_types"),
            difficulty_distribution=data.get("difficulty_distribution"),
            avg_answer_length=data.get("avg_answer_length")
        )
    
    def load_inference_results(self) -> List[Dict[str, Any]]:
        """加载推理结果详情"""
        if not self.inference_results_file.exists():
            logger.warning(f"Inference results file not found: {self.inference_results_file}")
            return []
        
        with open(self.inference_results_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def get_optimal_thresholds(self, use_actual: bool = True) -> Dict[str, int]:
        """
        基于实际数据获取最优的长度阈值
        
        Args:
            use_actual: 是否使用实际的总长度（包含输出）作为阈值基础
            
        Returns:
            优化后的阈值字典
        """
        seq_stats = self.load_sequence_stats()
        
        if use_actual and seq_stats.actual_total_stats:
            # 使用实际总长度统计数据
            stats = seq_stats.actual_total_stats
            thresholds = seq_stats.thresholds_total
        else:
            # 使用输入长度统计数据
            stats = seq_stats.input_token_stats
            thresholds = seq_stats.thresholds_input
        
        # 基于分位数调整阈值
        p50 = stats.get("p50", 0)
        p90 = stats.get("p90", 0)
        max_val = stats.get("max", 0)
        
        # 调整阈值使其更符合实际分布
        adjusted_thresholds = {
            "short": min(int(p50 * 1.2), thresholds.get("short", 5000)),
            "medium": min(int(p90 * 1.1), thresholds.get("medium", 10000)),
            "long": min(int(max_val * 0.9), thresholds.get("long", 15000))
        }
        
        logger.info(f"Optimized thresholds based on {'actual' if use_actual else 'input'} data:")
        logger.info(f"  short: <= {adjusted_thresholds['short']} tokens")
        logger.info(f"  medium: <= {adjusted_thresholds['medium']} tokens") 
        logger.info(f"  long: > {adjusted_thresholds['medium']} tokens")
        
        return adjusted_thresholds
    
    def get_representative_sequences(self, count_per_category: int = 5) -> Dict[str, List[Dict[str, Any]]]:
        """
        获取每个类别的代表性序列
        
        Args:
            count_per_category: 每个类别选择的序列数量
            
        Returns:
            按类别组织的代表性序列列表
        """
        inference_results = self.load_inference_results()
        if not inference_results:
            logger.warning("No inference results available for representative sequence selection")
            return {}
        
        # 按类别分组
        by_category = {}
        for result in inference_results:
            category = result.get("category_total_based", result.get("category_input_based", "unknown"))
            if category not in by_category:
                by_category[category] = []
            by_category[category].append(result)
        
        # 从每个类别中选择代表性序列
        representatives = {}
        for category, sequences in by_category.items():
            # 按总长度排序，选择中间值附近的序列
            sorted_seqs = sorted(sequences, key=lambda x: x.get("actual_total_tokens", 0))
            if len(sorted_seqs) <= count_per_category:
                representatives[category] = sorted_seqs
            else:
                # 选择均匀分布的代表性序列
                step = len(sorted_seqs) // count_per_category
                representatives[category] = [sorted_seqs[i * step] for i in range(count_per_category)]
        
        return representatives
    
    def print_analysis_report(self):
        """打印详细的分析报告"""
        seq_stats = self.load_sequence_stats()
        dataset_stats = self.load_dataset_stats()
        
        print("\n" + "=" * 80)
        print("PROFILING RESULTS ANALYSIS REPORT")
        print("=" * 80)
        
        print(f"\nDataset Summary:")
        print(f"  Total Questions: {dataset_stats.total_questions}")
        print(f"  Average Question Length: {dataset_stats.avg_question_length:.1f} chars")
        print(f"  Length Range: {dataset_stats.min_question_length} - {dataset_stats.max_question_length} chars")
        
        if dataset_stats.problem_types:
            print(f"  Problem Types: {dataset_stats.problem_types}")
        
        print(f"\nSequence Analysis (Input-based):")
        print(f"  Total Sequences: {seq_stats.total_sequences}")
        print(f"  Completed: {seq_stats.completed_sequences}")
        print(f"  Input Thresholds: short<={seq_stats.thresholds_input['short']}, "
              f"medium<={seq_stats.thresholds_input['medium']}, "
              f"long<={seq_stats.thresholds_input['long']}")
        
        if seq_stats.actual_total_stats:
            print(f"\nSequence Analysis (Total-based, including output):")
            print(f"  Total Thresholds: short<={seq_stats.thresholds_total['short']}, "
                  f"medium<={seq_stats.thresholds_total['medium']}, "
                  f"long<={seq_stats.thresholds_total['long']}")
        
        print(f"\nDistribution by Input Category:")
        for category, count in sorted(seq_stats.by_category_input.items()):
            percentage = (count / seq_stats.total_sequences) * 100
            print(f"  {category:12}: {count:3} ({percentage:5.1f}%)")
        
        if seq_stats.by_category_total:
            print(f"\nDistribution by Total Category (Input + Output):")
            for category, count in sorted(seq_stats.by_category_total.items()):
                percentage = (count / seq_stats.total_sequences) * 100
                print(f"  {category:12}: {count:3} ({percentage:5.1f}%)")
        
        print(f"\nInput Token Statistics:")
        stats = seq_stats.input_token_stats
        print(f"  Min: {stats['min']:6.0f}  Max: {stats['max']:6.0f}  Mean: {stats['mean']:6.1f}")
        print(f"  P50: {stats['p50']:6.0f}  P90: {stats['p90']:6.0f}  P99: {stats['p99']:6.0f}")
        
        if seq_stats.actual_total_stats:
            print(f"\nActual Total Token Statistics (Input + Output):")
            stats = seq_stats.actual_total_stats
            print(f"  Min: {stats['min']:6.0f}  Max: {stats['max']:6.0f}  Mean: {stats['mean']:6.1f}")
            print(f"  P50: {stats['p50']:6.0f}  P90: {stats['p90']:6.0f}  P99: {stats['p99']:6.0f}")
        
        print("=" * 80)


def create_adaptive_scenarios_from_profiling(
    profiling_dir: str,
    total_gpus: int = 8,
    exclude_tp1: bool = False
) -> tuple[list, list]:
    """
    基于profiling结果自动创建适应性的同构和异构测试场景
    
    Args:
        profiling_dir: profiling结果目录
        total_gpus: 总GPU数量
        
    Returns:
        tuple: (homogeneous_scenarios, heterogeneous_scenarios)
    """
    loader = ProfilingResultsLoader(profiling_dir)
    seq_stats = loader.load_sequence_stats()
    
    # 获取优化的阈值
    optimal_thresholds = loader.get_optimal_thresholds(use_actual=True)
    
    homogeneous_scenarios = []
    heterogeneous_scenarios = []
    
    # 分析序列分布来决定测试重点
    total_by_category = seq_stats.by_category_total or seq_stats.by_category_input
    dominant_categories = sorted(total_by_category.items(), key=lambda x: x[1], reverse=True)
    
    # 创建同构场景 - 针对主要的序列长度类别
    primary_category = dominant_categories[0][0] if dominant_categories else "medium"
    
    # TP=1场景：适合短序列（除非明确排除）
    if not exclude_tp1 and (primary_category in ["short"] or total_by_category.get("short", 0) > 0):
        homogeneous_scenarios.append({
            "tp": 1,
            "instances": min(4, total_gpus),  # 最多4个实例
            "description": f"Optimized for short sequences ({total_by_category.get('short', 0)} sequences)"
        })
    
    # TP=2场景：适合中等序列  
    if primary_category in ["medium", "short"] or total_by_category.get("medium", 0) > 0:
        if total_gpus >= 4:
            homogeneous_scenarios.append({
                "tp": 2,
                "instances": min(2, total_gpus // 2),
                "description": f"Optimized for medium sequences ({total_by_category.get('medium', 0)} sequences)"
            })
    
    # TP=4场景：适合长序列
    if primary_category in ["long", "medium"] or total_by_category.get("long", 0) > 0:
        if total_gpus >= 4:
            homogeneous_scenarios.append({
                "tp": 4,
                "instances": min(1, total_gpus // 4),
                "description": f"Optimized for long sequences ({total_by_category.get('long', 0)} sequences)"
            })
    
    # 创建异构场景 - 基于序列分布智能分配
    if len(dominant_categories) >= 2:
        # 场景1：混合配置，根据主要类别分配
        hetero_config_1 = []
        remaining_gpus = list(range(total_gpus))
        
        # 为主导类别分配最适合的TP配置
        for category, count in dominant_categories[:3]:  # 最多考虑前3个类别
            if not remaining_gpus:
                break
                
            if category == "short" and len(remaining_gpus) >= 1:
                # 短序列用TP=1
                hetero_config_1.append({
                    "tp": 1,
                    "gpus": [remaining_gpus.pop(0)],
                    "description": f"Short sequences handler ({count} sequences)"
                })
            elif category == "medium" and len(remaining_gpus) >= 2:
                # 中等序列用TP=2
                gpus = [remaining_gpus.pop(0), remaining_gpus.pop(0)]
                hetero_config_1.append({
                    "tp": 2,
                    "gpus": gpus,
                    "description": f"Medium sequences handler ({count} sequences)"
                })
            elif category in ["long", "extra_long"] and len(remaining_gpus) >= 4:
                # 长序列用TP=4
                gpus = [remaining_gpus.pop(0) for _ in range(4)]
                hetero_config_1.append({
                    "tp": 4,
                    "gpus": gpus,
                    "description": f"Long sequences handler ({count} sequences)"
                })
        
        if hetero_config_1:
            heterogeneous_scenarios.append({
                "name": "adaptive_hetero_1",
                "instances": hetero_config_1,
                "description": "Adaptive heterogeneous configuration based on sequence distribution",
                "length_thresholds": optimal_thresholds,
                "routing_rules": {
                    "short": [1, 2],
                    "medium": [2, 4],
                    "long": [4],
                    "extra_long": [4]
                }
            })
    
    # 场景2：平衡配置
    if total_gpus >= 8:
        heterogeneous_scenarios.append({
            "name": "balanced_hetero",
            "instances": [
                {"tp": 4, "gpus": [0, 1, 2, 3], "description": "Long sequences handler"},
                {"tp": 2, "gpus": [4, 5], "description": "Medium sequences handler"},
                {"tp": 1, "gpus": [6], "description": "Short sequences handler"},
                {"tp": 1, "gpus": [7], "description": "Short sequences handler"}
            ],
            "description": "Balanced heterogeneous configuration",
            "length_thresholds": optimal_thresholds,
            "routing_rules": {
                "short": [1, 2],
                "medium": [2, 4],
                "long": [4],
                "extra_long": [4]
            }
        })
    
    return homogeneous_scenarios, heterogeneous_scenarios