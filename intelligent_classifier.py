#!/usr/bin/env python3
"""
SCOOT 智能序列分类模块
集成到现有调度器框架中
"""

import json
import logging
from typing import Dict, List, Optional, Tuple
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)

class SequenceCategory(Enum):
    SHORT = "short"
    MEDIUM = "medium" 
    LONG = "long"
    EXTRA_LONG = "extra_long"

class IntelligentSequenceClassifier:
    """智能序列分类器"""
    
    def __init__(self, config_path: Optional[str] = None, profiling_dir: Optional[str] = None):
        self.config_path = config_path
        self.profiling_dir = profiling_dir
        self.thresholds = {}
        self.actual_tokens_map = {}  # 存储 profiling 数据映射
        
        # 加载配置和数据
        self._load_configuration()
        self._load_profiling_data()
    
    def _load_configuration(self):
        """加载分桶配置"""
        if self.config_path:
            config_file = Path(self.config_path)
            if config_file.exists():
                with open(config_file, 'r') as f:
                    config = json.load(f)
                    self.thresholds = config.get('scheduling', {}).get('length_thresholds', {})
                    logger.info(f"Loaded thresholds from config: {self.thresholds}")
            else:
                logger.warning(f"Config file not found: {self.config_path}")
        
        # 如果没有配置文件，使用默认值
        if not self.thresholds:
            self.thresholds = {
                'short': 512,
                'medium': 2048,
                'long': 8192,
                'extra_long': -1  # 无穷大
            }
            logger.info("Using default thresholds")
    
    def _load_profiling_data(self):
        """加载 profiling 数据"""
        if self.profiling_dir:
            profile_file = Path(self.profiling_dir) / "sequence_profile.json"
            if profile_file.exists():
                with open(profile_file, 'r') as f:
                    profile_data = json.load(f)
                    
                # 构建 request_id 到 actual_total_tokens 的映射
                for request_id, seq_info in profile_data.items():
                    if 'actual_total_tokens' in seq_info:
                        self.actual_tokens_map[request_id] = seq_info['actual_total_tokens']
                        
                logger.info(f"Loaded profiling data for {len(self.actual_tokens_map)} sequences")
            else:
                logger.warning(f"Profiling file not found: {profile_file}")
    
    def categorize_sequence(self, request_id: str, input_tokens: int, 
                          estimated_output_tokens: Optional[int] = None) -> Tuple[SequenceCategory, int]:
        """
        对序列进行分类
        
        Args:
            request_id: 请求ID
            input_tokens: 输入token数量
            estimated_output_tokens: 预估输出token数量（可选）
            
        Returns:
            (分类结果, 实际总token数)
        """
        # 优先使用 profiling 数据中的 actual_total_tokens
        actual_total_tokens = self.actual_tokens_map.get(request_id)
        
        if actual_total_tokens is not None:
            total_tokens = actual_total_tokens
            classification_method = "actual_total_tokens"
        else:
            # 回退到估算值
            output_tokens = estimated_output_tokens or input_tokens  # 简单估算
            total_tokens = input_tokens + output_tokens
            classification_method = "estimated"
            
        # 根据阈值进行分类
        if total_tokens <= self.thresholds['short']:
            category = SequenceCategory.SHORT
        elif total_tokens <= self.thresholds['medium']:
            category = SequenceCategory.MEDIUM
        elif total_tokens <= self.thresholds['long'] or self.thresholds['long'] == -1:
            category = SequenceCategory.LONG
        else:
            category = SequenceCategory.EXTRA_LONG
            
        # 记录分类日志
        logger.info({
            "type": "SEQUENCE_CLASSIFICATION_RESULT",
            "request_id": request_id,
            "input_tokens": input_tokens,
            "actual_total_tokens": actual_total_tokens,
            "estimated_total_tokens": total_tokens if actual_total_tokens is None else None,
            "sequence_category": category.value,
            "classification_method": classification_method,
            "has_profiling_data": actual_total_tokens is not None
        })
        
        return category, total_tokens
    
    def get_routing_recommendation(self, category: SequenceCategory) -> Dict:
        """
        基于序列分类给出路由建议
        这里可以根据你的异构实例配置调整
        """
        routing_map = {
            SequenceCategory.SHORT: {
                "recommended_tp": 1,
                "instance_type": "low_tp",
                "priority": "high"
            },
            SequenceCategory.MEDIUM: {
                "recommended_tp": 2,
                "instance_type": "medium_tp", 
                "priority": "normal"
            },
            SequenceCategory.LONG: {
                "recommended_tp": 4,
                "instance_type": "high_tp",
                "priority": "low"
            },
            SequenceCategory.EXTRA_LONG: {
                "recommended_tp": 8,
                "instance_type": "highest_tp",
                "priority": "lowest"
            }
        }
        
        return routing_map.get(category, routing_map[SequenceCategory.MEDIUM])
    
    def update_thresholds(self, new_thresholds: Dict[str, int]):
        """动态更新分桶阈值"""
        self.thresholds.update(new_thresholds)
        logger.info(f"Updated thresholds: {self.thresholds}")

# 使用示例和集成代码
def integrate_with_scoot_scheduler():
    """
    展示如何将智能分类器集成到 SCOOT 调度器中
    """
    
    # 示例：在 benchmark_serving.py 中的使用
    classifier = IntelligentSequenceClassifier(
        config_path="./bucket_config.json",
        profiling_dir="./profiling_results"
    )
    
    # 模拟调度器中的序列处理
    sample_requests = [
        ("req_001", 100, 400),  # 短序列
        ("req_002", 800, 1500), # 中等序列
        ("req_003", 2000, 7000), # 长序列
        ("req_004", 3000, 15000) # 超长序列
    ]
    
    for request_id, input_tokens, est_output in sample_requests:
        # 分类序列
        category, total_tokens = classifier.categorize_sequence(
            request_id, input_tokens, est_output
        )
        
        # 获取路由建议
        routing_info = classifier.get_routing_recommendation(category)
        
        print(f"Request {request_id}:")
        print(f"  - Total tokens: {total_tokens}")
        print(f"  - Category: {category.value}")
        print(f"  - Recommended TP: {routing_info['recommended_tp']}")
        print(f"  - Instance type: {routing_info['instance_type']}")
        print()

# 配置文件模板
BUCKET_CONFIG_TEMPLATE = {
    "scheduling": {
        "length_thresholds": {
            "short": 512,
            "medium": 2048,
            "long": 8192,
            "extra_long": -1
        },
        "bucket_names": ["short", "medium", "long", "extra_long"],
        "classification_method": "actual_total_tokens"
    },
    "routing_policy": {
        "short": {"tp_degree": 1, "instance_pool": "tp1_instances"},
        "medium": {"tp_degree": 2, "instance_pool": "tp2_instances"},
        "long": {"tp_degree": 4, "instance_pool": "tp4_instances"},
        "extra_long": {"tp_degree": 8, "instance_pool": "tp8_instances"}
    }
}

if __name__ == "__main__":
    # 运行集成示例
    integrate_with_scoot_scheduler()