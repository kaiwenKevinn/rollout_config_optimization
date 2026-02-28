#!/usr/bin/env python3
"""
SCOOT 序列分桶优化器
基于实际序列长度分布自动计算最优分桶阈值
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from pathlib import Path
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BucketOptimizer:
    """序列分桶优化器"""
    
    def __init__(self, profiling_dir: str):
        self.profiling_dir = Path(profiling_dir)
        self.sequence_data = []
        self.thresholds = {}
        
    def load_profiling_data(self) -> List[Dict]:
        """加载 profiling 数据"""
        profile_file = self.profiling_dir / "sequence_profile.json"
        if not profile_file.exists():
            raise FileNotFoundError(f"Profiling file not found: {profile_file}")
            
        with open(profile_file, 'r') as f:
            data = json.load(f)
            
        # 提取 actual_total_tokens
        self.sequence_data = [
            seq['actual_total_tokens'] 
            for seq in data.values() 
            if 'actual_total_tokens' in seq
        ]
        
        logger.info(f"Loaded {len(self.sequence_data)} sequences from profiling data")
        return self.sequence_data
    
    def analyze_distribution(self) -> Dict:
        """分析序列长度分布"""
        if not self.sequence_data:
            raise ValueError("No sequence data loaded")
            
        data_array = np.array(self.sequence_data)
        
        stats = {
            'count': len(data_array),
            'mean': np.mean(data_array),
            'std': np.std(data_array),
            'min': np.min(data_array),
            'max': np.max(data_array),
            'percentiles': {
                '25%': np.percentile(data_array, 25),
                '50%': np.percentile(data_array, 50),  # 中位数
                '75%': np.percentile(data_array, 75),
                '90%': np.percentile(data_array, 90),
                '95%': np.percentile(data_array, 95),
                '99%': np.percentile(data_array, 99),
            }
        }
        
        logger.info("Sequence Length Distribution Analysis:")
        logger.info(f"  Count: {stats['count']}")
        logger.info(f"  Range: {stats['min']} - {stats['max']} tokens")
        logger.info(f"  Mean: {stats['mean']:.1f} tokens")
        logger.info(f"  Std Dev: {stats['std']:.1f} tokens")
        logger.info("  Percentiles:")
        for p, value in stats['percentiles'].items():
            logger.info(f"    {p}: {value:.1f} tokens")
            
        return stats
    
    def calculate_optimal_thresholds(self, method: str = 'percentile') -> Dict[str, int]:
        """
        计算最优分桶阈值
        
        Args:
            method: 'percentile' (基于百分位数) 或 'kmeans' (K-means聚类)
        """
        if not self.sequence_data:
            raise ValueError("No sequence data loaded")
            
        data_array = np.array(self.sequence_data)
        
        if method == 'percentile':
            # 基于百分位数的方法（推荐）
            self.thresholds = {
                'short': int(np.percentile(data_array, 25)),      # 25% 分位数
                'medium': int(np.percentile(data_array, 75)),     # 75% 分位数
                'long': int(np.percentile(data_array, 95)),       # 95% 分位数
                'extra_long': float('inf')
            }
        elif method == 'kmeans':
            # K-means 聚类方法
            from sklearn.cluster import KMeans
            
            # 尝试不同的聚类数
            inertias = []
            k_range = range(2, 6)
            
            for k in k_range:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                kmeans.fit(data_array.reshape(-1, 1))
                inertias.append(kmeans.inertia_)
            
            # 选择肘部点
            optimal_k = 4  # 默认使用4个桶
            kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
            kmeans.fit(data_array.reshape(-1, 1))
            
            # 获取聚类中心并排序
            centers = sorted(kmeans.cluster_centers_.flatten())
            
            self.thresholds = {
                'short': int(centers[0]),
                'medium': int(centers[1]),
                'long': int(centers[2]),
                'extra_long': float('inf')
            }
        else:
            raise ValueError(f"Unknown method: {method}")
            
        logger.info(f"Optimal thresholds calculated using {method} method:")
        for bucket, threshold in self.thresholds.items():
            if threshold != float('inf'):
                logger.info(f"  {bucket}: ≤ {threshold} tokens")
            else:
                logger.info(f"  {bucket}: > {self.thresholds['long']} tokens")
                
        return self.thresholds
    
    def evaluate_bucket_balance(self) -> Dict[str, int]:
        """评估各桶的负载平衡情况"""
        if not self.thresholds or not self.sequence_data:
            raise ValueError("Thresholds or sequence data not available")
            
        bucket_counts = {'short': 0, 'medium': 0, 'long': 0, 'extra_long': 0}
        
        for length in self.sequence_data:
            if length <= self.thresholds['short']:
                bucket_counts['short'] += 1
            elif length <= self.thresholds['medium']:
                bucket_counts['medium'] += 1
            elif length <= self.thresholds['long']:
                bucket_counts['long'] += 1
            else:
                bucket_counts['extra_long'] += 1
                
        total = len(self.sequence_data)
        logger.info("Bucket Distribution:")
        for bucket, count in bucket_counts.items():
            percentage = (count / total) * 100
            logger.info(f"  {bucket}: {count} sequences ({percentage:.1f}%)")
            
        return bucket_counts
    
    def visualize_distribution(self, output_path: str = None):
        """可视化序列长度分布和分桶结果"""
        if not self.sequence_data:
            raise ValueError("No sequence data loaded")
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 左图：直方图显示分布
        data_array = np.array(self.sequence_data)
        ax1.hist(data_array, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_xlabel('Total Tokens (Prompt + Output)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Sequence Length Distribution')
        ax1.grid(True, alpha=0.3)
        
        # 添加分桶阈值线
        colors = ['red', 'orange', 'green']
        labels = ['Short', 'Medium', 'Long']
        thresholds_list = [
            self.thresholds['short'], 
            self.thresholds['medium'], 
            self.thresholds['long']
        ]
        
        for i, (thresh, color, label) in enumerate(zip(thresholds_list, colors, labels)):
            ax1.axvline(thresh, color=color, linestyle='--', linewidth=2, 
                       label=f'{label} Threshold ({thresh})')
        
        ax1.legend()
        
        # 右图：累积分布函数
        sorted_data = np.sort(data_array)
        cumulative = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        ax2.plot(sorted_data, cumulative, linewidth=2, color='blue')
        ax2.set_xlabel('Total Tokens')
        ax2.set_ylabel('Cumulative Probability')
        ax2.set_title('Cumulative Distribution Function')
        ax2.grid(True, alpha=0.3)
        
        # 添加百分位数标记
        percentiles = [25, 50, 75, 90, 95, 99]
        for p in percentiles:
            value = np.percentile(data_array, p)
            ax2.axvline(value, color='red', linestyle=':', alpha=0.7)
            ax2.text(value, 0.02, f'{p}%', rotation=90, fontsize=8)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Distribution plot saved to {output_path}")
        else:
            plt.show()
    
    def generate_config_file(self, output_path: str = None) -> Dict:
        """生成配置文件"""
        config = {
            "scheduling": {
                "length_thresholds": {
                    "short": self.thresholds['short'],
                    "medium": self.thresholds['medium'],
                    "long": self.thresholds['long'],
                    "extra_long": -1  # 表示无穷大
                },
                "bucket_names": ["short", "medium", "long", "extra_long"],
                "classification_method": "actual_total_tokens"
            },
            "metadata": {
                "profiling_source": str(self.profiling_dir),
                "sequence_count": len(self.sequence_data),
                "optimization_method": "percentile_based",
                "created_at": str(np.datetime64('now'))
            }
        }
        
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(config, f, indent=2)
            logger.info(f"Configuration saved to {output_path}")
            
        return config

def main():
    parser = argparse.ArgumentParser(description='SCOOT Sequence Bucket Optimizer')
    parser.add_argument('--profiling-dir', required=True, 
                       help='Path to profiling directory containing sequence_profile.json')
    parser.add_argument('--method', choices=['percentile', 'kmeans'], 
                       default='percentile', help='Threshold calculation method')
    parser.add_argument('--output-config', help='Output configuration file path')
    parser.add_argument('--output-plot', help='Output visualization plot path')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # 初始化优化器
        optimizer = BucketOptimizer(args.profiling_dir)
        
        # 加载和分析数据
        optimizer.load_profiling_data()
        stats = optimizer.analyze_distribution()
        
        # 计算最优阈值
        thresholds = optimizer.calculate_optimal_thresholds(method=args.method)
        
        # 评估负载平衡
        bucket_counts = optimizer.evaluate_bucket_balance()
        
        # 生成可视化
        if args.output_plot:
            optimizer.visualize_distribution(args.output_plot)
            
        # 生成配置文件
        if args.output_config:
            config = optimizer.generate_config_file(args.output_config)
            
        logger.info("✅ Bucket optimization completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Error during bucket optimization: {e}")
        raise

if __name__ == "__main__":
    main()