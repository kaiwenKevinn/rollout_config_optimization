#!/usr/bin/env python3
"""
同构vs异构配置对比分析脚本

此脚本用于对比分析同构和异构配置的性能差异，
基于profiling结果提供定量的性能评估。
"""

import json
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_benchmark_results(results_dir: str):
    """加载基准测试结果"""
    results_path = Path(results_dir)
    if not results_path.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")
    
    # 查找metrics.json文件
    metrics_files = list(results_path.rglob("metrics.json"))
    if not metrics_files:
        raise FileNotFoundError(f"No metrics.json found in {results_dir}")
    
    results = {}
    for metrics_file in metrics_files:
        with open(metrics_file, 'r') as f:
            data = json.load(f)
        
        # 提取场景名称
        scenario_name = metrics_file.parent.name
        results[scenario_name] = data
    
    return results

def calculate_performance_metrics(results: dict):
    """计算性能指标"""
    metrics_summary = {}
    
    for scenario_name, data in results.items():
        runs = data.get("runs", [])
        if not runs:
            continue
            
        # 提取关键指标
        throughputs = [run.get("throughput", 0) for run in runs if run.get("success", False)]
        latencies = [run.get("avg_latency_ms", 0) for run in runs if run.get("success", False)]
        success_rates = [run.get("success_rate", 0) for run in runs if "success_rate" in run]
        
        if throughputs:
            metrics_summary[scenario_name] = {
                "avg_throughput": np.mean(throughputs),
                "std_throughput": np.std(throughputs),
                "avg_latency": np.mean(latencies) if latencies else 0,
                "std_latency": np.std(latencies) if latencies else 0,
                "success_rate": np.mean(success_rates) if success_rates else 1.0,
                "num_successful_runs": len([r for r in runs if r.get("success", False)]),
                "total_runs": len(runs)
            }
    
    return metrics_summary

def compare_homogeneous_vs_heterogeneous(homo_results: dict, hetero_results: dict):
    """对比同构和异构配置性能"""
    
    homo_metrics = calculate_performance_metrics(homo_results)
    hetero_metrics = calculate_performance_metrics(hetero_results)
    
    print("=" * 80)
    print("HOMOGENEOUS VS HETEROGENEOUS CONFIGURATION COMPARISON")
    print("=" * 80)
    
    print("\n同构配置性能:")
    print("-" * 40)
    for name, metrics in homo_metrics.items():
        print(f"{name:20}: {metrics['avg_throughput']:8.2f} ± {metrics['std_throughput']:6.2f} tokens/s")
        print(f"{'':20}  Latency: {metrics['avg_latency']:6.2f} ± {metrics['std_latency']:6.2f} ms")
        print(f"{'':20}  Success: {metrics['success_rate']:6.1%} ({metrics['num_successful_runs']}/{metrics['total_runs']})")
    
    print("\n异构配置性能:")
    print("-" * 40)
    for name, metrics in hetero_metrics.items():
        print(f"{name:20}: {metrics['avg_throughput']:8.2f} ± {metrics['std_throughput']:6.2f} tokens/s")
        print(f"{'':20}  Latency: {metrics['avg_latency']:6.2f} ± {metrics['std_latency']:6.2f} ms")
        print(f"{'':20}  Success: {metrics['success_rate']:6.1%} ({metrics['num_successful_runs']}/{metrics['total_runs']})")
    
    # 计算总体性能指标
    if homo_metrics and hetero_metrics:
        homo_avg_throughput = np.mean([m['avg_throughput'] for m in homo_metrics.values()])
        hetero_avg_throughput = np.mean([m['avg_throughput'] for m in hetero_metrics.values()])
        
        homo_avg_latency = np.mean([m['avg_latency'] for m in homo_metrics.values() if m['avg_latency'] > 0])
        hetero_avg_latency = np.mean([m['avg_latency'] for m in hetero_metrics.values() if m['avg_latency'] > 0])
        
        print("\n总体对比:")
        print("-" * 40)
        throughput_improvement = ((hetero_avg_throughput - homo_avg_throughput) / homo_avg_throughput) * 100
        print(f"平均吞吐量提升: {throughput_improvement:+.2f}%")
        print(f"同构配置: {homo_avg_throughput:.2f} tokens/s")
        print(f"异构配置: {hetero_avg_throughput:.2f} tokens/s")
        
        if homo_avg_latency > 0 and hetero_avg_latency > 0:
            latency_improvement = ((homo_avg_latency - hetero_avg_latency) / homo_avg_latency) * 100
            print(f"平均延迟改善: {latency_improvement:+.2f}%")
            print(f"同构配置: {homo_avg_latency:.2f} ms")
            print(f"异构配置: {hetero_avg_latency:.2f} ms")

def plot_performance_comparison(homo_results: dict, hetero_results: dict, output_dir: str):
    """绘制性能对比图表"""
    
    homo_metrics = calculate_performance_metrics(homo_results)
    hetero_metrics = calculate_performance_metrics(hetero_results)
    
    if not homo_metrics or not hetero_metrics:
        print("警告: 没有足够的数据生成图表")
        return
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 吞吐量对比图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 吞吐量柱状图
    all_scenarios = list(homo_metrics.keys()) + list(hetero_metrics.keys())
    homo_values = [homo_metrics.get(name, {}).get('avg_throughput', 0) for name in homo_metrics.keys()]
    hetero_values = [hetero_metrics.get(name, {}).get('avg_throughput', 0) for name in hetero_metrics.keys()]
    
    x = np.arange(len(all_scenarios))
    width = 0.35
    
    ax1.bar(x[:len(homo_values)], homo_values, width, label='Homogeneous', alpha=0.8)
    ax1.bar(x[len(homo_values):], hetero_values, width, label='Heterogeneous', alpha=0.8)
    
    ax1.set_xlabel('Configuration')
    ax1.set_ylabel('Throughput (tokens/s)')
    ax1.set_title('Throughput Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(all_scenarios, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 延迟对比图
    homo_latencies = [homo_metrics.get(name, {}).get('avg_latency', 0) for name in homo_metrics.keys()]
    hetero_latencies = [hetero_metrics.get(name, {}).get('avg_latency', 0) for name in hetero_metrics.keys()]
    
    ax2.bar(x[:len(homo_latencies)], homo_latencies, width, label='Homogeneous', alpha=0.8, color='orange')
    ax2.bar(x[len(homo_latencies):], hetero_latencies, width, label='Heterogeneous', alpha=0.8, color='red')
    
    ax2.set_xlabel('Configuration')
    ax2.set_ylabel('Latency (ms)')
    ax2.set_title('Latency Comparison')
    ax2.set_xticks(x)
    ax2.set_xticklabels(all_scenarios, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    throughput_plot_path = output_path / "performance_comparison.png"
    plt.savefig(throughput_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"性能对比图表已保存: {throughput_plot_path}")

def main():
    parser = argparse.ArgumentParser(description="Compare homogeneous vs heterogeneous configurations")
    parser.add_argument("--homogeneous-results", "-homo", required=True,
                       help="Path to homogeneous benchmark results directory")
    parser.add_argument("--heterogeneous-results", "-hetero", required=True,
                       help="Path to heterogeneous benchmark results directory")
    parser.add_argument("--output", "-o", default="./comparison_analysis",
                       help="Output directory for analysis results")
    
    args = parser.parse_args()
    
    try:
        # 加载结果
        print("加载同构配置结果...")
        homo_results = load_benchmark_results(args.homogeneous_results)
        
        print("加载异构配置结果...")
        hetero_results = load_benchmark_results(args.heterogeneous_results)
        
        # 执行对比分析
        compare_homogeneous_vs_heterogeneous(homo_results, hetero_results)
        
        # 生成可视化图表
        plot_performance_comparison(homo_results, hetero_results, args.output)
        
        print(f"\n分析完成！详细结果保存在: {args.output}")
        
    except Exception as e:
        print(f"错误: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())