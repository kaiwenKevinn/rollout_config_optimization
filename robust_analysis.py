#!/usr/bin/env python3
"""
SCOOT 实验结果稳健分析器
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def load_analysis_results(filename='scoot_analysis_results.json'):
    """加载分析结果"""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"错误: 找不到文件 {filename}")
        return []

def safe_mean(values):
    """安全计算平均值"""
    if not values:
        return 0
    return np.mean(values)

def safe_max(values):
    """安全计算最大值"""
    if not values:
        return 0
    return max(values)

def safe_min(values):
    """安全计算最小值"""
    if not values:
        return float('inf')
    return min(values)

def create_analysis_report():
    """创建分析报告"""
    results = load_analysis_results()
    
    if not results:
        print("没有找到分析结果数据")
        return
    
    # 分离成功和失败的实验
    successful = [r for r in results if not r.get('server_failed', True)]
    failed = [r for r in results if r.get('server_failed', False)]
    
    print("=" * 60)
    print("SCOOT 实验分析报告")
    print("=" * 60)
    
    # 1. 整体统计
    print(f"\n1. 实验概况:")
    print(f"   总实验数: {len(results)}")
    print(f"   成功实验: {len(successful)} ({len(successful)/len(results)*100:.1f}%)")
    print(f"   失败实验: {len(failed)} ({len(failed)/len(results)*100:.1f}%)")
    
    # 2. 按TP分组分析
    tp_groups = defaultdict(list)
    for run in successful:
        tp_groups[run['tp']].append(run)
    
    print(f"\n2. 按Tensor Parallel大小分析:")
    print("   TP\t实验数\t成功率\t平均吞吐量\t平均延迟")
    print("   " + "-" * 50)
    
    for tp in sorted(tp_groups.keys()):
        group_runs = tp_groups[tp]
        total_tp_experiments = len([r for r in results if r['tp'] == tp])
        success_rate = len(group_runs) / total_tp_experiments * 100
        
        # 安全提取性能指标
        throughputs = [r['request_throughput'] for r in group_runs if 'request_throughput' in r]
        latencies = [r['p95_latency_ms'] for r in group_runs if 'p95_latency_ms' in r]
        
        avg_throughput = safe_mean(throughputs)
        avg_latency = safe_mean(latencies)
        
        print(f"   {tp}\t{total_tp_experiments}\t{success_rate:.1f}%\t{avg_throughput:.2f} req/s\t{avg_latency:.2f} ms")
    
    # 3. 性能最佳配置
    if successful:
        print(f"\n3. 性能最佳配置:")
        
        # 过滤有完整数据的实验
        valid_runs = [r for r in successful 
                     if 'request_throughput' in r and 'p95_latency_ms' in r]
        
        if valid_runs:
            best_throughput_run = max(valid_runs, key=lambda x: x['request_throughput'])
            best_latency_run = min(valid_runs, key=lambda x: x['p95_latency_ms'])
            
            print(f"   最高吞吐量: {best_throughput_run['request_throughput']:.2f} req/s")
            print(f"     配置: TP={best_throughput_run['tp']}, "
                  f"max_num_seqs={best_throughput_run['max_num_seqs']}")
            
            print(f"   最低延迟: {best_latency_run['p95_latency_ms']:.2f} ms")
            print(f"     配置: TP={best_latency_run['tp']}, "
                  f"max_num_seqs={best_latency_run['max_num_seqs']}")
    
    # 4. 失败分析
    if failed:
        print(f"\n4. 失败实验分析:")
        tp_failures = defaultdict(int)
        for run in failed:
            tp_failures[run['tp']] += 1
        
        for tp in sorted(tp_failures.keys()):
            total_tp_runs = len([r for r in results if r['tp'] == tp])
            print(f"   TP={tp}: {tp_failures[tp]}/{total_tp_runs} 次实验失败")

def create_simple_visualization():
    """创建简单的可视化图表"""
    results = load_analysis_results()
    successful = [r for r in results if not r.get('server_failed', True)]
    
    if not successful:
        print("没有成功的实验数据用于可视化")
        return
    
    # 准备数据
    tp_throughput = defaultdict(list)
    tp_latency = defaultdict(list)
    
    for run in successful:
        if 'request_throughput' in run and 'p95_latency_ms' in run:
            tp_throughput[run['tp']].append(run['request_throughput'])
            tp_latency[run['tp']].append(run['p95_latency_ms'])
    
    if not tp_throughput:
        print("没有足够的有效数据进行可视化")
        return
    
    # 创建图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 吞吐量柱状图
    tps = sorted(tp_throughput.keys())
    avg_throughputs = [safe_mean(tp_throughput[tp]) for tp in tps]
    
    bars = ax1.bar(tps, avg_throughputs, alpha=0.7, color=['red', 'blue', 'green', 'orange'])
    ax1.set_xlabel('Tensor Parallel 大小')
    ax1.set_ylabel('平均吞吐量 (req/s)')
    ax1.set_title('不同TP配置的吞吐量对比')
    ax1.grid(True, alpha=0.3)
    
    # 标注数值
    for bar, value in zip(bars, avg_throughputs):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{value:.1f}', ha='center', va='bottom')
    
    # 延迟箱线图
    latency_data = [tp_latency[tp] for tp in tps if tp_latency[tp]]
    if latency_data:
        ax2.boxplot(latency_data, labels=[f'TP={tp}' for tp in tps if tp_latency[tp]])
        ax2.set_ylabel('P95 延迟 (ms)')
        ax2.set_title('不同TP配置的延迟分布')
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('scoot_simple_analysis.png', dpi=300, bbox_inches='tight')
    print("简单分析图表已保存为: scoot_simple_analysis.png")
    plt.show()

def generate_recommendations():
    """生成优化建议"""
    results = load_analysis_results()
    successful = [r for r in results if not r.get('server_failed', True)]
    
    print(f"\n{'='*60}")
    print("优化建议")
    print("=" * 60)
    
    if not successful:
        print("没有足够的成功实验数据提供建议")
        return
    
    # 统计成功率
    tp_success = defaultdict(int)
    tp_total = defaultdict(int)
    
    for run in results:
        tp_total[run['tp']] += 1
        if not run.get('server_failed', True):
            tp_success[run['tp']] += 1
    
    print("1. TP配置建议:")
    for tp in sorted(tp_total.keys()):
        success_rate = tp_success[tp] / tp_total[tp] * 100
        print(f"   TP={tp}: 成功率 {success_rate:.1f}% - ", end="")
        if success_rate >= 50:
            print("✅ 推荐使用")
        elif success_rate >= 30:
            print("⚠️  需谨慎使用")
        else:
            print("❌ 不推荐使用")
    
    # 性能建议
    valid_runs = [r for r in successful 
                 if 'request_throughput' in r and 'p95_latency_ms' in r]
    
    if valid_runs:
        print("\n2. 性能优化建议:")
        best_throughput = max(valid_runs, key=lambda x: x['request_throughput'])
        best_latency = min(valid_runs, key=lambda x: x['p95_latency_ms'])
        
        print(f"   追求高吞吐量: 使用 TP={best_throughput['tp']}, "
              f"max_num_seqs={best_throughput['max_num_seqs']}")
        print(f"   追求低延迟: 使用 TP={best_latency['tp']}, "
              f"max_num_seqs={best_latency['max_num_seqs']}")

def main():
    print("开始分析SCOOT实验结果...\n")
    
    # 生成分析报告
    create_analysis_report()
    
    # 生成可视化
    create_simple_visualization()
    
    # 生成建议
    generate_recommendations()
    
    print(f"\n分析完成！")

if __name__ == "__main__":
    main()