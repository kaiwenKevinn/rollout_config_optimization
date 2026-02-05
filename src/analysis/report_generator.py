"""
Report Generator Module

Generates comprehensive benchmark reports including plots and markdown analysis.
"""

import json
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False

from .aggregator import MetricsAggregator, ScenarioAggregatedMetrics, ComparisonResult

logger = logging.getLogger(__name__)


class ReportGenerator:
    """
    Generates benchmark reports with visualizations and analysis.
    
    Features:
    - Comparison charts (throughput, latency)
    - Distribution plots
    - Sequence length analysis
    - Markdown report generation
    """
    
    def __init__(
        self,
        aggregator: MetricsAggregator,
        output_dir: str = "./results/analysis"
    ):
        """
        Initialize the report generator.
        
        Args:
            aggregator: MetricsAggregator with loaded results
            output_dir: Directory for output files
        """
        self.aggregator = aggregator
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set plot style
        if SEABORN_AVAILABLE:
            sns.set_style("whitegrid")
            sns.set_palette("husl")
    
    def generate_full_report(self) -> str:
        """
        Generate a full benchmark report.
        
        Returns:
            Path to the generated report
        """
        logger.info("Generating full benchmark report...")
        
        # Generate plots
        if MATPLOTLIB_AVAILABLE:
            self.plot_throughput_comparison()
            self.plot_latency_comparison()
            self.plot_latency_distribution()
            self.plot_ttft_tpot_comparison()
            self.plot_sequence_length_analysis()
            self.plot_tp_performance()
        
        # Generate markdown report
        report_path = self.generate_markdown_report()
        
        # Export data
        self.aggregator.export_to_csv(str(self.output_dir / "metrics_summary.csv"))
        self.aggregator.export_to_json(str(self.output_dir / "metrics_full.json"))
        
        logger.info(f"Report generated: {report_path}")
        return report_path
    
    def plot_throughput_comparison(self) -> Optional[str]:
        """Plot throughput comparison across scenarios."""
        if not MATPLOTLIB_AVAILABLE:
            return None
        
        df = self.aggregator.to_dataframe()
        if df.empty:
            return None
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Sort by throughput
        df_sorted = df.sort_values("throughput_mean", ascending=True)
        
        # Create bar plot
        colors = ['#2ecc71' if 'hetero' in s else '#3498db' for s in df_sorted['scenario']]
        bars = ax.barh(df_sorted['scenario'], df_sorted['throughput_mean'], 
                       xerr=df_sorted['throughput_std'], color=colors, capsize=3)
        
        ax.set_xlabel('Throughput (tokens/s)')
        ax.set_title('Throughput Comparison Across Scenarios')
        
        # Add legend
        homo_patch = mpatches.Patch(color='#3498db', label='Homogeneous')
        hetero_patch = mpatches.Patch(color='#2ecc71', label='Heterogeneous')
        ax.legend(handles=[homo_patch, hetero_patch], loc='lower right')
        
        # Add value labels
        for bar, val in zip(bars, df_sorted['throughput_mean']):
            ax.text(val + 1, bar.get_y() + bar.get_height()/2, 
                   f'{val:.1f}', va='center', fontsize=9)
        
        plt.tight_layout()
        output_path = str(self.output_dir / "throughput_comparison.png")
        plt.savefig(output_path, dpi=150)
        plt.close()
        
        return output_path
    
    def plot_latency_comparison(self) -> Optional[str]:
        """Plot latency comparison with percentiles."""
        if not MATPLOTLIB_AVAILABLE:
            return None
        
        df = self.aggregator.to_dataframe()
        if df.empty:
            return None
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        scenarios = df['scenario'].tolist()
        x = np.arange(len(scenarios))
        width = 0.25
        
        # Plot mean, p90, p99
        bars1 = ax.bar(x - width, df['latency_mean'], width, label='Mean', color='#3498db')
        bars2 = ax.bar(x, df['latency_p90'], width, label='P90', color='#e74c3c')
        bars3 = ax.bar(x + width, df['latency_p99'], width, label='P99', color='#9b59b6')
        
        ax.set_ylabel('Latency (ms)')
        ax.set_title('Latency Comparison (Mean, P90, P99)')
        ax.set_xticks(x)
        ax.set_xticklabels(scenarios, rotation=45, ha='right')
        ax.legend()
        
        plt.tight_layout()
        output_path = str(self.output_dir / "latency_comparison.png")
        plt.savefig(output_path, dpi=150)
        plt.close()
        
        return output_path
    
    def plot_latency_distribution(self) -> Optional[str]:
        """Plot latency distribution box plots."""
        if not MATPLOTLIB_AVAILABLE or not SEABORN_AVAILABLE:
            return None
        
        # Get request-level data
        request_df = self.aggregator.get_request_dataframe()
        if request_df.empty or 'total_latency' not in request_df.columns:
            return None
        
        # Add scenario info (from instance_id pattern)
        if 'instance_id' in request_df.columns:
            request_df['scenario_type'] = request_df['instance_id'].apply(
                lambda x: 'heterogeneous' if 'hetero' in str(x) else 'homogeneous'
            )
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Filter successful requests
        success_df = request_df[request_df.get('success', True) == True]
        
        if 'instance_id' in success_df.columns:
            sns.boxplot(data=success_df, x='instance_id', y='total_latency', ax=ax)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        
        ax.set_ylabel('Total Latency (ms)')
        ax.set_title('Latency Distribution by Instance')
        
        plt.tight_layout()
        output_path = str(self.output_dir / "latency_distribution.png")
        plt.savefig(output_path, dpi=150)
        plt.close()
        
        return output_path
    
    def plot_ttft_tpot_comparison(self) -> Optional[str]:
        """Plot TTFT and TPOT comparison."""
        if not MATPLOTLIB_AVAILABLE:
            return None
        
        df = self.aggregator.to_dataframe()
        if df.empty:
            return None
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        scenarios = df['scenario'].tolist()
        x = np.arange(len(scenarios))
        
        # TTFT
        colors = ['#2ecc71' if 'hetero' in s else '#3498db' for s in scenarios]
        ax1.bar(x, df['ttft_mean'], color=colors)
        ax1.errorbar(x, df['ttft_mean'], yerr=df['ttft_p99'] - df['ttft_mean'],
                    fmt='none', color='black', capsize=3)
        ax1.set_ylabel('TTFT (ms)')
        ax1.set_title('Time to First Token (Mean + P99)')
        ax1.set_xticks(x)
        ax1.set_xticklabels(scenarios, rotation=45, ha='right')
        
        # TPOT
        ax2.bar(x, df['tpot_mean'], color=colors)
        ax2.errorbar(x, df['tpot_mean'], yerr=df['tpot_p99'] - df['tpot_mean'],
                    fmt='none', color='black', capsize=3)
        ax2.set_ylabel('TPOT (ms)')
        ax2.set_title('Time per Output Token (Mean + P99)')
        ax2.set_xticks(x)
        ax2.set_xticklabels(scenarios, rotation=45, ha='right')
        
        plt.tight_layout()
        output_path = str(self.output_dir / "ttft_tpot_comparison.png")
        plt.savefig(output_path, dpi=150)
        plt.close()
        
        return output_path
    
    def plot_sequence_length_analysis(self) -> Optional[str]:
        """Plot latency vs sequence length."""
        if not MATPLOTLIB_AVAILABLE:
            return None
        
        request_df = self.aggregator.get_request_dataframe()
        if request_df.empty:
            return None
        
        if 'input_tokens' not in request_df.columns or 'total_latency' not in request_df.columns:
            return None
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Filter successful requests
        success_df = request_df[request_df.get('success', True) == True]
        
        if 'tp_degree' in success_df.columns:
            for tp in sorted(success_df['tp_degree'].unique()):
                tp_data = success_df[success_df['tp_degree'] == tp]
                ax.scatter(tp_data['input_tokens'], tp_data['total_latency'],
                          label=f'TP={tp}', alpha=0.5, s=20)
        else:
            ax.scatter(success_df['input_tokens'], success_df['total_latency'],
                      alpha=0.5, s=20)
        
        ax.set_xlabel('Input Tokens')
        ax.set_ylabel('Total Latency (ms)')
        ax.set_title('Latency vs Sequence Length')
        ax.legend()
        
        plt.tight_layout()
        output_path = str(self.output_dir / "sequence_length_analysis.png")
        plt.savefig(output_path, dpi=150)
        plt.close()
        
        return output_path
    
    def plot_tp_performance(self) -> Optional[str]:
        """Plot performance by TP degree."""
        if not MATPLOTLIB_AVAILABLE:
            return None
        
        # Collect per-TP metrics
        tp_data = []
        for name, metrics in self.aggregator._scenario_metrics.items():
            for tp, tp_metrics in metrics.per_tp_metrics.items():
                tp_data.append({
                    'scenario': name,
                    'tp_degree': tp,
                    'request_count': tp_metrics.get('request_count', 0),
                    'latency_mean': tp_metrics.get('latency_mean', 0),
                    'throughput_mean': tp_metrics.get('throughput_mean', 0)
                })
        
        if not tp_data:
            return None
        
        tp_df = pd.DataFrame(tp_data)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Group by TP degree
        tp_grouped = tp_df.groupby('tp_degree').agg({
            'latency_mean': 'mean',
            'throughput_mean': 'mean',
            'request_count': 'sum'
        }).reset_index()
        
        # Latency by TP
        ax1.bar(tp_grouped['tp_degree'].astype(str), tp_grouped['latency_mean'],
               color='#e74c3c')
        ax1.set_xlabel('TP Degree')
        ax1.set_ylabel('Mean Latency (ms)')
        ax1.set_title('Mean Latency by TP Degree')
        
        # Throughput by TP
        ax2.bar(tp_grouped['tp_degree'].astype(str), tp_grouped['throughput_mean'],
               color='#3498db')
        ax2.set_xlabel('TP Degree')
        ax2.set_ylabel('Mean Throughput (tokens/s)')
        ax2.set_title('Mean Throughput by TP Degree')
        
        plt.tight_layout()
        output_path = str(self.output_dir / "tp_performance.png")
        plt.savefig(output_path, dpi=150)
        plt.close()
        
        return output_path
    
    def generate_markdown_report(self) -> str:
        """Generate a comprehensive markdown report."""
        report_lines = [
            "# Heterogeneous TP Configuration Benchmark Report",
            "",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Executive Summary",
            "",
        ]
        
        # Summary statistics
        df = self.aggregator.to_dataframe()
        if not df.empty:
            best_throughput = df.loc[df['throughput_mean'].idxmax()]
            best_latency = df.loc[df['latency_mean'].idxmin()]
            
            report_lines.extend([
                f"- **Best Throughput**: {best_throughput['scenario']} "
                f"({best_throughput['throughput_mean']:.2f} tokens/s)",
                f"- **Best Latency**: {best_latency['scenario']} "
                f"({best_latency['latency_mean']:.2f} ms mean)",
                f"- **Total Scenarios Tested**: {len(df)}",
                "",
            ])
        
        # Detailed results table
        report_lines.extend([
            "## Detailed Results",
            "",
            "### Performance Metrics by Scenario",
            "",
            "| Scenario | Type | Throughput (tokens/s) | Latency Mean (ms) | P99 (ms) | TTFT (ms) | TPOT (ms) |",
            "|----------|------|----------------------|-------------------|----------|-----------|-----------|",
        ])
        
        for _, row in df.iterrows():
            report_lines.append(
                f"| {row['scenario']} | {row['type']} | "
                f"{row['throughput_mean']:.2f} | {row['latency_mean']:.2f} | "
                f"{row['latency_p99']:.2f} | {row['ttft_mean']:.2f} | "
                f"{row['tpot_mean']:.2f} |"
            )
        
        report_lines.append("")
        
        # Comparisons
        report_lines.extend([
            "## Scenario Comparisons",
            "",
        ])
        
        # Compare heterogeneous vs homogeneous
        homo_scenarios = df[df['type'] == 'homogeneous']['scenario'].tolist()
        hetero_scenarios = df[df['type'] == 'heterogeneous']['scenario'].tolist()
        
        if homo_scenarios and hetero_scenarios:
            report_lines.append("### Homogeneous vs Heterogeneous")
            report_lines.append("")
            
            for hetero in hetero_scenarios:
                for homo in homo_scenarios:
                    comparison = self.aggregator.compare_scenarios(homo, hetero)
                    if comparison:
                        report_lines.extend([
                            f"**{hetero} vs {homo}**:",
                            f"- Throughput: {comparison.throughput_improvement:+.1f}%",
                            f"- Latency: {comparison.latency_improvement:+.1f}%",
                            f"- TTFT: {comparison.ttft_improvement:+.1f}%",
                            f"- TPOT: {comparison.tpot_improvement:+.1f}%",
                            ""
                        ])
        
        # Per-category analysis
        report_lines.extend([
            "## Sequence Category Analysis",
            "",
        ])
        
        for name, metrics in self.aggregator._scenario_metrics.items():
            if metrics.per_category_metrics:
                report_lines.append(f"### {name}")
                report_lines.append("")
                report_lines.append("| Category | Count | Avg Input Tokens | Latency Mean (ms) | P99 (ms) |")
                report_lines.append("|----------|-------|------------------|-------------------|----------|")
                
                for cat, cat_metrics in metrics.per_category_metrics.items():
                    report_lines.append(
                        f"| {cat} | {cat_metrics['request_count']} | "
                        f"{cat_metrics['avg_input_tokens']:.0f} | "
                        f"{cat_metrics['latency_mean']:.2f} | "
                        f"{cat_metrics['latency_p99']:.2f} |"
                    )
                report_lines.append("")
        
        # Plots section
        report_lines.extend([
            "## Visualizations",
            "",
            "### Throughput Comparison",
            "![Throughput Comparison](throughput_comparison.png)",
            "",
            "### Latency Comparison",
            "![Latency Comparison](latency_comparison.png)",
            "",
            "### Latency Distribution",
            "![Latency Distribution](latency_distribution.png)",
            "",
            "### TTFT and TPOT",
            "![TTFT TPOT](ttft_tpot_comparison.png)",
            "",
            "### Sequence Length Analysis",
            "![Sequence Length](sequence_length_analysis.png)",
            "",
            "### TP Degree Performance",
            "![TP Performance](tp_performance.png)",
            "",
        ])
        
        # Conclusions
        report_lines.extend([
            "## Conclusions",
            "",
            "Based on the benchmark results:",
            "",
        ])
        
        if not df.empty:
            # Find best configurations
            best_tp = df.loc[df['throughput_mean'].idxmax()]
            best_lat = df.loc[df['latency_mean'].idxmin()]
            
            report_lines.extend([
                f"1. **Best Throughput Configuration**: {best_tp['scenario']} "
                f"achieves {best_tp['throughput_mean']:.2f} tokens/s",
                "",
                f"2. **Best Latency Configuration**: {best_lat['scenario']} "
                f"achieves {best_lat['latency_mean']:.2f} ms mean latency",
                "",
            ])
            
            # Heterogeneous analysis
            hetero_df = df[df['type'] == 'heterogeneous']
            homo_df = df[df['type'] == 'homogeneous']
            
            if not hetero_df.empty and not homo_df.empty:
                hetero_throughput = hetero_df['throughput_mean'].mean()
                homo_throughput = homo_df['throughput_mean'].mean()
                improvement = (hetero_throughput - homo_throughput) / homo_throughput * 100
                
                report_lines.extend([
                    f"3. **Heterogeneous vs Homogeneous**: Heterogeneous configurations "
                    f"show {improvement:+.1f}% throughput difference on average",
                    ""
                ])
        
        # Write report
        report_content = "\n".join(report_lines)
        report_path = self.output_dir / "benchmark_report.md"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        return str(report_path)


def generate_report(results_dir: str = "./results", output_dir: str = None) -> str:
    """
    Convenience function to generate a benchmark report.
    
    Args:
        results_dir: Directory containing benchmark results
        output_dir: Output directory for report (default: results_dir/analysis)
        
    Returns:
        Path to the generated report
    """
    if output_dir is None:
        output_dir = str(Path(results_dir) / "analysis")
    
    # Load results
    aggregator = MetricsAggregator(results_dir)
    aggregator.load_all_scenarios()
    
    # Generate report
    generator = ReportGenerator(aggregator, output_dir)
    return generator.generate_full_report()
