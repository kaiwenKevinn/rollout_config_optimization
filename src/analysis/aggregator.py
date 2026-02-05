"""
Metrics Aggregator Module

Aggregates and analyzes benchmark results across multiple scenarios and runs.
"""

import json
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class PercentileStats:
    min: float
    max: float
    mean: float
    std: float
    p50: float
    p90: float
    p95: float
    p99: float
    
    @classmethod
    def from_values(cls, values: List[float]) -> 'PercentileStats':
        if not values:
            return cls(0, 0, 0, 0, 0, 0, 0, 0)
        arr = np.array(values)
        return cls(
            min=float(np.min(arr)), max=float(np.max(arr)),
            mean=float(np.mean(arr)), std=float(np.std(arr)),
            p50=float(np.percentile(arr, 50)), p90=float(np.percentile(arr, 90)),
            p95=float(np.percentile(arr, 95)), p99=float(np.percentile(arr, 99))
        )
    
    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class ScenarioAggregatedMetrics:
    scenario_name: str
    scenario_type: str
    num_runs: int
    total_requests: int
    successful_requests: int
    throughput: PercentileStats
    requests_per_second: PercentileStats
    latency: PercentileStats
    ttft: PercentileStats
    tpot: PercentileStats
    gpu_memory_peak_gb: float
    gpu_utilization_mean: float
    per_tp_metrics: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    per_category_metrics: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_name": self.scenario_name, "scenario_type": self.scenario_type,
            "num_runs": self.num_runs, "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "success_rate": self.successful_requests / self.total_requests if self.total_requests > 0 else 0,
            "throughput": self.throughput.to_dict(),
            "latency": self.latency.to_dict(), "ttft": self.ttft.to_dict(), "tpot": self.tpot.to_dict(),
            "gpu_memory_peak_gb": self.gpu_memory_peak_gb,
            "per_tp_metrics": self.per_tp_metrics, "per_category_metrics": self.per_category_metrics
        }


@dataclass
class ComparisonResult:
    baseline_scenario: str
    comparison_scenario: str
    throughput_improvement: float
    latency_improvement: float
    ttft_improvement: float
    tpot_improvement: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MetricsAggregator:
    """Aggregates benchmark metrics across scenarios and runs."""
    
    def __init__(self, results_dir: str = "./results"):
        self.results_dir = Path(results_dir)
        self._scenario_metrics: Dict[str, ScenarioAggregatedMetrics] = {}
        self._raw_data: Dict[str, List[Dict[str, Any]]] = {}
    
    def load_scenario_results(self, scenario_name: str) -> Optional[ScenarioAggregatedMetrics]:
        scenario_dir = self.results_dir / scenario_name
        if not scenario_dir.exists():
            return None
        
        run_data, request_data = [], []
        for run_file in sorted(scenario_dir.glob("run_*.json")):
            if "requests" in run_file.name:
                continue
            with open(run_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if data.get("success"):
                    run_data.append(data)
        
        for requests_file in sorted(scenario_dir.glob("run_*_requests.json")):
            with open(requests_file, 'r', encoding='utf-8') as f:
                request_data.extend(json.load(f))
        
        if not run_data:
            return None
        
        metrics = self._aggregate_runs(scenario_name, run_data, request_data)
        self._scenario_metrics[scenario_name] = metrics
        self._raw_data[scenario_name] = request_data
        return metrics
    
    def load_all_scenarios(self) -> Dict[str, ScenarioAggregatedMetrics]:
        for scenario_dir in self.results_dir.iterdir():
            if scenario_dir.is_dir() and not scenario_dir.name.startswith("."):
                self.load_scenario_results(scenario_dir.name)
        return self._scenario_metrics
    
    def _aggregate_runs(self, scenario_name: str, run_data: List[Dict], request_data: List[Dict]) -> ScenarioAggregatedMetrics:
        throughputs, latencies, ttfts, tpots = [], [], [], []
        gpu_memory_peaks, total_requests, successful_requests = [], 0, 0
        
        for run in run_data:
            metrics = run.get("metrics", {})
            if not metrics:
                continue
            throughputs.append(metrics.get("throughput_tokens_per_second", 0))
            latencies.append(metrics.get("latency_mean", 0))
            ttfts.append(metrics.get("ttft_mean", 0))
            tpots.append(metrics.get("tpot_mean", 0))
            gpu_memory_peaks.append(metrics.get("gpu_memory_peak_gb", 0))
            total_requests += metrics.get("total_requests", 0)
            successful_requests += metrics.get("successful_requests", 0)
        
        request_latencies = [r.get("total_latency", 0) for r in request_data if r.get("success")]
        request_ttfts = [r.get("ttft", 0) for r in request_data if r.get("success") and r.get("ttft", 0) > 0]
        request_tpots = [r.get("tpot", 0) for r in request_data if r.get("success") and r.get("tpot", 0) > 0]
        
        per_tp = self._calculate_per_tp_metrics(request_data)
        per_category = self._calculate_per_category_metrics(request_data)
        scenario_type = "heterogeneous" if "hetero" in scenario_name.lower() else "homogeneous"
        
        return ScenarioAggregatedMetrics(
            scenario_name=scenario_name, scenario_type=scenario_type,
            num_runs=len(run_data), total_requests=total_requests,
            successful_requests=successful_requests,
            throughput=PercentileStats.from_values(throughputs),
            requests_per_second=PercentileStats.from_values([]),
            latency=PercentileStats.from_values(request_latencies),
            ttft=PercentileStats.from_values(request_ttfts),
            tpot=PercentileStats.from_values(request_tpots),
            gpu_memory_peak_gb=max(gpu_memory_peaks) if gpu_memory_peaks else 0,
            gpu_utilization_mean=0, per_tp_metrics=per_tp, per_category_metrics=per_category
        )
    
    def _calculate_per_tp_metrics(self, request_data: List[Dict]) -> Dict[int, Dict[str, Any]]:
        by_tp: Dict[int, List[Dict]] = defaultdict(list)
        for req in request_data:
            tp = req.get("tp_degree", 0)
            if tp > 0 and req.get("success"):
                by_tp[tp].append(req)
        
        result = {}
        for tp, requests in by_tp.items():
            latencies = [r.get("total_latency", 0) for r in requests]
            result[tp] = {
                "request_count": len(requests),
                "latency_mean": np.mean(latencies) if latencies else 0,
                "latency_p99": np.percentile(latencies, 99) if latencies else 0
            }
        return result
    
    def _calculate_per_category_metrics(self, request_data: List[Dict]) -> Dict[str, Dict[str, Any]]:
        by_category: Dict[str, List[Dict]] = defaultdict(list)
        for req in request_data:
            category = req.get("sequence_category", "unknown")
            if req.get("success"):
                by_category[category].append(req)
        
        result = {}
        for category, requests in by_category.items():
            latencies = [r.get("total_latency", 0) for r in requests]
            input_tokens = [r.get("input_tokens", 0) for r in requests]
            result[category] = {
                "request_count": len(requests),
                "avg_input_tokens": np.mean(input_tokens) if input_tokens else 0,
                "latency_mean": np.mean(latencies) if latencies else 0,
                "latency_p99": np.percentile(latencies, 99) if latencies else 0
            }
        return result
    
    def compare_scenarios(self, baseline: str, comparison: str) -> Optional[ComparisonResult]:
        if baseline not in self._scenario_metrics:
            self.load_scenario_results(baseline)
        if comparison not in self._scenario_metrics:
            self.load_scenario_results(comparison)
        
        base_metrics = self._scenario_metrics.get(baseline)
        comp_metrics = self._scenario_metrics.get(comparison)
        if not base_metrics or not comp_metrics:
            return None
        
        throughput_imp = ((comp_metrics.throughput.mean - base_metrics.throughput.mean) /
                         base_metrics.throughput.mean * 100) if base_metrics.throughput.mean > 0 else 0
        latency_imp = ((base_metrics.latency.mean - comp_metrics.latency.mean) /
                      base_metrics.latency.mean * 100) if base_metrics.latency.mean > 0 else 0
        ttft_imp = ((base_metrics.ttft.mean - comp_metrics.ttft.mean) /
                   base_metrics.ttft.mean * 100) if base_metrics.ttft.mean > 0 else 0
        tpot_imp = ((base_metrics.tpot.mean - comp_metrics.tpot.mean) /
                   base_metrics.tpot.mean * 100) if base_metrics.tpot.mean > 0 else 0
        
        return ComparisonResult(baseline, comparison, throughput_imp, latency_imp, ttft_imp, tpot_imp)
    
    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for name, metrics in self._scenario_metrics.items():
            rows.append({
                "scenario": name, "type": metrics.scenario_type, "num_runs": metrics.num_runs,
                "total_requests": metrics.total_requests,
                "success_rate": metrics.successful_requests / metrics.total_requests if metrics.total_requests > 0 else 0,
                "throughput_mean": metrics.throughput.mean, "throughput_std": metrics.throughput.std,
                "latency_mean": metrics.latency.mean, "latency_p50": metrics.latency.p50,
                "latency_p90": metrics.latency.p90, "latency_p99": metrics.latency.p99,
                "ttft_mean": metrics.ttft.mean, "ttft_p99": metrics.ttft.p99,
                "tpot_mean": metrics.tpot.mean, "tpot_p99": metrics.tpot.p99,
                "gpu_memory_peak_gb": metrics.gpu_memory_peak_gb
            })
        return pd.DataFrame(rows)
    
    def get_request_dataframe(self, scenario_name: Optional[str] = None) -> pd.DataFrame:
        if scenario_name:
            data = self._raw_data.get(scenario_name, [])
        else:
            data = [r for requests in self._raw_data.values() for r in requests]
        return pd.DataFrame(data)
    
    def export_to_csv(self, output_path: str) -> None:
        self.to_dataframe().to_csv(output_path, index=False)
    
    def export_to_json(self, output_path: str) -> None:
        data = {name: metrics.to_dict() for name, metrics in self._scenario_metrics.items()}
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def print_summary(self) -> None:
        if not self._scenario_metrics:
            print("No scenarios loaded")
            return
        print("\n" + "=" * 80)
        print("Benchmark Results Summary")
        print("=" * 80)
        for name, metrics in sorted(self._scenario_metrics.items()):
            print(f"\n{name} ({metrics.scenario_type})")
            print(f"  Throughput: {metrics.throughput.mean:.2f} tokens/s")
            print(f"  Latency: {metrics.latency.mean:.2f} ms (p99: {metrics.latency.p99:.2f})")
            print(f"  TTFT: {metrics.ttft.mean:.2f} ms, TPOT: {metrics.tpot.mean:.2f} ms")
        print("=" * 80)
