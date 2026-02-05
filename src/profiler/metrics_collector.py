"""
Metrics Collector Module

Collects and manages performance metrics during benchmark execution.
Supports GPU monitoring, latency tracking, and throughput calculation.
"""

import os
import json
import time
import logging
import threading
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from contextlib import contextmanager

import numpy as np

try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class RequestMetrics:
    """Metrics for a single inference request."""
    request_id: str
    question_id: str
    instance_id: str
    tp_degree: int
    input_tokens: int
    output_tokens: int
    ttft: float  # Time to first token (ms)
    tpot: float  # Time per output token (ms)
    total_latency: float  # Total request latency (ms)
    start_time: float
    first_token_time: Optional[float] = None
    end_time: Optional[float] = None
    gpu_memory_used_gb: float = 0.0
    gpu_utilization: float = 0.0
    success: bool = True
    error_message: str = ""
    sequence_category: str = ""
    queue_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @property
    def throughput(self) -> float:
        if self.total_latency > 0:
            return (self.output_tokens / self.total_latency) * 1000
        return 0.0


@dataclass
class GPUSnapshot:
    """Snapshot of GPU state."""
    timestamp: float
    gpu_id: int
    memory_used_gb: float
    memory_total_gb: float
    memory_utilization: float
    gpu_utilization: float
    temperature: float
    power_usage_w: float


@dataclass
class BenchmarkMetrics:
    """Aggregated metrics for a benchmark run."""
    scenario_name: str
    run_id: int
    start_time: str
    end_time: str
    duration_seconds: float
    total_requests: int
    successful_requests: int
    failed_requests: int
    total_input_tokens: int
    total_output_tokens: int
    throughput_tokens_per_second: float
    requests_per_second: float
    ttft_mean: float
    ttft_std: float
    ttft_p50: float
    ttft_p90: float
    ttft_p99: float
    tpot_mean: float
    tpot_std: float
    tpot_p50: float
    tpot_p90: float
    tpot_p99: float
    latency_mean: float
    latency_std: float
    latency_p50: float
    latency_p90: float
    latency_p99: float
    gpu_memory_peak_gb: float
    gpu_utilization_mean: float
    per_instance_metrics: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    scheduling_success_rate: float = 1.0
    avg_queue_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class GPUMonitor:
    """Monitors GPU metrics in background."""
    
    def __init__(self, gpu_ids: List[int], interval: float = 0.5):
        self.gpu_ids = gpu_ids
        self.interval = interval
        self._snapshots: List[GPUSnapshot] = []
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._nvml_initialized = False
        
        if PYNVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self._nvml_initialized = True
            except Exception as e:
                logger.warning(f"Failed to initialize NVML: {e}")
    
    def start(self) -> None:
        if not self._nvml_initialized:
            return
        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
    
    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
    
    def _monitor_loop(self) -> None:
        while self._running:
            try:
                for gpu_id in self.gpu_ids:
                    snapshot = self._collect_snapshot(gpu_id)
                    if snapshot:
                        with self._lock:
                            self._snapshots.append(snapshot)
            except Exception as e:
                logger.error(f"GPU monitoring error: {e}")
            time.sleep(self.interval)
    
    def _collect_snapshot(self, gpu_id: int) -> Optional[GPUSnapshot]:
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            try:
                power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000
            except:
                power = 0.0
            
            return GPUSnapshot(
                timestamp=time.time(),
                gpu_id=gpu_id,
                memory_used_gb=mem_info.used / (1024 ** 3),
                memory_total_gb=mem_info.total / (1024 ** 3),
                memory_utilization=mem_info.used / mem_info.total * 100,
                gpu_utilization=util.gpu,
                temperature=temp,
                power_usage_w=power
            )
        except Exception as e:
            return None
    
    def get_current_metrics(self, gpu_id: int) -> Dict[str, float]:
        if not self._nvml_initialized:
            return {'memory_used_gb': 0.0, 'gpu_utilization': 0.0}
        snapshot = self._collect_snapshot(gpu_id)
        if snapshot:
            return {
                'memory_used_gb': snapshot.memory_used_gb,
                'gpu_utilization': snapshot.gpu_utilization
            }
        return {'memory_used_gb': 0.0, 'gpu_utilization': 0.0}
    
    def get_snapshots(self) -> List[GPUSnapshot]:
        with self._lock:
            return list(self._snapshots)
    
    def get_peak_memory(self) -> Dict[int, float]:
        peak: Dict[int, float] = defaultdict(float)
        with self._lock:
            for s in self._snapshots:
                if s.memory_used_gb > peak[s.gpu_id]:
                    peak[s.gpu_id] = s.memory_used_gb
        return dict(peak)
    
    def get_average_utilization(self) -> Dict[int, float]:
        util_sum: Dict[int, float] = defaultdict(float)
        util_count: Dict[int, int] = defaultdict(int)
        with self._lock:
            for s in self._snapshots:
                util_sum[s.gpu_id] += s.gpu_utilization
                util_count[s.gpu_id] += 1
        return {gid: util_sum[gid] / util_count[gid] for gid in util_sum if util_count[gid] > 0}
    
    def clear(self) -> None:
        with self._lock:
            self._snapshots.clear()


class MetricsCollector:
    """Collects and manages metrics during benchmark execution."""
    
    def __init__(
        self,
        gpu_ids: Optional[List[int]] = None,
        gpu_monitor_interval: float = 0.5,
        save_intermediate: bool = True,
        output_dir: Optional[str] = None
    ):
        self.gpu_ids = gpu_ids or []
        self.save_intermediate = save_intermediate
        self.output_dir = Path(output_dir) if output_dir else None
        self.gpu_monitor = GPUMonitor(self.gpu_ids, gpu_monitor_interval)
        self._request_metrics: List[RequestMetrics] = []
        self._benchmark_start_time: Optional[float] = None
        self._benchmark_end_time: Optional[float] = None
        self._current_scenario: str = ""
        self._current_run_id: int = 0
        self._lock = threading.Lock()
    
    def start_benchmark(self, scenario_name: str, run_id: int) -> None:
        self._current_scenario = scenario_name
        self._current_run_id = run_id
        self._benchmark_start_time = time.time()
        self._request_metrics.clear()
        self.gpu_monitor.clear()
        self.gpu_monitor.start()
    
    def end_benchmark(self) -> BenchmarkMetrics:
        self._benchmark_end_time = time.time()
        self.gpu_monitor.stop()
        metrics = self._calculate_benchmark_metrics()
        if self.save_intermediate and self.output_dir:
            self._save_intermediate_results(metrics)
        return metrics
    
    def record_request(self, metrics: RequestMetrics) -> None:
        with self._lock:
            self._request_metrics.append(metrics)
    
    @contextmanager
    def track_request(self, request_id: str, question_id: str, instance_id: str,
                     tp_degree: int, input_tokens: int, sequence_category: str = ""):
        tracker = RequestTracker(
            request_id, question_id, instance_id, tp_degree,
            input_tokens, sequence_category, self.gpu_monitor, self.gpu_ids
        )
        try:
            yield tracker
        except Exception as e:
            tracker.set_error(str(e))
        finally:
            self.record_request(tracker.finalize())
    
    def _calculate_benchmark_metrics(self) -> BenchmarkMetrics:
        if not self._request_metrics:
            raise ValueError("No request metrics collected")
        
        successful = [m for m in self._request_metrics if m.success]
        failed = [m for m in self._request_metrics if not m.success]
        
        ttft_values = [m.ttft for m in successful if m.ttft > 0]
        tpot_values = [m.tpot for m in successful if m.tpot > 0]
        latency_values = [m.total_latency for m in successful]
        
        total_output_tokens = sum(m.output_tokens for m in successful)
        total_input_tokens = sum(m.input_tokens for m in successful)
        duration = self._benchmark_end_time - self._benchmark_start_time
        
        peak_memory = self.gpu_monitor.get_peak_memory()
        avg_utilization = self.gpu_monitor.get_average_utilization()
        per_instance = self._calculate_per_instance_metrics(successful)
        queue_times = [m.queue_time for m in successful if m.queue_time > 0]
        
        return BenchmarkMetrics(
            scenario_name=self._current_scenario,
            run_id=self._current_run_id,
            start_time=datetime.fromtimestamp(self._benchmark_start_time).isoformat(),
            end_time=datetime.fromtimestamp(self._benchmark_end_time).isoformat(),
            duration_seconds=duration,
            total_requests=len(self._request_metrics),
            successful_requests=len(successful),
            failed_requests=len(failed),
            total_input_tokens=total_input_tokens,
            total_output_tokens=total_output_tokens,
            throughput_tokens_per_second=total_output_tokens / duration if duration > 0 else 0,
            requests_per_second=len(successful) / duration if duration > 0 else 0,
            ttft_mean=np.mean(ttft_values) if ttft_values else 0,
            ttft_std=np.std(ttft_values) if ttft_values else 0,
            ttft_p50=np.percentile(ttft_values, 50) if ttft_values else 0,
            ttft_p90=np.percentile(ttft_values, 90) if ttft_values else 0,
            ttft_p99=np.percentile(ttft_values, 99) if ttft_values else 0,
            tpot_mean=np.mean(tpot_values) if tpot_values else 0,
            tpot_std=np.std(tpot_values) if tpot_values else 0,
            tpot_p50=np.percentile(tpot_values, 50) if tpot_values else 0,
            tpot_p90=np.percentile(tpot_values, 90) if tpot_values else 0,
            tpot_p99=np.percentile(tpot_values, 99) if tpot_values else 0,
            latency_mean=np.mean(latency_values) if latency_values else 0,
            latency_std=np.std(latency_values) if latency_values else 0,
            latency_p50=np.percentile(latency_values, 50) if latency_values else 0,
            latency_p90=np.percentile(latency_values, 90) if latency_values else 0,
            latency_p99=np.percentile(latency_values, 99) if latency_values else 0,
            gpu_memory_peak_gb=max(peak_memory.values()) if peak_memory else 0,
            gpu_utilization_mean=np.mean(list(avg_utilization.values())) if avg_utilization else 0,
            per_instance_metrics=per_instance,
            scheduling_success_rate=len(successful) / len(self._request_metrics) if self._request_metrics else 1.0,
            avg_queue_time=np.mean(queue_times) if queue_times else 0
        )
    
    def _calculate_per_instance_metrics(self, metrics: List[RequestMetrics]) -> Dict[str, Dict[str, Any]]:
        by_instance: Dict[str, List[RequestMetrics]] = defaultdict(list)
        for m in metrics:
            by_instance[m.instance_id].append(m)
        
        result = {}
        for instance_id, instance_metrics in by_instance.items():
            latencies = [m.total_latency for m in instance_metrics]
            throughputs = [m.throughput for m in instance_metrics]
            result[instance_id] = {
                'request_count': len(instance_metrics),
                'tp_degree': instance_metrics[0].tp_degree if instance_metrics else 0,
                'total_tokens': sum(m.output_tokens for m in instance_metrics),
                'latency_mean': np.mean(latencies),
                'latency_p50': np.percentile(latencies, 50),
                'latency_p99': np.percentile(latencies, 99),
                'throughput_mean': np.mean(throughputs)
            }
        return result
    
    def _save_intermediate_results(self, metrics: BenchmarkMetrics) -> None:
        output_dir = self.output_dir / self._current_scenario
        output_dir.mkdir(parents=True, exist_ok=True)
        
        metrics_file = output_dir / f"run_{self._current_run_id}_metrics.json"
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(metrics.to_dict(), f, indent=2, ensure_ascii=False)
        
        requests_file = output_dir / f"run_{self._current_run_id}_requests.json"
        with open(requests_file, 'w', encoding='utf-8') as f:
            json.dump([m.to_dict() for m in self._request_metrics], f, indent=2, ensure_ascii=False)
    
    def get_request_metrics(self) -> List[RequestMetrics]:
        return list(self._request_metrics)


class RequestTracker:
    """Helper class for tracking individual request metrics."""
    
    def __init__(self, request_id: str, question_id: str, instance_id: str,
                 tp_degree: int, input_tokens: int, sequence_category: str,
                 gpu_monitor: GPUMonitor, gpu_ids: List[int]):
        self.request_id = request_id
        self.question_id = question_id
        self.instance_id = instance_id
        self.tp_degree = tp_degree
        self.input_tokens = input_tokens
        self.sequence_category = sequence_category
        self.gpu_monitor = gpu_monitor
        self.gpu_ids = gpu_ids
        self.start_time = time.time()
        self.first_token_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.output_tokens = 0
        self.success = True
        self.error_message = ""
        self.queue_time = 0.0
    
    def record_first_token(self) -> None:
        if self.first_token_time is None:
            self.first_token_time = time.time()
    
    def set_output_tokens(self, count: int) -> None:
        self.output_tokens = count
    
    def set_queue_time(self, queue_time: float) -> None:
        self.queue_time = queue_time
    
    def set_error(self, message: str) -> None:
        self.success = False
        self.error_message = message
    
    def finalize(self) -> RequestMetrics:
        self.end_time = time.time()
        total_latency = (self.end_time - self.start_time) * 1000
        
        if self.first_token_time:
            ttft = (self.first_token_time - self.start_time) * 1000
            tpot = (self.end_time - self.first_token_time) * 1000 / (self.output_tokens - 1) if self.output_tokens > 1 else 0.0
        else:
            ttft = total_latency
            tpot = total_latency / self.output_tokens if self.output_tokens > 0 else 0.0
        
        gpu_metrics = self.gpu_monitor.get_current_metrics(self.gpu_ids[0]) if self.gpu_ids else {}
        
        return RequestMetrics(
            request_id=self.request_id,
            question_id=self.question_id,
            instance_id=self.instance_id,
            tp_degree=self.tp_degree,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            ttft=ttft,
            tpot=tpot,
            total_latency=total_latency,
            start_time=self.start_time,
            first_token_time=self.first_token_time,
            end_time=self.end_time,
            gpu_memory_used_gb=gpu_metrics.get('memory_used_gb', 0.0),
            gpu_utilization=gpu_metrics.get('gpu_utilization', 0.0),
            success=self.success,
            error_message=self.error_message,
            sequence_category=self.sequence_category,
            queue_time=self.queue_time
        )
