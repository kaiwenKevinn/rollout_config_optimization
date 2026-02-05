"""
Benchmark Runner Module

Executes benchmark scenarios with concurrent request handling,
progress tracking, and result collection.
"""

import os
import sys
import json
import time
import random
import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

from ..profiler.dataset_loader import GPQADatasetLoader, GPQAQuestion
from ..profiler.sequence_profiler import SequenceProfiler, SequenceInfo
from ..profiler.metrics_collector import MetricsCollector, BenchmarkMetrics, RequestMetrics
from ..instance_manager.vllm_instance import VLLMInstance, GenerationParams
from ..instance_manager.instance_pool import InstancePool, PoolConfig
from ..instance_manager.gpu_allocator import GPUAllocator
from ..scheduler.base_scheduler import BaseScheduler, SchedulerRequest, create_scheduler_request
from ..scheduler.homogeneous import HomogeneousScheduler, create_homogeneous_scheduler
from ..scheduler.heterogeneous import HeterogeneousScheduler, create_heterogeneous_scheduler
from .scenarios import ScenarioConfig, ScenarioType

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkRunResult:
    """Result of a single benchmark run."""
    run_id: int
    scenario_name: str
    metrics: BenchmarkMetrics
    request_results: List[RequestMetrics]
    scheduler_stats: Dict[str, Any]
    duration: float
    success: bool
    error_message: str = ""


@dataclass
class ScenarioResult:
    """Aggregated result for a scenario (multiple runs)."""
    scenario: ScenarioConfig
    runs: List[BenchmarkRunResult]
    avg_throughput: float
    avg_latency: float
    avg_ttft: float
    avg_tpot: float
    total_requests: int
    success_rate: float


class BenchmarkRunner:
    """
    Executes benchmark scenarios.
    
    Features:
    - Concurrent request handling with semaphore
    - Warmup phase before actual benchmark
    - Progress tracking with tqdm
    - Result collection and saving
    - Support for both homogeneous and heterogeneous scenarios
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        dataset_loader: GPQADatasetLoader,
        sequence_profiler: SequenceProfiler,
        output_dir: str = "./results"
    ):
        """
        Initialize the benchmark runner.
        
        Args:
            config: Configuration dictionary
            dataset_loader: Loaded dataset
            sequence_profiler: Profiler with sequence information
            output_dir: Directory for results
        """
        self.config = config
        self.dataset_loader = dataset_loader
        self.sequence_profiler = sequence_profiler
        self.output_dir = Path(output_dir)
        
        # Get sequences sorted by length for profiling
        self.sequences = sequence_profiler.get_sequences_sorted_by_length()
        self.questions = {q.question_id: q for q in dataset_loader.get_questions()}
        
        # Random seed for reproducibility
        self.random_seed = config.get("benchmark", {}).get("random_seed", 42)
        random.seed(self.random_seed)
        
        # Generation parameters
        gen_config = config.get("benchmark", {}).get("generation", {})
        self.generation_params = GenerationParams(
            max_tokens=gen_config.get("max_new_tokens", 512),
            temperature=gen_config.get("temperature", 0.7),
            top_p=gen_config.get("top_p", 0.9),
            stream=True
        )
        
        # Results storage
        self.results: Dict[str, ScenarioResult] = {}
        
        logger.info(f"BenchmarkRunner initialized with {len(self.sequences)} sequences")
    
    async def run_scenario(
        self,
        scenario: ScenarioConfig,
        save_results: bool = True
    ) -> ScenarioResult:
        """
        Run a complete benchmark scenario.
        
        Args:
            scenario: Scenario configuration
            save_results: Whether to save results to disk
            
        Returns:
            ScenarioResult with aggregated metrics
        """
        logger.info(f"Starting scenario: {scenario.name}")
        logger.info(f"  Type: {scenario.scenario_type.value}")
        logger.info(f"  Instances: {len(scenario.instances)}")
        logger.info(f"  Runs: {scenario.num_runs}")
        
        runs: List[BenchmarkRunResult] = []
        
        # Create output directory
        scenario_dir = self.output_dir / scenario.name
        scenario_dir.mkdir(parents=True, exist_ok=True)
        
        # Save environment info
        self._save_environment_info(scenario_dir)
        
        # Run multiple times
        for run_id in range(1, scenario.num_runs + 1):
            logger.info(f"\n{'='*50}")
            logger.info(f"Starting run {run_id}/{scenario.num_runs}")
            logger.info(f"{'='*50}")
            
            try:
                result = await self._execute_single_run(scenario, run_id)
                runs.append(result)
                
                if save_results:
                    self._save_run_result(scenario_dir, result)
                
            except Exception as e:
                logger.error(f"Run {run_id} failed: {e}")
                runs.append(BenchmarkRunResult(
                    run_id=run_id,
                    scenario_name=scenario.name,
                    metrics=None,
                    request_results=[],
                    scheduler_stats={},
                    duration=0,
                    success=False,
                    error_message=str(e)
                ))
        
        # Calculate aggregated results
        scenario_result = self._aggregate_runs(scenario, runs)
        self.results[scenario.name] = scenario_result
        
        if save_results:
            self._save_scenario_result(scenario_dir, scenario_result)
        
        logger.info(f"\nScenario {scenario.name} completed")
        self._print_scenario_summary(scenario_result)
        
        return scenario_result
    
    async def _execute_single_run(
        self,
        scenario: ScenarioConfig,
        run_id: int
    ) -> BenchmarkRunResult:
        """Execute a single benchmark run."""
        start_time = time.time()
        
        # Create instance pool
        pool_config = PoolConfig(
            instances=[
                {
                    "instance_id": inst.instance_id,
                    "tp": inst.tp_degree,
                    "gpus": inst.gpu_ids
                }
                for inst in scenario.instances
            ],
            model_config=self.config.get("model", {}),
            server_config=self.config.get("vllm_server", {}),
            startup_timeout=self.config.get("vllm_server", {}).get("startup_timeout", 300)
        )
        
        gpu_allocator = GPUAllocator(
            total_gpus=self.config.get("gpu", {}).get("total_gpus", 8)
        )
        
        pool = InstancePool(pool_config, gpu_allocator)
        
        # Create scheduler
        if scenario.scenario_type == ScenarioType.HETEROGENEOUS:
            scheduler = create_heterogeneous_scheduler(self.config)
            scheduler.update_thresholds(scenario.length_thresholds)
            for cat, tps in scenario.routing_rules.items():
                scheduler.update_routing_rule(cat, tps)
        else:
            # Homogeneous - use the TP degree from first instance
            tp_degree = scenario.instances[0].tp_degree if scenario.instances else 1
            scheduler = create_homogeneous_scheduler(self.config, tp_degree)
        
        # Create metrics collector
        all_gpu_ids = []
        for inst in scenario.instances:
            all_gpu_ids.extend(inst.gpu_ids)
        
        metrics_collector = MetricsCollector(
            gpu_ids=list(set(all_gpu_ids)),
            gpu_monitor_interval=self.config.get("metrics", {}).get("gpu_monitor_interval", 0.5),
            save_intermediate=True,
            output_dir=str(self.output_dir / scenario.name)
        )
        
        try:
            # Start instance pool
            logger.info("Starting instance pool...")
            success = await pool.start()
            
            if not success:
                raise RuntimeError("Failed to start instance pool")
            
            pool.print_status()
            
            # Run warmup
            logger.info(f"Running warmup ({scenario.warmup_requests} requests)...")
            await self._run_warmup(pool, scheduler, scenario.warmup_requests)
            
            # Start metrics collection
            metrics_collector.start_benchmark(scenario.name, run_id)
            
            # Run benchmark
            logger.info(f"Running benchmark ({len(self.sequences)} requests)...")
            request_results = await self._run_benchmark_requests(
                pool, scheduler, metrics_collector, scenario
            )
            
            # End metrics collection
            benchmark_metrics = metrics_collector.end_benchmark()
            
            # Get scheduler stats
            scheduler_stats = scheduler.get_statistics().to_dict()
            
            duration = time.time() - start_time
            
            return BenchmarkRunResult(
                run_id=run_id,
                scenario_name=scenario.name,
                metrics=benchmark_metrics,
                request_results=request_results,
                scheduler_stats=scheduler_stats,
                duration=duration,
                success=True
            )
            
        finally:
            # Always stop the pool
            logger.info("Stopping instance pool...")
            await pool.stop()
    
    async def _run_warmup(
        self,
        pool: InstancePool,
        scheduler: BaseScheduler,
        num_requests: int
    ) -> None:
        """Run warmup requests."""
        warmup_sequences = self.sequences[:num_requests]
        
        for seq in warmup_sequences:
            question = self.questions.get(seq.question_id)
            if not question:
                continue
            
            request = create_scheduler_request(
                request_id=f"warmup_{seq.question_id}",
                question_id=seq.question_id,
                prompt=question.prompt,
                input_tokens=seq.input_tokens,
                thresholds=self.config.get("scheduling", {}).get("length_thresholds")
            )
            
            # Route and execute
            result = await scheduler.route_request(request, pool.instances)
            
            if result.success:
                instance = pool.get_instance(result.instance_id)
                if instance:
                    # Execute with reduced tokens for warmup
                    warmup_params = GenerationParams(
                        max_tokens=32,
                        temperature=0.7,
                        stream=False
                    )
                    await instance.generate(question.prompt, warmup_params)
        
        logger.info("Warmup completed")
    
    async def _run_benchmark_requests(
        self,
        pool: InstancePool,
        scheduler: BaseScheduler,
        metrics_collector: MetricsCollector,
        scenario: ScenarioConfig
    ) -> List[RequestMetrics]:
        """Run benchmark requests with concurrency control."""
        semaphore = asyncio.Semaphore(scenario.max_concurrent_requests)
        results: List[RequestMetrics] = []
        results_lock = asyncio.Lock()
        
        async def process_request(seq: SequenceInfo, idx: int) -> None:
            async with semaphore:
                question = self.questions.get(seq.question_id)
                if not question:
                    return
                
                request_id = f"bench_{idx}_{seq.question_id}"
                
                # Create scheduler request
                request = create_scheduler_request(
                    request_id=request_id,
                    question_id=seq.question_id,
                    prompt=question.prompt,
                    input_tokens=seq.input_tokens,
                    thresholds=scenario.length_thresholds or 
                               self.config.get("scheduling", {}).get("length_thresholds")
                )
                
                # Route request
                routing_result = await scheduler.route_request(request, pool.instances)
                
                if not routing_result.success:
                    logger.warning(f"Failed to route request {request_id}")
                    return
                
                instance = pool.get_instance(routing_result.instance_id)
                if not instance:
                    logger.warning(f"Instance {routing_result.instance_id} not found")
                    return
                
                # Execute request with metrics tracking
                with metrics_collector.track_request(
                    request_id=request_id,
                    question_id=seq.question_id,
                    instance_id=routing_result.instance_id,
                    tp_degree=routing_result.tp_degree,
                    input_tokens=seq.input_tokens,
                    sequence_category=seq.category
                ) as tracker:
                    tracker.set_queue_time(routing_result.queue_time)
                    
                    # Generate
                    gen_result = await instance.generate(
                        question.prompt,
                        self.generation_params,
                        request_id=request_id
                    )
                    
                    if gen_result.first_token_time:
                        tracker.first_token_time = gen_result.first_token_time
                    
                    tracker.set_output_tokens(gen_result.output_tokens)
                    
                    if not gen_result.success:
                        tracker.set_error(gen_result.error_message)
        
        # Create tasks
        tasks = [
            process_request(seq, idx)
            for idx, seq in enumerate(self.sequences)
        ]
        
        # Run with progress bar
        with tqdm(total=len(tasks), desc="Benchmark progress") as pbar:
            for coro in asyncio.as_completed(tasks):
                await coro
                pbar.update(1)
        
        return metrics_collector.get_request_metrics()
    
    def _aggregate_runs(
        self,
        scenario: ScenarioConfig,
        runs: List[BenchmarkRunResult]
    ) -> ScenarioResult:
        """Aggregate results from multiple runs."""
        successful_runs = [r for r in runs if r.success and r.metrics]
        
        if not successful_runs:
            return ScenarioResult(
                scenario=scenario,
                runs=runs,
                avg_throughput=0,
                avg_latency=0,
                avg_ttft=0,
                avg_tpot=0,
                total_requests=0,
                success_rate=0
            )
        
        # Calculate averages
        throughputs = [r.metrics.throughput_tokens_per_second for r in successful_runs]
        latencies = [r.metrics.latency_mean for r in successful_runs]
        ttfts = [r.metrics.ttft_mean for r in successful_runs]
        tpots = [r.metrics.tpot_mean for r in successful_runs]
        total_reqs = sum(r.metrics.total_requests for r in successful_runs)
        success_reqs = sum(r.metrics.successful_requests for r in successful_runs)
        
        return ScenarioResult(
            scenario=scenario,
            runs=runs,
            avg_throughput=sum(throughputs) / len(throughputs),
            avg_latency=sum(latencies) / len(latencies),
            avg_ttft=sum(ttfts) / len(ttfts),
            avg_tpot=sum(tpots) / len(tpots),
            total_requests=total_reqs,
            success_rate=success_reqs / total_reqs if total_reqs > 0 else 0
        )
    
    def _save_environment_info(self, output_dir: Path) -> None:
        """Save environment information."""
        env_info = {
            "timestamp": datetime.now().isoformat(),
            "random_seed": self.random_seed,
            "config": self.config,
            "num_sequences": len(self.sequences),
            "python_version": sys.version,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "")
        }
        
        env_file = output_dir / "environment.json"
        with open(env_file, 'w', encoding='utf-8') as f:
            json.dump(env_info, f, indent=2, ensure_ascii=False, default=str)
    
    def _save_run_result(
        self,
        output_dir: Path,
        result: BenchmarkRunResult
    ) -> None:
        """Save a single run result."""
        run_file = output_dir / f"run_{result.run_id}.json"
        
        data = {
            "run_id": result.run_id,
            "scenario_name": result.scenario_name,
            "success": result.success,
            "duration": result.duration,
            "error_message": result.error_message,
            "metrics": result.metrics.to_dict() if result.metrics else None,
            "scheduler_stats": result.scheduler_stats,
            "request_count": len(result.request_results)
        }
        
        with open(run_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        # Save detailed request results
        requests_file = output_dir / f"run_{result.run_id}_requests.json"
        with open(requests_file, 'w', encoding='utf-8') as f:
            json.dump(
                [r.to_dict() for r in result.request_results],
                f, indent=2, ensure_ascii=False
            )
    
    def _save_scenario_result(
        self,
        output_dir: Path,
        result: ScenarioResult
    ) -> None:
        """Save aggregated scenario result."""
        summary_file = output_dir / "summary.json"
        
        data = {
            "scenario": result.scenario.to_dict(),
            "num_runs": len(result.runs),
            "successful_runs": sum(1 for r in result.runs if r.success),
            "avg_throughput_tokens_per_second": result.avg_throughput,
            "avg_latency_ms": result.avg_latency,
            "avg_ttft_ms": result.avg_ttft,
            "avg_tpot_ms": result.avg_tpot,
            "total_requests": result.total_requests,
            "success_rate": result.success_rate
        }
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def _print_scenario_summary(self, result: ScenarioResult) -> None:
        """Print scenario summary."""
        print("\n" + "=" * 60)
        print(f"Scenario Summary: {result.scenario.name}")
        print("=" * 60)
        print(f"Runs: {len(result.runs)} ({sum(1 for r in result.runs if r.success)} successful)")
        print(f"\nPerformance Metrics (averaged over {len([r for r in result.runs if r.success])} runs):")
        print(f"  Throughput: {result.avg_throughput:.2f} tokens/s")
        print(f"  Latency:    {result.avg_latency:.2f} ms (mean)")
        print(f"  TTFT:       {result.avg_ttft:.2f} ms (mean)")
        print(f"  TPOT:       {result.avg_tpot:.2f} ms (mean)")
        print(f"\nTotal Requests: {result.total_requests}")
        print(f"Success Rate:   {result.success_rate:.2%}")
        print("=" * 60)
    
    async def run_all_homogeneous(self) -> Dict[str, ScenarioResult]:
        """Run all predefined homogeneous scenarios."""
        from .scenarios import get_homogeneous_scenarios
        
        results = {}
        for scenario in get_homogeneous_scenarios():
            result = await self.run_scenario(scenario)
            results[scenario.name] = result
        
        return results
    
    async def run_all_heterogeneous(self) -> Dict[str, ScenarioResult]:
        """Run all predefined heterogeneous scenarios."""
        from .scenarios import get_heterogeneous_scenarios
        
        results = {}
        for scenario in get_heterogeneous_scenarios():
            result = await self.run_scenario(scenario)
            results[scenario.name] = result
        
        return results
    
    def get_results(self) -> Dict[str, ScenarioResult]:
        """Get all collected results."""
        return self.results


async def run_benchmark(
    config: Dict[str, Any],
    scenario_names: Optional[List[str]] = None,
    output_dir: str = "./results"
) -> Dict[str, ScenarioResult]:
    """
    Convenience function to run benchmarks.
    
    Args:
        config: Configuration dictionary
        scenario_names: List of scenario names to run (None for all)
        output_dir: Output directory
        
    Returns:
        Dictionary of scenario results
    """
    from .scenarios import SCENARIOS, create_scenario_from_config
    
    # Load dataset
    logger.info("Loading dataset...")
    dataset_loader = GPQADatasetLoader(
        dataset_name=config.get("dataset", {}).get("name", "Idavidrein/gpqa"),
        subset=config.get("dataset", {}).get("subset", "gpqa_diamond"),
        split=config.get("dataset", {}).get("split", "train"),
        cache_dir=config.get("dataset", {}).get("cache_dir"),
        max_samples=config.get("dataset", {}).get("max_samples")
    )
    dataset_loader.load()
    
    # Profile sequences
    logger.info("Profiling sequences...")
    profiler = SequenceProfiler(
        model_name=config.get("model", {}).get("name", "Qwen/Qwen3-32B"),
        thresholds=config.get("scheduling", {}).get("length_thresholds"),
        trust_remote_code=config.get("model", {}).get("trust_remote_code", True)
    )
    profiler.profile_questions(dataset_loader.get_questions())
    profiler.print_summary()
    
    # Create runner
    runner = BenchmarkRunner(
        config=config,
        dataset_loader=dataset_loader,
        sequence_profiler=profiler,
        output_dir=output_dir
    )
    
    # Determine scenarios to run
    if scenario_names is None:
        scenario_names = list(SCENARIOS.keys())
    
    # Run scenarios
    for name in scenario_names:
        scenario = create_scenario_from_config(config, name)
        if scenario:
            await runner.run_scenario(scenario)
        else:
            logger.warning(f"Unknown scenario: {name}")
    
    return runner.get_results()
