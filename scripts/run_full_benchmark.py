#!/usr/bin/env python3
"""
Full Benchmark Script

Runs complete benchmark suite including:
1. Profiling phase
2. All homogeneous configurations
3. All heterogeneous configurations
4. Comprehensive comparison report
"""

import os
import sys
import asyncio
import argparse
import logging
import json
from pathlib import Path
from datetime import datetime

import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.profiler.dataset_loader import GPQADatasetLoader
from src.profiler.sequence_profiler import SequenceProfiler
from src.benchmark.runner import BenchmarkRunner
from src.benchmark.scenarios import (
    get_homogeneous_scenarios,
    get_heterogeneous_scenarios,
    create_custom_homogeneous_scenario,
    print_scenario_info
)
from src.analysis.aggregator import MetricsAggregator
from src.analysis.report_generator import ReportGenerator


def setup_logging(log_level: str = "INFO", log_file: str = None):
    """Setup logging configuration."""
    handlers = [logging.StreamHandler()]
    
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers
    )


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def save_environment_info(output_dir: Path, config: dict):
    """Save complete environment information."""
    import platform
    
    env_info = {
        "timestamp": datetime.now().isoformat(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "python_version": platform.python_version()
        },
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "all"),
        "config": config
    }
    
    # Try to get GPU info
    try:
        import pynvml
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        gpus = []
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode('utf-8')
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpus.append({
                "id": i,
                "name": name,
                "memory_total_gb": memory.total / (1024**3)
            })
        env_info["gpus"] = gpus
    except Exception:
        pass
    
    env_file = output_dir / "environment.json"
    with open(env_file, 'w', encoding='utf-8') as f:
        json.dump(env_info, f, indent=2, ensure_ascii=False)
    
    return env_info


async def run_full_benchmark(
    config: dict,
    output_dir: str,
    run_profiling: bool = True,
    run_homogeneous: bool = True,
    run_heterogeneous: bool = True,
    tp_degrees: list = None,
    num_runs: int = 3
):
    """Run the complete benchmark suite."""
    logger = logging.getLogger(__name__)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save environment info
    logger.info("=" * 70)
    logger.info("HETEROGENEOUS TP CONFIGURATION BENCHMARK SUITE")
    logger.info("=" * 70)
    
    env_info = save_environment_info(output_path, config)
    logger.info(f"Results directory: {output_path}")
    logger.info(f"Timestamp: {env_info['timestamp']}")
    
    # Load dataset
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 1: Loading and Profiling Dataset")
    logger.info("=" * 70)
    
    dataset_config = config.get("dataset", {})
    loader = GPQADatasetLoader(
        dataset_name=dataset_config.get("name", "Idavidrein/gpqa"),
        subset=dataset_config.get("subset", "gpqa_diamond"),
        split=dataset_config.get("split", "train"),
        cache_dir=dataset_config.get("cache_dir"),
        max_samples=dataset_config.get("max_samples")
    )
    questions = loader.load()
    
    logger.info(f"Loaded {len(questions)} questions from GPQA-Diamond")
    
    # Profile sequences
    model_config = config.get("model", {})
    scheduling_config = config.get("scheduling", {})
    
    profiler = SequenceProfiler(
        model_name=model_config.get("name", "Qwen/Qwen3-32B"),
        thresholds=scheduling_config.get("length_thresholds"),
        trust_remote_code=model_config.get("trust_remote_code", True)
    )
    profiler.profile_questions(questions)
    profiler.print_summary()
    
    # Save profiling results
    if run_profiling:
        profiling_dir = output_path / "profiling"
        profiling_dir.mkdir(exist_ok=True)
        profiler.export_profile(str(profiling_dir / "sequence_profile.json"))
        logger.info(f"Saved profiling results to: {profiling_dir}")
    
    # Create benchmark runner
    runner = BenchmarkRunner(
        config=config,
        dataset_loader=loader,
        sequence_profiler=profiler,
        output_dir=str(output_path)
    )
    
    all_results = {}
    total_gpus = config.get("gpu", {}).get("total_gpus", 8)
    benchmark_config = config.get("benchmark", {})
    
    # Phase 2: Homogeneous benchmarks
    if run_homogeneous:
        logger.info("\n" + "=" * 70)
        logger.info("PHASE 2: Homogeneous Configuration Benchmarks")
        logger.info("=" * 70)
        
        if tp_degrees is None:
            tp_degrees = [1, 2, 4, 8]
        
        for tp in tp_degrees:
            if total_gpus % tp != 0:
                logger.warning(f"Skipping TP={tp}: cannot evenly divide {total_gpus} GPUs")
                continue
            
            logger.info(f"\n--- Running Homogeneous TP={tp} ---")
            
            scenario = create_custom_homogeneous_scenario(
                tp_degree=tp,
                total_gpus=total_gpus,
                num_runs=num_runs,
                warmup_requests=benchmark_config.get("warmup_requests", 5),
                max_concurrent_requests=benchmark_config.get("max_concurrent_requests", 32)
            )
            
            try:
                result = await runner.run_scenario(scenario, save_results=True)
                all_results[scenario.name] = result
                logger.info(f"TP={tp} completed: {result.avg_throughput:.2f} tokens/s, "
                           f"{result.avg_latency:.2f} ms")
            except Exception as e:
                logger.error(f"TP={tp} failed: {e}")
    
    # Phase 3: Heterogeneous benchmarks
    if run_heterogeneous:
        logger.info("\n" + "=" * 70)
        logger.info("PHASE 3: Heterogeneous Configuration Benchmarks")
        logger.info("=" * 70)
        
        for scenario in get_heterogeneous_scenarios():
            logger.info(f"\n--- Running {scenario.name} ---")
            
            # Update scenario settings
            scenario.num_runs = num_runs
            scenario.warmup_requests = benchmark_config.get("warmup_requests", 5)
            scenario.max_concurrent_requests = benchmark_config.get("max_concurrent_requests", 32)
            
            if not scenario.length_thresholds:
                scenario.length_thresholds = scheduling_config.get("length_thresholds", {
                    "short": 256, "medium": 512, "long": 1024
                })
            
            if not scenario.routing_rules:
                scenario.routing_rules = scheduling_config.get("routing_rules", {
                    "short": [1, 2], "medium": [2, 4], "long": [4, 8]
                })
            
            try:
                result = await runner.run_scenario(scenario, save_results=True)
                all_results[scenario.name] = result
                logger.info(f"{scenario.name} completed: {result.avg_throughput:.2f} tokens/s, "
                           f"{result.avg_latency:.2f} ms")
            except Exception as e:
                logger.error(f"{scenario.name} failed: {e}")
    
    # Phase 4: Analysis and Report Generation
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 4: Analysis and Report Generation")
    logger.info("=" * 70)
    
    aggregator = MetricsAggregator(str(output_path))
    aggregator.load_all_scenarios()
    
    # Print summary comparison
    aggregator.print_summary()
    
    # Compare homogeneous vs heterogeneous
    homo_scenarios = [n for n in all_results.keys() if "homo" in n.lower()]
    hetero_scenarios = [n for n in all_results.keys() if "hetero" in n.lower()]
    
    if homo_scenarios and hetero_scenarios:
        logger.info("\n--- Homogeneous vs Heterogeneous Comparison ---")
        
        # Find best homogeneous
        best_homo = max(homo_scenarios, 
                       key=lambda n: all_results[n].avg_throughput if n in all_results else 0)
        
        for hetero in hetero_scenarios:
            comparison = aggregator.compare_scenarios(best_homo, hetero)
            if comparison:
                logger.info(f"\n{hetero} vs {best_homo}:")
                logger.info(f"  Throughput: {comparison.throughput_improvement:+.1f}%")
                logger.info(f"  Latency: {comparison.latency_improvement:+.1f}%")
                logger.info(f"  TTFT: {comparison.ttft_improvement:+.1f}%")
                logger.info(f"  TPOT: {comparison.tpot_improvement:+.1f}%")
    
    # Generate comprehensive report
    analysis_dir = output_path / "analysis"
    generator = ReportGenerator(aggregator, str(analysis_dir))
    report_path = generator.generate_full_report()
    
    # Export data
    aggregator.export_to_csv(str(analysis_dir / "all_results.csv"))
    aggregator.export_to_json(str(analysis_dir / "all_results.json"))
    
    # Print final summary
    logger.info("\n" + "=" * 70)
    logger.info("BENCHMARK SUITE COMPLETED")
    logger.info("=" * 70)
    
    logger.info(f"\nTotal scenarios tested: {len(all_results)}")
    
    if all_results:
        # Find best performers
        best_throughput = max(all_results.items(), 
                             key=lambda x: x[1].avg_throughput if x[1] else 0)
        best_latency = min(all_results.items(), 
                          key=lambda x: x[1].avg_latency if x[1] and x[1].avg_latency > 0 else float('inf'))
        
        logger.info(f"\nBest Throughput: {best_throughput[0]}")
        logger.info(f"  {best_throughput[1].avg_throughput:.2f} tokens/s")
        
        logger.info(f"\nBest Latency: {best_latency[0]}")
        logger.info(f"  {best_latency[1].avg_latency:.2f} ms")
    
    logger.info(f"\nResults saved to: {output_path}")
    logger.info(f"Report: {report_path}")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Run complete heterogeneous TP benchmark suite"
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output directory (default: results/benchmark_YYYYMMDD_HHMMSS)"
    )
    parser.add_argument(
        "--tp",
        type=int,
        nargs="+",
        default=None,
        help="TP degrees for homogeneous tests (default: 1 2 4 8)"
    )
    parser.add_argument(
        "--runs", "-n",
        type=int,
        default=3,
        help="Number of runs per scenario (default: 3)"
    )
    parser.add_argument(
        "--skip-profiling",
        action="store_true",
        help="Skip profiling phase"
    )
    parser.add_argument(
        "--skip-homogeneous",
        action="store_true",
        help="Skip homogeneous benchmarks"
    )
    parser.add_argument(
        "--skip-heterogeneous",
        action="store_true",
        help="Skip heterogeneous benchmarks"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Setup output directory
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = PROJECT_ROOT / f"results/benchmark_{timestamp}"
    else:
        output_dir = PROJECT_ROOT / args.output
    
    # Setup logging
    log_file = output_dir / "logs" / "benchmark.log"
    setup_logging(args.log_level, str(log_file))
    
    logger = logging.getLogger(__name__)
    
    # Load config
    config_path = PROJECT_ROOT / args.config
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)
    
    config = load_config(str(config_path))
    
    # Run full benchmark
    try:
        results = asyncio.run(run_full_benchmark(
            config=config,
            output_dir=str(output_dir),
            run_profiling=not args.skip_profiling,
            run_homogeneous=not args.skip_homogeneous,
            run_heterogeneous=not args.skip_heterogeneous,
            tp_degrees=args.tp,
            num_runs=args.runs
        ))
        
        print(f"\n{'='*70}")
        print("Benchmark completed successfully!")
        print(f"Results: {output_dir}")
        print(f"{'='*70}")
        
    except KeyboardInterrupt:
        logger.info("\nBenchmark interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
