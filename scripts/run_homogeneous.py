#!/usr/bin/env python3
"""
Homogeneous Benchmark Script

Runs benchmarks for homogeneous TP configurations (all instances have same TP degree).
"""

import os
import sys
import asyncio
import argparse
import logging
from pathlib import Path

import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.profiler.dataset_loader import GPQADatasetLoader
from src.profiler.sequence_profiler import SequenceProfiler
from src.benchmark.runner import BenchmarkRunner
from src.benchmark.scenarios import (
    create_custom_homogeneous_scenario,
    get_homogeneous_scenarios,
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


async def run_homogeneous_benchmark(
    config: dict,
    tp_degrees: list,
    output_dir: str,
    num_runs: int = 3
):
    """Run homogeneous benchmarks for specified TP degrees."""
    logger = logging.getLogger(__name__)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    logger.info("=" * 60)
    logger.info("Loading dataset...")
    logger.info("=" * 60)
    
    dataset_config = config.get("dataset", {})
    loader = GPQADatasetLoader(
        dataset_name=dataset_config.get("name", "Idavidrein/gpqa"),
        subset=dataset_config.get("subset", "gpqa_diamond"),
        split=dataset_config.get("split", "train"),
        cache_dir=dataset_config.get("cache_dir"),
        max_samples=dataset_config.get("max_samples")
    )
    loader.load()
    
    # Profile sequences
    logger.info("\n" + "=" * 60)
    logger.info("Profiling sequences...")
    logger.info("=" * 60)
    
    model_config = config.get("model", {})
    scheduling_config = config.get("scheduling", {})
    
    profiler = SequenceProfiler(
        model_name=model_config.get("name", "Qwen/Qwen3-32B"),
        thresholds=scheduling_config.get("length_thresholds"),
        trust_remote_code=model_config.get("trust_remote_code", True)
    )
    profiler.profile_questions(loader.get_questions())
    profiler.print_summary()
    
    # Create benchmark runner
    runner = BenchmarkRunner(
        config=config,
        dataset_loader=loader,
        sequence_profiler=profiler,
        output_dir=str(output_path)
    )
    
    # Run benchmarks for each TP degree
    total_gpus = config.get("gpu", {}).get("total_gpus", 8)
    benchmark_config = config.get("benchmark", {})
    
    for tp in tp_degrees:
        if total_gpus % tp != 0:
            logger.warning(f"Skipping TP={tp}: cannot evenly divide {total_gpus} GPUs")
            continue
        
        logger.info("\n" + "=" * 60)
        logger.info(f"Running Homogeneous TP={tp} Benchmark")
        logger.info("=" * 60)
        
        # Create scenario
        scenario = create_custom_homogeneous_scenario(
            tp_degree=tp,
            total_gpus=total_gpus,
            num_runs=num_runs,
            warmup_requests=benchmark_config.get("warmup_requests", 5),
            max_concurrent_requests=benchmark_config.get("max_concurrent_requests", 32)
        )
        
        print_scenario_info(scenario)
        
        # Run benchmark
        try:
            result = await runner.run_scenario(scenario, save_results=True)
            logger.info(f"TP={tp} completed: {result.avg_throughput:.2f} tokens/s")
        except Exception as e:
            logger.error(f"TP={tp} failed: {e}")
            continue
    
    # Generate analysis report
    logger.info("\n" + "=" * 60)
    logger.info("Generating analysis report...")
    logger.info("=" * 60)
    
    aggregator = MetricsAggregator(str(output_path))
    aggregator.load_all_scenarios()
    aggregator.print_summary()
    
    generator = ReportGenerator(aggregator, str(output_path / "analysis"))
    report_path = generator.generate_full_report()
    
    logger.info(f"\nReport generated: {report_path}")
    
    return runner.get_results()


def main():
    parser = argparse.ArgumentParser(
        description="Run homogeneous TP configuration benchmarks"
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
        default="results/homogeneous",
        help="Output directory for benchmark results"
    )
    parser.add_argument(
        "--tp",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8],
        help="TP degrees to test (default: 1 2 4 8)"
    )
    parser.add_argument(
        "--runs", "-n",
        type=int,
        default=3,
        help="Number of runs per scenario (default: 3)"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Log file path (optional)"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_file = args.log_file or f"{args.output}/logs/homogeneous_benchmark.log"
    setup_logging(args.log_level, log_file)
    
    logger = logging.getLogger(__name__)
    
    # Load config
    config_path = PROJECT_ROOT / args.config
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)
    
    config = load_config(str(config_path))
    
    # Print configuration
    logger.info("=" * 60)
    logger.info("Homogeneous TP Benchmark Configuration")
    logger.info("=" * 60)
    logger.info(f"Model: {config.get('model', {}).get('name', 'N/A')}")
    logger.info(f"Total GPUs: {config.get('gpu', {}).get('total_gpus', 8)}")
    logger.info(f"TP Degrees to test: {args.tp}")
    logger.info(f"Runs per scenario: {args.runs}")
    logger.info(f"Output directory: {args.output}")
    
    # Run benchmark
    output_dir = PROJECT_ROOT / args.output
    
    try:
        results = asyncio.run(run_homogeneous_benchmark(
            config=config,
            tp_degrees=args.tp,
            output_dir=str(output_dir),
            num_runs=args.runs
        ))
        
        logger.info("\n" + "=" * 60)
        logger.info("Benchmark completed successfully!")
        logger.info("=" * 60)
        
        for name, result in results.items():
            logger.info(f"  {name}: {result.avg_throughput:.2f} tokens/s, "
                       f"{result.avg_latency:.2f} ms latency")
        
    except KeyboardInterrupt:
        logger.info("\nBenchmark interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
