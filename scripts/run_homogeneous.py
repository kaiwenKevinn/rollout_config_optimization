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

from src.profiler.dataset_loader import (
    GPQADatasetLoader,
    AIME25DatasetLoader,
    load_dataset_by_type
)
from src.profiler.sequence_profiler import SequenceProfiler
from src.benchmark.runner import BenchmarkRunner
from src.benchmark.scenarios import (
    create_custom_homogeneous_scenario,
    get_homogeneous_scenarios,
    print_scenario_info
)
from src.analysis.aggregator import MetricsAggregator
from src.analysis.report_generator import ReportGenerator
from src.utils.profiling_loader import ProfilingResultsLoader, create_adaptive_scenarios_from_profiling


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
    num_runs: int = 3,
    profiling_dir: str = None,
    save_vllm_logs: bool = True,
    exclude_tp1: bool = False,
    exclude_tp2: bool = False
):
    """Run homogeneous benchmarks for specified TP degrees.
    
    Args:
        config: Configuration dictionary
        tp_degrees: List of TP degrees to test
        output_dir: Output directory for results
        num_runs: Number of runs per scenario
        profiling_dir: Directory containing profiling results for adaptive configuration
    """
    logger = logging.getLogger(__name__)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 如果提供了profiling目录，则使用自适应配置
    adaptive_configs = None
    if profiling_dir and Path(profiling_dir).exists():
        logger.info(f"Loading profiling results from: {profiling_dir}")
        try:
            profiling_loader = ProfilingResultsLoader(profiling_dir)
            profiling_loader.print_analysis_report()
            
            # 基于profiling结果创建自适应场景
            adaptive_homogeneous, _ = create_adaptive_scenarios_from_profiling(
                profiling_dir, 
                config.get("gpu", {}).get("total_gpus", 8),
                exclude_tp1=exclude_tp1,
                exclude_tp2=exclude_tp2
            )
            adaptive_configs = adaptive_homogeneous
            
            logger.info(f"Created {len(adaptive_configs)} adaptive homogeneous configurations")
            for cfg in adaptive_configs:
                logger.info(f"  - TP={cfg['tp']}: {cfg['instances']} instances - {cfg['description']}")
                
        except Exception as e:
            logger.warning(f"Failed to load profiling results: {e}")
            logger.warning("Falling back to manual TP degrees configuration")
    
    # Load dataset
    logger.info("=" * 60)
    logger.info("Loading dataset...")
    logger.info("=" * 60)
    
    # Use the factory function to load appropriate dataset
    loader = load_dataset_by_type(config)
    questions = loader.get_questions()
    
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
    # Add vLLM log directory to config if needed
    if save_vllm_logs:
        vllm_log_dir = str(output_path / "vllm_logs")
        config.setdefault("vllm_server", {})["log_dir"] = vllm_log_dir
        logger.info(f"vLLM logs will be saved to: {vllm_log_dir}")
    
    runner = BenchmarkRunner(
        config=config,
        dataset_loader=loader,
        sequence_profiler=profiler,
        output_dir=str(output_path),
        profiling_dir=profiling_dir  # 传递profiling目录给runner
    )
    
    # Run benchmarks for each TP degree
    total_gpus = config.get("gpu", {}).get("total_gpus", 8)
    benchmark_config = config.get("benchmark", {})
    
    # 决定要运行的配置
    configs_to_run = []
    
    if adaptive_configs:
        # 使用自适应配置
        logger.info("\n" + "=" * 60)
        logger.info("Using Adaptive Configurations from Profiling Results")
        logger.info("=" * 60)
        configs_to_run = adaptive_configs
    else:
        # 使用手动指定的TP度数
        logger.info("\n" + "=" * 60)
        logger.info("Using Manual TP Degree Configuration")
        logger.info("=" * 60)
        configs_to_run = [{"tp": tp} for tp in tp_degrees]
    
    for config_item in configs_to_run:
        tp = config_item["tp"]
        description = config_item.get("description", f"Manual TP={tp} configuration")
        
        # 检查是否需要排除TP=1或TP=2
        if exclude_tp1 and tp == 1:
            logger.info(f"Skipping TP={tp}: excluded by --exclude-tp1 flag")
            continue
        if exclude_tp2 and tp == 2:
            logger.info(f"Skipping TP={tp}: excluded by --exclude-tp2 flag")
            continue
            
        if total_gpus % tp != 0:
            logger.warning(f"Skipping TP={tp}: cannot evenly divide {total_gpus} GPUs")
            continue
        
        logger.info("\n" + "=" * 60)
        logger.info(f"Running Homogeneous TP={tp} Benchmark")
        logger.info(f"Configuration: {description}")
        logger.info("=" * 60)
        
        # Create scenario
        try:
            scenario = create_custom_homogeneous_scenario(
                tp_degree=tp,
                total_gpus=total_gpus,
                num_runs=num_runs,
                warmup_requests=benchmark_config.get("warmup_requests", 5),
                max_concurrent_requests=benchmark_config.get("max_concurrent_requests", 32)
            )
        except ValueError as e:
            logger.warning(f"Skipping TP={tp}: {e}")
            continue
        
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
        default=[1, 2, 4],
        help="TP degrees to test (default: 1 2 4)"
    )
    parser.add_argument(
        "--profiling-dir", "-p",
        type=str,
        default=None,
        help="Directory containing profiling results for adaptive configuration"
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
    parser.add_argument(
        "--save-vllm-logs",
        action="store_true",
        default=True,
        help="Save vLLM instance logs to files (default: True)"
    )
    parser.add_argument(
        "--no-save-vllm-logs",
        dest="save_vllm_logs",
        action="store_false",
        help="Do not save vLLM instance logs"
    )
    parser.add_argument(
        "--exclude-tp1",
        action="store_true",
        default=False,
        help="Exclude TP=1 configurations when using adaptive mode"
    )
    parser.add_argument(
        "--exclude-tp2",
        action="store_true",
        default=False,
        help="Exclude TP=2 configurations when using adaptive mode"
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
    if args.profiling_dir:
        logger.info(f"Profiling directory: {args.profiling_dir}")
    
    # Run benchmark
    output_dir = PROJECT_ROOT / args.output
    
    try:
        results = asyncio.run(run_homogeneous_benchmark(
            config=config,
            tp_degrees=args.tp,
            output_dir=str(output_dir),
            num_runs=args.runs,
            profiling_dir=args.profiling_dir,
            save_vllm_logs=args.save_vllm_logs,
            exclude_tp1=args.exclude_tp1,
            exclude_tp2=args.exclude_tp2
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
