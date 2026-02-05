#!/usr/bin/env python3
"""
Heterogeneous Benchmark Script

Runs benchmarks for heterogeneous TP configurations with intelligent
length-based routing.
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
    HETEROGENEOUS_MIX_1,
    HETEROGENEOUS_MIX_2,
    HETEROGENEOUS_MIX_3,
    create_custom_heterogeneous_scenario,
    get_heterogeneous_scenarios,
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


async def run_heterogeneous_benchmark(
    config: dict,
    scenarios: list,
    output_dir: str,
    num_runs: int = 3
):
    """Run heterogeneous benchmarks for specified scenarios."""
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
    
    # Show sequence distribution for routing analysis
    by_category = profiler.get_sequences_by_category()
    logger.info("\nSequence distribution for routing:")
    for cat, seqs in sorted(by_category.items()):
        logger.info(f"  {cat}: {len(seqs)} sequences")
    
    # Create benchmark runner
    runner = BenchmarkRunner(
        config=config,
        dataset_loader=loader,
        sequence_profiler=profiler,
        output_dir=str(output_path)
    )
    
    benchmark_config = config.get("benchmark", {})
    
    # Run benchmarks for each scenario
    for scenario in scenarios:
        logger.info("\n" + "=" * 60)
        logger.info(f"Running Heterogeneous Scenario: {scenario.name}")
        logger.info("=" * 60)
        
        # Update scenario settings
        scenario.num_runs = num_runs
        scenario.warmup_requests = benchmark_config.get("warmup_requests", 5)
        scenario.max_concurrent_requests = benchmark_config.get("max_concurrent_requests", 32)
        
        # Use thresholds from config if not set
        if not scenario.length_thresholds:
            scenario.length_thresholds = scheduling_config.get("length_thresholds", {
                "short": 256,
                "medium": 512,
                "long": 1024
            })
        
        # Use routing rules from config if not set
        if not scenario.routing_rules:
            scenario.routing_rules = scheduling_config.get("routing_rules", {
                "short": [1, 2],
                "medium": [2, 4],
                "long": [4, 8],
                "extra_long": [8, 4]
            })
        
        print_scenario_info(scenario)
        
        # Run benchmark
        try:
            result = await runner.run_scenario(scenario, save_results=True)
            logger.info(f"{scenario.name} completed: {result.avg_throughput:.2f} tokens/s")
            
            # Print routing statistics
            if result.runs:
                for run_result in result.runs:
                    if run_result.success and run_result.scheduler_stats:
                        stats = run_result.scheduler_stats
                        logger.info(f"  Routing stats (run {run_result.run_id}):")
                        if "preferred_routes" in stats:
                            logger.info(f"    Preferred routes: {stats.get('preferred_routes', 0)}")
                            logger.info(f"    Fallback routes: {stats.get('fallback_routes', 0)}")
                        if "by_tp_degree" in stats:
                            logger.info(f"    By TP: {stats['by_tp_degree']}")
                        
        except Exception as e:
            logger.error(f"{scenario.name} failed: {e}")
            import traceback
            traceback.print_exc()
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


def get_predefined_scenarios(scenario_names: list):
    """Get predefined scenarios by name."""
    all_scenarios = {
        "mix1": HETEROGENEOUS_MIX_1,
        "mix2": HETEROGENEOUS_MIX_2,
        "mix3": HETEROGENEOUS_MIX_3,
        "heterogeneous_mix_1": HETEROGENEOUS_MIX_1,
        "heterogeneous_mix_2": HETEROGENEOUS_MIX_2,
        "heterogeneous_mix_3": HETEROGENEOUS_MIX_3,
    }
    
    scenarios = []
    for name in scenario_names:
        if name.lower() == "all":
            return list(get_heterogeneous_scenarios())
        
        scenario = all_scenarios.get(name.lower())
        if scenario:
            scenarios.append(scenario)
        else:
            print(f"Warning: Unknown scenario '{name}'")
    
    return scenarios


def main():
    parser = argparse.ArgumentParser(
        description="Run heterogeneous TP configuration benchmarks with intelligent routing"
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
        default="results/heterogeneous",
        help="Output directory for benchmark results"
    )
    parser.add_argument(
        "--scenarios", "-s",
        type=str,
        nargs="+",
        default=["all"],
        help="Scenarios to run: all, mix1, mix2, mix3 (default: all)"
    )
    parser.add_argument(
        "--runs", "-n",
        type=int,
        default=3,
        help="Number of runs per scenario (default: 3)"
    )
    parser.add_argument(
        "--custom-config",
        type=str,
        default=None,
        help="Path to custom heterogeneous configuration JSON file"
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
    log_file = args.log_file or f"{args.output}/logs/heterogeneous_benchmark.log"
    setup_logging(args.log_level, log_file)
    
    logger = logging.getLogger(__name__)
    
    # Load config
    config_path = PROJECT_ROOT / args.config
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)
    
    config = load_config(str(config_path))
    
    # Get scenarios
    if args.custom_config:
        # Load custom configuration
        import json
        with open(args.custom_config, 'r') as f:
            custom_config = json.load(f)
        
        scenarios = [create_custom_heterogeneous_scenario(
            instance_configs=custom_config.get("instances", []),
            name=custom_config.get("name", "custom_heterogeneous"),
            description=custom_config.get("description", "Custom configuration"),
            length_thresholds=custom_config.get("length_thresholds"),
            routing_rules=custom_config.get("routing_rules")
        )]
    else:
        scenarios = get_predefined_scenarios(args.scenarios)
    
    if not scenarios:
        print("Error: No valid scenarios specified")
        sys.exit(1)
    
    # Print configuration
    logger.info("=" * 60)
    logger.info("Heterogeneous TP Benchmark Configuration")
    logger.info("=" * 60)
    logger.info(f"Model: {config.get('model', {}).get('name', 'N/A')}")
    logger.info(f"Total GPUs: {config.get('gpu', {}).get('total_gpus', 8)}")
    logger.info(f"Scenarios to run: {[s.name for s in scenarios]}")
    logger.info(f"Runs per scenario: {args.runs}")
    logger.info(f"Output directory: {args.output}")
    
    scheduling_config = config.get("scheduling", {})
    logger.info(f"Length thresholds: {scheduling_config.get('length_thresholds', {})}")
    logger.info(f"Routing rules: {scheduling_config.get('routing_rules', {})}")
    
    # Run benchmark
    output_dir = PROJECT_ROOT / args.output
    
    try:
        results = asyncio.run(run_heterogeneous_benchmark(
            config=config,
            scenarios=scenarios,
            output_dir=str(output_dir),
            num_runs=args.runs
        ))
        
        logger.info("\n" + "=" * 60)
        logger.info("Benchmark completed successfully!")
        logger.info("=" * 60)
        
        for name, result in results.items():
            logger.info(f"  {name}: {result.avg_throughput:.2f} tokens/s, "
                       f"{result.avg_latency:.2f} ms latency, "
                       f"{result.success_rate:.1%} success rate")
        
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
