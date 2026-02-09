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

from src.profiler.dataset_loader import (
    GPQADatasetLoader,
    AIME25DatasetLoader,
    load_dataset_by_type
)
from src.profiler.sequence_profiler import SequenceProfiler
from src.benchmark.runner import BenchmarkRunner
from src.benchmark.scenarios import (
    HETEROGENEOUS_MIX_1,
    HETEROGENEOUS_MIX_2,
    HETEROGENEOUS_MIX_3,
    create_custom_heterogeneous_scenario,
    create_scenario_from_config,
    get_heterogeneous_scenarios,
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


async def run_heterogeneous_benchmark(
    config: dict,
    scenarios: list,
    output_dir: str,
    num_runs: int = 3,
    profiling_dir: str = None,
    save_vllm_logs: bool = True,
    exclude_tp1: bool = False,
    use_config_hetero: bool = False
):
    """Run heterogeneous benchmarks for specified scenarios.
    
    Args:
        config: Configuration dictionary
        scenarios: List of scenario configurations
        output_dir: Output directory for results
        num_runs: Number of runs per scenario
        profiling_dir: Directory containing profiling results for adaptive configuration
    """
    logger = logging.getLogger(__name__)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 如果提供了profiling目录且未强制使用配置文件，则使用自适应配置
    adaptive_configs = None
    if profiling_dir and Path(profiling_dir).exists() and not use_config_hetero:
        logger.info(f"Loading profiling results from: {profiling_dir}")
        try:
            profiling_loader = ProfilingResultsLoader(profiling_dir)
            profiling_loader.print_analysis_report()
            
            # 基于profiling结果创建自适应场景
            _, adaptive_heterogeneous = create_adaptive_scenarios_from_profiling(
                profiling_dir, 
                config.get("gpu", {}).get("total_gpus", 8),
                exclude_tp1=exclude_tp1
            )
            adaptive_configs = adaptive_heterogeneous
            
            logger.info(f"Created {len(adaptive_configs)} adaptive heterogeneous configurations")
            for cfg in adaptive_configs:
                logger.info(f"  - {cfg['name']}: {cfg['description']}")
                
        except Exception as e:
            logger.warning(f"Failed to load profiling results: {e}")
            logger.warning("Falling back to predefined scenarios")
    
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
    
    # Show sequence distribution for routing analysis
    # by_category = profiler.get_sequences_by_category()
    # logger.info("\nSequence distribution for routing:")
    # for cat, seqs in sorted(by_category.items()):
    #     logger.info(f"  {cat}: {len(seqs)} sequences")
    
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
        profiling_dir=profiling_dir
    )
    
    benchmark_config = config.get("benchmark", {})
    
    # 决定要运行的场景
    scenarios_to_run = []
    
    if adaptive_configs:
        # 使用自适应配置创建场景
        logger.info("\n" + "=" * 60)
        logger.info("Using Adaptive Configurations from Profiling Results")
        logger.info("=" * 60)
        
        for cfg in adaptive_configs:
            scenario = create_custom_heterogeneous_scenario(
                instance_configs=cfg["instances"],
                name=cfg["name"],
                description=cfg["description"],
                length_thresholds=cfg.get("length_thresholds"),
                routing_rules=cfg.get("routing_rules"),
                num_runs=num_runs,
                warmup_requests=benchmark_config.get("warmup_requests", 5),
                max_concurrent_requests=benchmark_config.get("max_concurrent_requests", 32)
            )
            scenarios_to_run.append(scenario)
    else:
        # 使用预定义场景
        logger.info("\n" + "=" * 60)
        logger.info("Using Predefined Scenarios")
        logger.info("=" * 60)
        scenarios_to_run = scenarios
    
    # Run benchmarks for each scenario
    for scenario in scenarios_to_run:
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


def get_config_heterogeneous_scenarios(config: dict) -> list:
    """从配置文件中获取异构场景配置。
    
    Args:
        config: 配置字典
        
    Returns:
        异构场景配置列表
    """
    logger = logging.getLogger(__name__)
    
    tp_configs = config.get("tp_configs", {})
    benchmark_config = config.get("benchmark", {})
    scheduling_config = config.get("scheduling", {})
    
    # 检查是否有异构配置
    heterogeneous_configs = tp_configs.get("heterogeneous", [])
    
    if not heterogeneous_configs:
        logger.info("No heterogeneous configurations found in config file")
        return []
    
    logger.info(f"Found {len(heterogeneous_configs)} heterogeneous instance configurations in config")
    
    # 转换配置格式以匹配create_custom_heterogeneous_scenario期望的格式
    instance_configs = []
    for idx, cfg in enumerate(heterogeneous_configs):
        # 从配置中提取必要的字段
        instance_config = {
            "tp": cfg.get("tp", 1),
            "gpus": cfg.get("gpus", [idx]),
            "instance_id": cfg.get("instance_id", f"config_hetero_{idx}")
        }
        instance_configs.append(instance_config)
        
        logger.info(f"  Instance {idx}: TP={instance_config['tp']}, GPUs={instance_config['gpus']}")
        if "description" in cfg:
            logger.info(f"    Description: {cfg['description']}")
    
    # 创建场景
    scenario = create_custom_heterogeneous_scenario(
        instance_configs=instance_configs,
        name="config_based_heterogeneous",
        description="Heterogeneous configuration loaded from config file",
        length_thresholds=scheduling_config.get("length_thresholds"),
        routing_rules=scheduling_config.get("routing_rules"),
        num_runs=benchmark_config.get("num_runs", 3),
        warmup_requests=benchmark_config.get("warmup_requests", 5),
        max_concurrent_requests=benchmark_config.get("max_concurrent_requests", 32)
    )
    
    return [scenario]


def get_predefined_scenarios(scenario_names: list, config: dict = None):
    """Get predefined scenarios by name, with support for config-based scenarios."""
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
            # 先添加预定义场景
            predefined = list(get_heterogeneous_scenarios())
            
            # 如果有配置文件且包含异构配置，则添加配置中的场景
            if config:
                config_scenarios = get_config_heterogeneous_scenarios(config)
                if config_scenarios:
                    predefined.extend(config_scenarios)
            
            return predefined
        
        # 首先检查预定义场景
        scenario = all_scenarios.get(name.lower())
        if scenario:
            scenarios.append(scenario)
        else:
            # 尝试从配置创建场景
            if config:
                config_scenario = create_scenario_from_config(config, name)
                if config_scenario and config_scenario.scenario_type.name == "HETEROGENEOUS":
                    scenarios.append(config_scenario)
                else:
                    print(f"Warning: Unknown scenario '{name}'")
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
        "--use-config-hetero",
        action="store_true",
        default=False,
        help="Use heterogeneous configurations from config file instead of predefined scenarios"
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
    elif args.use_config_hetero:
        # 使用配置文件中的异构配置
        config_scenarios = get_config_heterogeneous_scenarios(config)
        if not config_scenarios:
            print("Error: No heterogeneous configurations found in config file")
            sys.exit(1)
        scenarios = config_scenarios
    else:
        scenarios = get_predefined_scenarios(args.scenarios, config)
    
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
    if args.profiling_dir:
        logger.info(f"Profiling directory: {args.profiling_dir}")
    
    # 显示配置中的异构实例信息
    tp_configs = config.get("tp_configs", {})
    heterogeneous_configs = tp_configs.get("heterogeneous", [])
    if heterogeneous_configs:
        logger.info("\nConfig-based Heterogeneous Instances:")
        for idx, cfg in enumerate(heterogeneous_configs):
            logger.info(f"  Instance {idx}: TP={cfg.get('tp', 1)}, GPUs={cfg.get('gpus', [idx])}")
            if "description" in cfg:
                logger.info(f"    Description: {cfg['description']}")
    
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
            num_runs=args.runs,
            profiling_dir=args.profiling_dir,
            save_vllm_logs=args.save_vllm_logs,
            exclude_tp1=args.exclude_tp1,
            use_config_hetero=args.use_config_hetero
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
