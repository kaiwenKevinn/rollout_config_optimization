#!/usr/bin/env python3
"""
Profiling Script with Actual Inference

Runs the profiling phase to analyze dataset sequences with actual model inference.
This script performs real generation to obtain accurate output token counts,
enabling precise total sequence length calculation for TP routing decisions.
"""

import os
import sys
import json
import asyncio
import argparse
import logging
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

import yaml
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def setup_hf_environment():
    """Setup HuggingFace environment and login."""
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    hf_token = os.environ.get("HF_TOKEN", "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")

    # Only login if token is available and valid
    if hf_token:
        try:
            from huggingface_hub import login
            login(token=hf_token, add_to_git_credential=False)
            print("HuggingFace login successful")
        except Exception as e:
            print(f"Warning: HuggingFace login failed: {e}")
            print("Continuing without login (may affect private model/dataset access)")


# Import project modules after setting up path
from src.profiler.dataset_loader import (
    BaseDatasetLoader, GPQADatasetLoader, AIME25DatasetLoader, 
    create_dataset_loader, load_dataset_by_type
)
from src.profiler.sequence_profiler import SequenceProfiler, SequenceInfo
from src.instance_manager.vllm_instance import (
    VLLMInstance, InstanceConfig, GenerationParams, GenerationResult
)


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


def run_profiling(config: dict, output_dir: str):
    """Run the profiling phase (estimation only, no actual inference)."""
    logger = logging.getLogger(__name__)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    logger.info("=" * 60)
    dataset_config = config.get("dataset", {})
    dataset_type = dataset_config.get("type", "gpqa")
    
    if dataset_type.lower() == "gpqa":
        logger.info("Loading GPQA-Diamond dataset...")
        loader = load_dataset_by_type(config)
    elif dataset_type.lower() == "aime25":
        logger.info("Loading AIME 25 dataset...")
        loader = load_dataset_by_type(config)
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")
    
    logger.info("=" * 60)
    
    questions = loader.load()
    stats = loader.get_statistics()
    logger.info(f"Avg question length: {stats.avg_question_length:.1f} chars")
    
    # Profile sequences (estimation only)
    logger.info("\n" + "=" * 60)
    logger.info("Profiling sequences (input token estimation)...")
    logger.info("=" * 60)
    
    model_config = config.get("model", {})
    scheduling_config = config.get("scheduling", {})
    
    profiler = SequenceProfiler(
        model_name=model_config.get("name", "Qwen/Qwen3-32B"),
        thresholds=scheduling_config.get("length_thresholds", {
            "short": 256,
            "medium": 512,
            "long": 1024
        }),
        total_thresholds=scheduling_config.get("total_length_thresholds", {
            "short": 512,
            "medium": 1024,
            "long": 2048
        }),
        trust_remote_code=model_config.get("trust_remote_code", True),
        cache_dir=dataset_config.get("cache_dir")
    )
    
    sequences = profiler.profile_questions(questions)
    
    # Print summary
    profiler.print_summary()
    
    # Save results
    _save_profiling_results(profiler, loader, output_path, logger)
    
    return {
        "total_questions": len(questions),
        "profiler": profiler,
        "loader": loader,
        "output_dir": str(output_path)
    }


async def run_profiling_with_inference(
    config: dict, 
    output_dir: str,
    tp_degree: int = 4,
    gpu_ids: Optional[List[int]] = None,
    max_concurrent: int = 8,
    save_intermediate: bool = True,
    request_timeout: int = 600
):
    """
    Run profiling with actual model inference to get real output token counts.
    
    This function:
    1. Loads and tokenizes all prompts (input token estimation)
    2. Starts a vLLM instance for inference
    3. Generates responses for all prompts
    4. Updates sequences with actual output token counts
    5. Recategorizes sequences based on total length (input + output)
    
    Args:
        config: Configuration dictionary
        output_dir: Output directory for results
        tp_degree: Tensor parallelism degree for the inference instance
        gpu_ids: GPU IDs to use (defaults to config or [0,1,2,3])
        max_concurrent: Maximum concurrent requests
        save_intermediate: Whether to save intermediate results
        
    Returns:
        Dict with profiling results including actual token statistics
    """
    logger = logging.getLogger(__name__)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # ========== Phase 1: Load dataset and estimate input tokens ==========
    logger.info("=" * 60)
    dataset_config = config.get("dataset", {})
    dataset_type = dataset_config.get("type", "gpqa")
    
    if dataset_type.lower() == "gpqa":
        logger.info("Phase 1: Loading GPQA-Diamond dataset...")
        loader = load_dataset_by_type(config)
    elif dataset_type.lower() == "aime25":
        logger.info("Phase 1: Loading AIME 25 dataset...")
        loader = load_dataset_by_type(config)
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")
    
    logger.info("=" * 60)
    
    questions = loader.load()
    stats = loader.get_statistics()
    
    # ========== Phase 2: Profile input tokens ==========
    logger.info("\n" + "=" * 60)
    logger.info("Phase 2: Profiling input sequences...")
    logger.info("=" * 60)
    
    model_config = config.get("model", {})
    scheduling_config = config.get("scheduling", {})
    
    profiler = SequenceProfiler(
        model_name=model_config.get("name", "Qwen/Qwen3-32B"),
        thresholds=scheduling_config.get("length_thresholds", {
            "short": 256,
            "medium": 512,
            "long": 1024
        }),
        total_thresholds=scheduling_config.get("total_length_thresholds", {
            "short": 512,
            "medium": 1024,
            "long": 2048
        }),
        trust_remote_code=model_config.get("trust_remote_code", True),
        cache_dir=dataset_config.get("cache_dir")
    )
    
    sequences = profiler.profile_questions(questions)
    logger.info(f"Profiled {len(sequences)} sequences")
    
    # Print pre-inference summary
    logger.info("\nPre-inference distribution (input-based):")
    profiler.print_summary()
    
    # ========== Phase 3: Start vLLM instance ==========
    logger.info("\n" + "=" * 60)
    logger.info("Phase 3: Starting vLLM instance for inference...")
    logger.info("=" * 60)
    
    gpu_config = config.get("gpu", {})
    server_config = config.get("vllm_server", {})
    
    # Determine GPU IDs
    if gpu_ids is None:
        available_gpus = gpu_config.get("available_gpus", [0, 1, 2, 3])
        gpu_ids = available_gpus[:tp_degree]
    else:
        # User specified GPUs, but we need to match tp_degree
        if len(gpu_ids) < tp_degree:
            raise ValueError(f"Not enough GPUs specified. Need {tp_degree} GPUs for TP={tp_degree}, but only got {len(gpu_ids)}: {gpu_ids}")
        elif len(gpu_ids) > tp_degree:
            logger.warning(f"More GPUs specified ({len(gpu_ids)}) than needed for TP={tp_degree}. Using first {tp_degree} GPUs: {gpu_ids[:tp_degree]}")
            gpu_ids = gpu_ids[:tp_degree]
    
    instance_config = InstanceConfig(
        instance_id="profiling_instance",
        tp_degree=tp_degree,
        gpu_ids=gpu_ids,
        port=server_config.get("base_port", 8000),
        host=server_config.get("host", "127.0.0.1"),
        model_name=model_config.get("name", "Qwen/Qwen3-32B"),
        max_model_len=model_config.get("max_model_len", 8192),
        dtype=model_config.get("dtype", "auto"),
        trust_remote_code=model_config.get("trust_remote_code", True),
        gpu_memory_utilization=model_config.get("gpu_memory_utilization", 0.90),
        log_dir=output_dir  # Save vLLM logs to output directory
    )
    
    # Log the configuration for debugging
    logger.info(f"Instance config: model={instance_config.model_name}")
    logger.info(f"Instance config: max_model_len={instance_config.max_model_len}")
    logger.info(f"Instance config: tp_degree={instance_config.tp_degree}, gpu_ids={instance_config.gpu_ids}")
    
    instance = VLLMInstance(instance_config)
    
    try:
        # Start the instance
        startup_timeout = server_config.get("startup_timeout", 300)
        logger.info(f"Starting vLLM instance (TP={tp_degree}, GPUs={gpu_ids})...")
        logger.info(f"This may take several minutes for large models...")
        
        success = await instance.start(timeout=startup_timeout)
        if not success:
            raise RuntimeError("Failed to start vLLM instance")
        
        logger.info("vLLM instance started successfully")
        
        # ========== Phase 4: Run inference on all sequences ==========
        logger.info("\n" + "=" * 60)
        logger.info("Phase 4: Running inference to get actual output lengths...")
        logger.info("=" * 60)
        
        benchmark_config = config.get("benchmark", {})
        generation_config = benchmark_config.get("generation", {})
        
        gen_params = GenerationParams(
            max_tokens=generation_config.get("max_new_tokens", 512),
            temperature=generation_config.get("temperature", 0.7),
            top_p=generation_config.get("top_p", 0.9),
            top_k=generation_config.get("top_k", 50),
            stream=False  # Use non-streaming for accurate token counting
        )
        
        # Run inference with concurrency control
        results = await _run_inference_batch(
            instance=instance,
            sequences=sequences,
            gen_params=gen_params,
            max_concurrent=max_concurrent,
            request_timeout=request_timeout,
            logger=logger
        )
        
        # ========== Phase 5: Update sequences with actual results ==========
        logger.info("\n" + "=" * 60)
        logger.info("Phase 5: Updating sequences with actual output lengths...")
        logger.info("=" * 60)
        
        update_results = []
        for seq, result in zip(sequences, results):
            if result.success:
                update_results.append({
                    "question_id": seq.question_id,
                    "output_tokens": result.output_tokens,
                    "generation_time_ms": result.total_time * 1000
                })
        
        updated_count = profiler.batch_update_with_results(update_results)
        logger.info(f"Updated {updated_count}/{len(sequences)} sequences with actual results")
        
        # Print post-inference summary
        logger.info("\nPost-inference distribution (total length-based):")
        profiler.print_summary()
        
    finally:
        # ========== Cleanup ==========
        logger.info("\nStopping vLLM instance...")
        await instance.stop()
        logger.info("vLLM instance stopped")
    
    # ========== Phase 6: Save results ==========
    logger.info("\n" + "=" * 60)
    logger.info("Phase 6: Saving profiling results...")
    logger.info("=" * 60)
    
    _save_profiling_results(profiler, loader, output_path, logger, include_actual=True)
    
    # Save inference results detail
    inference_results_file = output_path / "inference_results.json"
    inference_data = []
    for seq, result in zip(sequences, results):
        inference_data.append({
            "question_id": seq.question_id,
            "input_tokens": seq.input_tokens,
            "actual_output_tokens": seq.actual_output_tokens,
            "actual_total_tokens": seq.actual_total_tokens,
            "estimated_output_tokens": seq.estimated_output_tokens,
            "category_input_based": seq.category,
            "category_total_based": seq.actual_category,
            "generation_time_ms": seq.generation_time_ms,
            "success": result.success,
            "error_message": result.error_message if not result.success else None
        })
    
    with open(inference_results_file, 'w', encoding='utf-8') as f:
        json.dump(inference_data, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved inference results to: {inference_results_file}")
    
    # Print routing recommendations summary
    _print_routing_summary(profiler, logger)
    
    logger.info("\n" + "=" * 60)
    logger.info("Profiling with inference completed!")
    logger.info(f"Results saved to: {output_path}")
    logger.info("=" * 60)
    
    return {
        "total_questions": len(questions),
        "completed_sequences": updated_count,
        "profiler": profiler,
        "loader": loader,
        "output_dir": str(output_path)
    }


async def _run_inference_batch(
    instance: VLLMInstance,
    sequences: List[SequenceInfo],
    gen_params: GenerationParams,
    max_concurrent: int,
    request_timeout: int,
    logger: logging.Logger
) -> List[GenerationResult]:
    """
    Run inference on a batch of sequences with concurrency control.
    
    Args:
        instance: VLLMInstance to use
        sequences: List of sequences to process
        gen_params: Generation parameters
        max_concurrent: Maximum concurrent requests
        logger: Logger instance
        
    Returns:
        List of GenerationResult objects
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    results: List[Optional[GenerationResult]] = [None] * len(sequences)
    
    async def process_sequence(idx: int, seq: SequenceInfo):
        async with semaphore:
            logger.info(f"Processing sequence {idx+1}/{len(sequences)}: {seq.question_id}")
            start_time = time.time()
            try:
                result = await asyncio.wait_for(
                    instance.generate(
                        prompt=seq.prompt,
                        params=gen_params,
                        request_id=seq.question_id
                    ),
                    timeout=request_timeout
                )
                elapsed = time.time() - start_time
                logger.info(f"Sequence {idx+1} completed in {elapsed:.1f}s with {result.output_tokens} output tokens")
                results[idx] = result
            except asyncio.TimeoutError:
                elapsed = time.time() - start_time
                logger.error(f"Sequence {idx+1} timed out after {elapsed:.1f}s")
                results[idx] = GenerationResult(
                    request_id=seq.question_id,
                    prompt=seq.prompt,
                    generated_text="",
                    input_tokens=0,
                    output_tokens=0,
                    finish_reason="timeout",
                    total_time=elapsed,
                    success=False,
                    error_message=f"Request timeout after {request_timeout}s"
                )
            except Exception as e:
                elapsed = time.time() - start_time
                logger.error(f"Sequence {idx+1} failed after {elapsed:.1f}s: {e}")
                results[idx] = GenerationResult(
                    request_id=seq.question_id,
                    prompt=seq.prompt,
                    generated_text="",
                    input_tokens=0,
                    output_tokens=0,
                    finish_reason="error",
                    total_time=elapsed,
                    success=False,
                    error_message=str(e)
                )
    
    # Create tasks
    tasks = [
        process_sequence(idx, seq) 
        for idx, seq in enumerate(sequences)
    ]
    
    # Run with progress bar
    logger.info(f"Processing {len(sequences)} sequences with max {max_concurrent} concurrent requests...")
    
    # Use tqdm for progress tracking
    start_time = time.time()
    completed = 0
    
    for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Inference"):
        await coro
        completed += 1
    
    elapsed = time.time() - start_time
    logger.info(f"Completed {completed} inferences in {elapsed:.1f}s ({completed/elapsed:.1f} req/s)")
    
    # Count successes/failures
    successes = sum(1 for r in results if r and r.success)
    failures = len(results) - successes
    logger.info(f"Success: {successes}, Failed: {failures}")
    
    return results


def _save_profiling_results(
    profiler: SequenceProfiler,
    loader: GPQADatasetLoader,
    output_path: Path,
    logger: logging.Logger,
    include_actual: bool = False
):
    """Save profiling results to files."""
    
    # Save sequence distribution
    distribution = profiler.get_distribution()
    dist_data = {
        "timestamp": datetime.now().isoformat(),
        "total_sequences": distribution.total_sequences,
        "completed_sequences": distribution.completed_count,
        "by_category_input_based": distribution.by_category,
        "by_subject": distribution.by_subject,
        "input_token_stats": distribution.input_token_stats,
        "estimated_output_stats": distribution.estimated_output_stats,
        "thresholds_input": distribution.thresholds,
        "thresholds_total": profiler.total_thresholds
    }
    
    if include_actual and distribution.actual_total_stats:
        dist_data["by_category_total_based"] = distribution.by_actual_category
        dist_data["actual_output_stats"] = distribution.actual_output_stats
        dist_data["actual_total_stats"] = distribution.actual_total_stats
    
    dist_file = output_path / "sequence_distribution.json"
    with open(dist_file, 'w', encoding='utf-8') as f:
        json.dump(dist_data, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved distribution to: {dist_file}")
    
    # Save full profile
    profile_file = output_path / "sequence_profile.json"
    profiler.export_profile(str(profile_file))
    logger.info(f"Saved profile to: {profile_file}")
    
    # Save dataset stats
    stats = loader.get_statistics()
    stats_file = output_path / "dataset_stats.json"
    
    # Handle different stats types
    if hasattr(stats, 'avg_choices_per_question'):
        # GPQA stats
        stats_data = {
            "total_questions": stats.total_questions,
            "subjects": stats.subjects,
            "avg_question_length": stats.avg_question_length,
            "min_question_length": stats.min_question_length,
            "max_question_length": stats.max_question_length,
            "avg_choices_per_question": stats.avg_choices_per_question
        }
    else:
        # AIME25 stats
        stats_data = {
            "total_questions": stats.total_questions,
            "subjects": stats.subjects,
            "avg_question_length": stats.avg_question_length,
            "min_question_length": stats.min_question_length,
            "max_question_length": stats.max_question_length,
            "problem_types": getattr(stats, 'problem_types', {}),
            "difficulty_distribution": getattr(stats, 'difficulty_distribution', {}),
            "avg_answer_length": getattr(stats, 'avg_answer_length', 0)
        }
    
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats_data, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved dataset stats to: {stats_file}")
    
    # Export questions to JSONL for reference
    questions_file = output_path / "questions.jsonl"
    loader.export_to_jsonl(str(questions_file))
    logger.info(f"Exported questions to: {questions_file}")


def _print_routing_summary(profiler: SequenceProfiler, logger: logging.Logger):
    """Print routing recommendations summary based on actual categories."""
    
    logger.info("\n" + "=" * 60)
    logger.info("Routing Recommendations (based on actual total length)")
    logger.info("=" * 60)
    
    by_actual_category = profiler.get_sequences_by_actual_category()
    
    tp_recommendations = {
        "short": "TP=1 or TP=2 (efficient for small sequences)",
        "medium": "TP=2 or TP=4 (balanced performance)",
        "long": "TP=4 (high memory, good parallelism)",
        "extra_long": "TP=4 (maximum parallelism needed)"
    }
    
    for cat in ["short", "medium", "long", "extra_long"]:
        seqs = by_actual_category.get(cat, [])
        if seqs:
            total_tokens = [s.actual_total_tokens for s in seqs if s.actual_total_tokens]
            if total_tokens:
                logger.info(f"\n{cat.upper()} ({len(seqs)} sequences):")
                logger.info(f"  Total token range: {min(total_tokens)} - {max(total_tokens)}")
                logger.info(f"  Avg total tokens: {sum(total_tokens)/len(total_tokens):.0f}")
                logger.info(f"  Recommended: {tp_recommendations.get(cat, 'TP=2 or TP=4')}")


def _get_recommended_tp(category: str) -> str:
    """Get recommended TP degrees for a category."""
    recommendations = {
        "short": "TP=1 or TP=2",
        "medium": "TP=2 or TP=4",
        "long": "TP=4",
        "extra_long": "TP=4"
    }
    return recommendations.get(category, "TP=2 or TP=4")


def main():
    parser = argparse.ArgumentParser(
        description="Profile datasets for heterogeneous TP benchmarking"
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
        default="results/profiling",
        help="Output directory for profiling results"
    )
    parser.add_argument(
        "--dataset-type",
        type=str,
        default=None,
        choices=["gpqa", "aime25"],
        help="Dataset type to use (overrides config file)"
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
        "--with-inference",
        action="store_true",
        help="Run actual inference to get real output token counts"
    )
    parser.add_argument(
        "--tp",
        type=int,
        default=4,
        help="Tensor parallelism degree for inference (default: 4)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1000,
        help="Maximum tokens to generate per request (default: 1000)"
    )
    parser.add_argument(
        "--request-timeout",
        type=int,
        default=600,
        help="Individual request timeout in seconds (default: 600)"
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help="Comma-separated GPU IDs to use (e.g., '0,1,2,3')"
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=4,
        help="Maximum concurrent requests during inference (default: 4)"
    )
    
    args = parser.parse_args()
    
    # Setup HuggingFace environment first
    print("Setting up HuggingFace environment...")
    setup_hf_environment()
    
    # Setup logging
    setup_logging(args.log_level, args.log_file)
    
    # Load config
    print("Loading configuration...")
    config_path = PROJECT_ROOT / args.config
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)
    
    config = load_config(str(config_path))
    print("Configuration loaded successfully")
    # Override dataset type from command line if specified
    if args.dataset_type:
        config.setdefault('dataset', {})['type'] = args.dataset_type
        print(f"Using dataset type from command line: {args.dataset_type}")
    
    # Determine dataset type for logging
    dataset_config = config.get('dataset', {})
    dataset_type = dataset_config.get('type', 'gpqa')
    print(f"Dataset type: {dataset_type}")
    
    # Parse GPU IDs
    gpu_ids = None
    if args.gpus:
        gpu_ids = [int(g.strip()) for g in args.gpus.split(",")]
    
    # Run profiling
    output_dir = PROJECT_ROOT / args.output
    
    if args.with_inference:
        # Determine max_concurrent: command line > config file > default
        benchmark_config = config.get('benchmark', {})
        max_concurrent = args.max_concurrent
        if max_concurrent == 4:  # default value, check config file
            max_concurrent = benchmark_config.get('max_concurrent_requests', 4)
        print(f"Using max_concurrent: {max_concurrent}")
        
        # Determine request_timeout: command line > config file > default
        scheduling_config = config.get('scheduling', {})
        request_timeout = args.request_timeout
        if request_timeout == 600:  # default value, check config file
            request_timeout = scheduling_config.get('request_timeout', 600)
        print(f"Using request_timeout: {request_timeout}s")
        
        # Run with actual inference
        result = asyncio.run(run_profiling_with_inference(
            config=config,
            output_dir=str(output_dir),
            tp_degree=args.tp,
            gpu_ids=gpu_ids,
            max_concurrent=max_concurrent,
            request_timeout=request_timeout
        ))
        print(f"\nProfiling with inference complete!")
        print(f"Analyzed {result['total_questions']} {dataset_type} questions")
        print(f"Completed inference for {result['completed_sequences']} sequences")
    else:
        # Run estimation only
        result = run_profiling(config, str(output_dir))
        print(f"\nProfiling complete! Analyzed {result['total_questions']} {dataset_type} questions.")
    
    print(f"Results saved to: {result['output_dir']}")


if __name__ == "__main__":
    main()
