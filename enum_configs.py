#!/usr/bin/env python
"""
枚举配置测试：对指定或自动生成的配置列表依次运行 benchmark，汇总结果并输出报告。

支持两种模式：
1. --config_file: 从 JSON 文件加载配置列表
2. --auto_enum: 根据设计空间自动枚举所有有效配置（网格搜索）

用法示例：
  # 从 JSON 文件加载配置
  python enum_configs.py --config_file configs.json --model_path ... --dataset_path ... --model Qwen3_32B --total_resource 8A800

  # 自动枚举设计空间（可限制数量）
  python enum_configs.py --auto_enum --max_configs 50 --model_path ... --dataset_path ... --model Qwen3_32B --total_resource 8A800
"""

import re
import os

# NPU环境下的设备设置
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,5"  # 注释掉CUDA设置
# 如果需要指定NPU设备，可以取消下面的注释
# os.environ["ASCEND_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
import torch

import argparse

import json
import logging
import itertools
import math
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

import pandas as pd

# 复用 bo_scoot 的 benchmark 逻辑
from utils import check_port, get_ref_config
from bo_scoot import run_benchmark_pipeline

RES_DIR_PREFIX = 'scoot_enum'
RES_DIR = './tune_res'
LOG_DIR = os.path.join(RES_DIR, 'logs')
RAW_DIR = os.path.join(RES_DIR, 'raw')
ENUM_DIR = os.path.join(RES_DIR, 'enum_results')
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(ENUM_DIR, exist_ok=True)

def add_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--request_rate", type=int, default=20)
    parser.add_argument("--num_requests", type=int, default=1000)
    parser.add_argument("--model", type=str, required=True, help="模型名称")
    parser.add_argument("--total_resource", type=str, required=True, help="资源描述，如 8A800")
    parser.add_argument("--dataset_name", type=str, default="sharegpt")
    parser.add_argument("--sequence_profile_path", type=str, default=None)
    parser.add_argument("--pressure_test", action='store_true')
    parser.add_argument("--output_dir", type=str, default=None,
                        help="结果输出目录，默认 tune_res/enum_results/<timestamp>")
    parser.add_argument("--resume", action='store_true',
                        help="断点续测：若 output_dir 已有 enum_results.json/csv，跳过已完成的配置")
    # 枚举模式
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--config_file", type=str,
                       help="JSON 配置文件路径，格式: [{\"tp\":2,\"pipeline_parallel_size\":1,...}, ...]")
    group.add_argument("--auto_enum", action='store_true',
                       help="根据设计空间自动枚举所有有效配置")
    parser.add_argument("--max_configs", type=int, default=None,
                        help="auto_enum 模式下最大配置数量，超出则随机抽样")
    parser.add_argument("--tune_params", type=str, default="tp,pp,block_size",
                        help="auto_enum 模式下调优的参数类型，逗号分隔，如 tp,pp,block_size；其他参数使用默认值")
    return parser


def config_to_combination(cfg: Dict) -> Tuple:
    """将配置 dict 转为 run_benchmark_pipeline 所需的 combination 元组"""
    return (
        int(cfg['tp']),
        int(cfg.get('pipeline_parallel_size', 1)),
        int(cfg['max_num_seqs']),
        int(cfg['max_num_batched_tokens']),
        int(cfg['block_size']),
        float(cfg.get('scheduler_delay_factor', 0.0)),
        "True",   # enable_chunked_prefill (固定)
        "True",   # enable_prefix_caching (固定)
        "True",   # disable_custom_all_reduce (固定)
        "True",   # use_v2_block_manager (固定)
        str(cfg.get('enable_expert_parallel', False)),
    )


def _save_results(results: List[Dict], output_dir: str) -> None:
    """即时保存结果，支持断点续测"""
    json_path = os.path.join(output_dir, 'enum_results.json')
    csv_path = os.path.join(output_dir, 'enum_results.csv')
    try:
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        if results:
            pd.DataFrame(results).to_csv(csv_path, index=False)
    except Exception as e:
        logging.warning(f"保存结果失败: {e}")


def config_fingerprint(cfg: Dict, exclude_expert_parallel: bool = True) -> tuple:
    """生成配置的唯一指纹，用于判断是否已完成。
    默认排除 enable_expert_parallel，因 vllm 结果可能与其实际传入值不一致，导致断点续测无法匹配。
    """
    skip_keys = {'_failed'}
    if exclude_expert_parallel:
        skip_keys = skip_keys | {'enable_expert_parallel'}
    return tuple(sorted((k, str(v)) for k, v in cfg.items() if k not in skip_keys))
    
def config_to_rec(cfg: Dict) -> pd.DataFrame:
    """将配置 dict 转为 obj() 所需的 DataFrame"""
    return pd.DataFrame([{
        'tp': cfg['tp'],
        'pipeline_parallel_size': cfg.get('pipeline_parallel_size', 1),
        'max_num_seqs': cfg['max_num_seqs'],
        'max_num_batched_tokens': cfg['max_num_batched_tokens'],
        'block_size': cfg['block_size'],
        'scheduler_delay_factor': cfg.get('scheduler_delay_factor', 0.0),
        'enable_expert_parallel': cfg.get('enable_expert_parallel', False),
    }])


def load_configs_from_file(path: str) -> List[Dict]:
    with open(path, 'r') as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and 'configs' in data:
        return data['configs']
    raise ValueError(f"config_file 需为配置列表或 {{\"configs\": [...]}} 格式")


def generate_enum_configs(gpu_nums: int, max_sequence_length: int, min_world_size: int,
                         max_configs: Optional[int] = None,
                         tune_params: Optional[List[str]] = None) -> List[Dict]:
    """根据设计空间自动生成枚举配置。

    tune_params: 要调优的参数列表，如 ['tp', 'pp', 'block_size']；其余参数使用默认值。
    """
    import numpy as np
    from hebo.optimizers.util import ensure_hard_constr

    tune_set = set((tune_params or ['tp', 'pp', 'block_size']))
    # 默认值（非调优参数使用）
    mns_default = 64
    mnbt_default = 32768
    sdf_default = 0.0
    ep_default = False

    # 与 bo_scoot 一致的参数空间取值
    tp_lb = max(min_world_size, 2)
    tp_vals = [2 ** i for i in range(int(math.log2(tp_lb)), int(math.log2(gpu_nums)) + 1)
               if 2 ** i <= gpu_nums and 2 ** i >= tp_lb]
    pp_vals = [2 ** i for i in range(0, int(math.log2(gpu_nums)) + 1) if 2 ** i <= gpu_nums]
    mns_vals = [2 ** i for i in range(6, 14) if 2 ** i <= 8192]  # 64..8192
    mnbt_ub = max(32768, max_sequence_length * 2)
    mnbt_vals = [2 ** i for i in range(6, int(math.log2(mnbt_ub)) + 1) if 2 ** i <= mnbt_ub]
    bs_vals = [16, 32, 64]  # block_size 下限为 16
    sdf_vals = list(range(0, 21, 2))  # 0,2,4,...,20 -> 0.0, 0.2, ..., 2.0
    expert_vals = [False, True]

    # 只对 tune_params 中指定的参数进行枚举
    tp_iter = tp_vals if 'tp' in tune_set else [tp_vals[0] if tp_vals else 2]
    pp_iter = pp_vals if 'pp' in tune_set or 'pipeline_parallel_size' in tune_set else [1]
    mns_iter = mns_vals if 'max_num_seqs' in tune_set else [mns_default]
    mnbt_iter = mnbt_vals if 'max_num_batched_tokens' in tune_set else [mnbt_default]
    bs_iter = bs_vals if 'block_size' in tune_set else [bs_vals[0]]
    sdf_iter = sdf_vals if 'scheduler_delay_factor' in tune_set else [int(sdf_default * 10)]
    expert_iter = expert_vals if 'enable_expert_parallel' in tune_set else [ep_default]

    all_combos = list(itertools.product(
        tp_iter, pp_iter, mns_iter, mnbt_iter, bs_iter, sdf_iter, expert_iter
    ))

    configs = []
    for t in all_combos:
        tp, pp, mns, mnbt, bs, sdf, ep = t
        if tp * pp > gpu_nums:  # tp_size x pp_size 必须小于 gpu_nums
            continue
        if mns > mnbt:
            continue
        configs.append({
            'tp': tp,
            'pipeline_parallel_size': pp,
            'max_num_seqs': mns,
            'max_num_batched_tokens': mnbt,
            'block_size': bs,
            'scheduler_delay_factor': sdf / 10.0 if isinstance(sdf, int) else sdf,
            'enable_expert_parallel': ep,
        })

    # 通过 ensure_hard_constr 再次过滤
    if configs:
        df = pd.DataFrame(configs)
        df = ensure_hard_constr(df.copy(), max_sequence_length)
        configs = df.to_dict('records')
        # 去重
        seen = set()
        unique = []
        for c in configs:
            key = tuple(sorted((k, v) for k, v in c.items()))
            if key not in seen:
                seen.add(key)
                unique.append(c)
        configs = unique

    if max_configs and len(configs) > max_configs:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(configs), max_configs, replace=False)
        configs = [configs[i] for i in sorted(idx)]

    return configs


def run_single_config(cfg: Dict, gpu_nums: int, res_dir_path: str, args,
                     min_world_size: int = 1) -> Optional[Dict]:
    """
    运行单配置 benchmark，返回结果 dict（含 throughput、ttft、tpot 等）或 None
    """
    try:
        os.system(f'pgrep -f "clients.api_server" | xargs kill -9 2>/dev/null')
    except Exception:
        pass
    ports = ",".join([str(8000 + i) for i in range(int(gpu_nums / min_world_size))])
    for port in ports.split(','):
        for _ in range(3):
            if check_port(int(port)):
                try:
                    os.system(f"lsof -t -i:{int(port)} | xargs kill -9 2>/dev/null")
                except Exception:
                    pass
            else:
                break
    import time
    time.sleep(5)

    combination = config_to_combination(cfg)
    run_benchmark_pipeline(combination, device_nums, args)

    vllm_files = []
    for root, _, files in os.walk(res_dir_path):
        for name in files:
            if name.startswith("vllm"):
                vllm_files.append(os.path.join(root, name))
    if not vllm_files:
        return None
    file_path = max(vllm_files, key=os.path.getmtime)
    with open(file_path, 'r') as f:
        act = json.load(f)

    act_pp = act.get('pp', act.get('pipeline_parallel_size', 1))
    act_ep = act.get('enable_expert_parallel', 'False')
    if combination != (int(act['tp']), int(act_pp), int(act['max_num_seqs']),
                      int(act['max_num_batched_tokens']), int(act['block_size']),
                      float(act['scheduler_delay_factor']),
                      str(act['enable_chunked_prefill']), str(act['enable_prefix_caching']),
                      str(act['disable_custom_all_reduce']), str(act['use_v2_block_manager']),
                      str(act_ep)):
        return None

    return {
        **cfg,
        'request_throughput': act.get('request_throughput'),
        'mean_ttft_ms': act.get('mean_ttft_ms'),
        'mean_tpot_ms': act.get('mean_tpot_ms'),
        'total_output_tokens': act.get('total_output_tokens'),
        'total_input_tokens': act.get('total_input_tokens'),
    }


def parse_device_num_from_total_resource(total_resource: str) -> Optional[int]:
    """从 total_resource 解析设备数量，如 4A800、8A800_mobo、4NPU910B -> 4 或 8"""
    m = re.match(r'^(\d+)', total_resource)
    if m:
        return int(m.group(1))
    return None


def main(args):
    print(f"枚举配置测试参数: {args}")
    # 优先使用 --total_resource 中解析的设备数量，否则使用可见设备数
    device_nums_from_arg = parse_device_num_from_total_resource(args.total_resource)
    
    # 检查可用设备类型（优先NPU，其次CUDA）
    visible_npu_count = 0
    visible_cuda_count = 0
    
    # 检查NPU设备
    try:
        # 尝试导入NPU相关库
        import torch_npu  # noqa: F401
        visible_npu_count = torch.npu.device_count()
        print(f"检测到 {visible_npu_count} 个NPU设备")
    except ImportError:
        print("未检测到NPU支持")
    
    # 检查CUDA设备
    try:
        visible_cuda_count = torch.cuda.device_count()
        if visible_cuda_count > 0:
            print(f"检测到 {visible_cuda_count} 个CUDA设备")
    except Exception:
        print("CUDA设备检测失败")
    
    # 确定使用的设备类型和数量
    if visible_npu_count > 0:
        # 优先使用NPU
        device_type = "NPU"
        visible_count = visible_npu_count
    elif visible_cuda_count > 0:
        # 回退到CUDA
        device_type = "CUDA" 
        visible_count = visible_cuda_count
    else:
        # 没有可用设备
        raise RuntimeError("未检测到任何可用的计算设备（NPU或CUDA）")
    
    if device_nums_from_arg is not None:
        device_nums = min(device_nums_from_arg, visible_count)
        if device_nums < visible_count:
            print(f"根据 --total_resource {args.total_resource} 限制使用 {device_nums} 个 {device_type} 设备（可见 {visible_count} 个）")
    else:
        device_nums = visible_count
        print(f"使用全部 {device_nums} 个 {device_type} 设备")
    
    assert device_nums % 2 == 0 or device_nums == 1

    # 确保在 SCOOT 根目录执行（以便 import bo_scoot、utils）
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir and os.getcwd() != script_dir:
        os.chdir(script_dir)

    try:
        min_world_size = get_ref_config('min_world_size')
        max_sequence_length = get_ref_config('max_sequence_length')
    except Exception:
        min_world_size = 2
        max_sequence_length = 4096

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        output_dir = args.output_dir
    elif args.resume:
        # 断点续测且未指定 output_dir：使用最新的同 model_dataset 目录
        prefix = f"{args.model}_{args.dataset_name}_"
        cands = [d for d in os.listdir(ENUM_DIR) if d.startswith(prefix)] if os.path.exists(ENUM_DIR) else []
        cands = sorted(cands, reverse=True)
        if cands:
            output_dir = os.path.join(ENUM_DIR, cands[0])
            print(f"[断点续测] 使用最新目录: {output_dir}")
        else:
            output_dir = os.path.join(ENUM_DIR, f"{args.model}_{args.dataset_name}_{timestamp}")
    else:
        output_dir = os.path.join(ENUM_DIR, f"{args.model}_{args.dataset_name}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    res_dir_path = os.path.join(output_dir, 'benchmark_runs')
    os.makedirs(res_dir_path, exist_ok=True)
    os.environ["RES_DIR_PATH"] = res_dir_path

    # 断点续测：加载已有结果（优先 enum_results，否则从 benchmark_runs 恢复）
    existing_results: List[Dict] = []
    completed_fingerprints: set = set()
    config_keys = {'tp', 'pipeline_parallel_size', 'max_num_seqs', 'max_num_batched_tokens',
                   'block_size', 'scheduler_delay_factor', 'enable_expert_parallel'}
    if args.resume:
        loaded = False
        for fname in ('enum_results.json', 'enum_results.csv'):
            p = os.path.join(output_dir, fname)
            if os.path.exists(p):
                try:
                    if fname.endswith('.json'):
                        with open(p, 'r') as f:
                            existing_results = json.load(f)
                    else:
                        existing_results = pd.read_csv(p).to_dict('records')
                    for r in existing_results:
                        cfg_part = {k: v for k, v in r.items() if k in config_keys}
                        if cfg_part:
                            completed_fingerprints.add(config_fingerprint(cfg_part))
                    logging.info(f"Resume: 从 {p} 加载 {len(existing_results)} 条已完成结果")
                    print(f"[断点续测] 已加载 {len(existing_results)} 条结果，将跳过 {len(completed_fingerprints)} 个已完成配置")
                    loaded = True
                except Exception as e:
                    logging.warning(f"Resume 加载 {p} 失败: {e}")
                break
        if not loaded:
            # 从 benchmark_runs 恢复：扫描 vllm-*.json  reconstruct 已完成配置
            bench_dir = os.path.join(output_dir, 'benchmark_runs')
            if os.path.isdir(bench_dir):
                vllm_files = [f for f in os.listdir(bench_dir) if f.startswith('vllm') and f.endswith('.json')]
                vllm_files = sorted(vllm_files, key=lambda x: os.path.getmtime(os.path.join(bench_dir, x)))
                for vf in vllm_files:
                    try:
                        with open(os.path.join(bench_dir, vf), 'r') as f:
                            act = json.load(f)
                        cfg = {
                            'tp': int(act.get('tp', 2)),
                            'pipeline_parallel_size': int(act.get('pp', act.get('pipeline_parallel_size', 1))),
                            'max_num_seqs': int(act.get('max_num_seqs', 64)),
                            'max_num_batched_tokens': int(act.get('max_num_batched_tokens', 4096)),
                            'block_size': int(act.get('block_size', 16)),
                            'scheduler_delay_factor': float(act.get('scheduler_delay_factor', 0.0)),
                            'enable_expert_parallel': act.get('enable_expert_parallel', 'False') in ('True', True),
                        }
                        existing_results.append({
                            **cfg,
                            'request_throughput': act.get('request_throughput'),
                            'mean_ttft_ms': act.get('mean_ttft_ms'),
                            'mean_tpot_ms': act.get('mean_tpot_ms'),
                            'total_output_tokens': act.get('total_output_tokens'),
                            'total_input_tokens': act.get('total_input_tokens'),
                        })
                        completed_fingerprints.add(config_fingerprint(cfg))
                    except Exception as e:
                        logging.warning(f"解析 {vf} 失败: {e}")
                if existing_results:
                    logging.info(f"Resume: 从 benchmark_runs 恢复 {len(existing_results)} 条结果")
                    print(f"[断点续测] 从 benchmark_runs 恢复 {len(existing_results)} 条已完成结果，将跳过对应配置")

    log_file = os.path.join(LOG_DIR, f"enum_{args.model}_{args.dataset_name}_{timestamp}.log")
    logging.basicConfig(filename=log_file, level=logging.INFO)
    logging.info(f"Enum configs started, output_dir={output_dir}")

    if args.config_file:
        configs = load_configs_from_file(args.config_file)
        print(f"从 {args.config_file} 加载 {len(configs)} 个配置")
    else:
        tune_params = [p.strip() for p in args.tune_params.split(',') if p.strip()]
        configs = generate_enum_configs(
            device_nums, max_sequence_length, min_world_size, args.max_configs, tune_params=tune_params
        )
        print(f"自动枚举得到 {len(configs)} 个有效配置 (调优参数: {tune_params})")

    if not configs:
        print("无有效配置，退出")
        return

    results = []
    for i, cfg in enumerate(configs):
        fp = config_fingerprint(cfg)
        if args.resume and fp in completed_fingerprints:
            # 断点续测：跳过已完成配置，从已有结果中取出
            for ex in existing_results:
                ex_cfg = {k: ex[k] for k in config_keys if k in ex}
                if config_fingerprint(ex_cfg) == fp:
                    results.append(ex)
                    print(f"\n[{i+1}/{len(configs)}] [跳过-已完成] {cfg}")
                    print(f"   -> throughput={ex.get('request_throughput')}, ttft={ex.get('mean_ttft_ms')}ms")
                    break
            else:
                results.append({**cfg, 'request_throughput': None, 'mean_ttft_ms': None, 'mean_tpot_ms': None, '_failed': True})
            continue

        print(f"\n[{i+1}/{len(configs)}] 测试配置: {cfg}")
        logging.info(f"Running config {i+1}/{len(configs)}: {cfg}")
        r = run_single_config(cfg, gpu_nums, res_dir_path, args, min_world_size)
        if r is not None:
            results.append(r)
            if args.resume:
                completed_fingerprints.add(fp)
            print(f"   -> throughput={r.get('request_throughput')}, ttft={r.get('mean_ttft_ms')}ms, tpot={r.get('mean_tpot_ms')}ms")
        else:
            print("   -> 失败或结果不匹配")
            results.append({**cfg, 'request_throughput': None, 'mean_ttft_ms': None, 'mean_tpot_ms': None, '_failed': True})

        # 即时保存，支持断点续测
        _save_results(results, output_dir)

    # 输出汇总（_save_results 已在循环中即时保存，此处确保最终一致）
    _save_results(results, output_dir)
    csv_path = os.path.join(output_dir, 'enum_results.csv')
    json_path = os.path.join(output_dir, 'enum_results.json')
    print(f"\n结果已保存至 {csv_path} 与 {json_path}")

    # 按吞吐量排序打印
    df = pd.DataFrame(results)
    valid = df[df['request_throughput'].notna()]
    if len(valid) > 0:
        valid = valid.sort_values('request_throughput', ascending=False)
        print("\n按 throughput 排序 Top 配置:")
        print(valid[['tp', 'pipeline_parallel_size', 'max_num_seqs', 'max_num_batched_tokens',
                     'block_size', 'scheduler_delay_factor', 'enable_expert_parallel',
                     'request_throughput', 'mean_ttft_ms', 'mean_tpot_ms']].head(10).to_string())
    else:
        print("无成功结果")

    logging.info(f"Enum completed, {len(valid)}/{len(configs)} succeeded")
    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="枚举配置 benchmark 测试")
    parser = add_args(parser)
    args = parser.parse_args()
    main(args)
