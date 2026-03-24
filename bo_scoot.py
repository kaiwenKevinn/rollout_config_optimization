import os
# NPU环境下的设备设置
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,5"  # 注释掉CUDA设置
# 如果需要指定NPU设备，可以取消下面的注释
# os.environ["ASCEND_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

import torch
import argparse
import random
import logging
import tqdm
import traceback
import json
import time
import math

from typing import Union, Tuple
from utils import gen_res_dir_path, check_port, get_ref_config, read_historical_data, find_available_base_port
import pandas as pd
import numpy as np
from hebo.design_space.design_space import DesignSpace
from hebo.optimizers.general import GeneralBO
from hebo.optimizers.hebo_constr import HEBOConstr
from transformers import AutoConfig
from sklearn.ensemble import RandomForestRegressor

pd.set_option('display.max_rows', None)

pd.set_option('display.max_columns', None)

pd.set_option('expand_frame_repr', False)

RES_DIR_PREFIX = 'scoot'
RES_DIR = './tune_res'
LOG_DIR = os.path.join(RES_DIR,'logs')
RAW_DIR = os.path.join(RES_DIR,'raw')
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(RAW_DIR, exist_ok=True)

def add_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--request_rate", type=int, default=20)
    parser.add_argument("--num_requests", type=int, default=1000)
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Name of the model.",
    )
    parser.add_argument(
        "--total_resource",
        type=str,
        required=True,
        help="Total Resources",
    )
    parser.add_argument("--bo_loop", type=int, default=50)
    parser.add_argument("--bo_batch_size", type=int, default=1)
    parser.add_argument(
        "--exp_num",
        type=int,
        default=1,
        help="Total experiment run time",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="sharegpt"
    )
    parser.add_argument(
        "--sequence_profile_path",
        type=str,
        default=None,
        help="Path to sequence_profile.json with actual_output_tokens for realistic load."
    )

    parser.add_argument(
        "--num_obj",
        type=int,
        default=2)

    parser.add_argument(
        "--pressure_test",
        action='store_true')

    parser.add_argument(
        "--tune_params",
        type=str,
        default="tp,pp,block_size",
        help="调优的参数类型，逗号分隔，如 tp,pp,block_size；其他参数使用默认值")
    return parser


def run_benchmark_pipeline(combination: Tuple, gpu_nums, args, base_port: int = None):
    tp_size = combination[0]
    pp_size = combination[1]
    world_size = tp_size * pp_size
    num_replicas = int(gpu_nums / world_size)
    # 优先使用传入的 base_port；否则自动寻找空闲端口
    if base_port is None:
        base_port = find_available_base_port(num_replicas)
    ports = ",".join([str(base_port + i) for i in range(num_replicas)])
    gpus = [str(i) for i in range(gpu_nums)]
    grouped_gpus = [','.join(gpus[i:i + world_size]) for i in range(0, gpu_nums, world_size)]
    grouped_gpus_string = '#'.join(grouped_gpus)
    # combination: [0]=tp, [1]=pp, [2]=mns, [3]=mnbt, [4]=bs, [5]=sdf, [6-9]=fixed, [10]=enable_expert_parallel
    raw_file_path = os.path.join(RAW_DIR, f'benchmark_tp_{combination[0]}_pp_{combination[1]}_mns_{combination[2]}_mnbt_{combination[3]}_bs_{combination[4]}.txt')
    seq_profile = args.sequence_profile_path or ""
    for i in range(3):
        # retry 3 times in case of failure
        try:
            logging.info(
                f"bash benchmark_pipeline.sh {args.model_path} {args.dataset_path} {args.request_rate} {args.num_requests} {args.pressure_test} {0} {combination[0]} {combination[1]} {combination[2]} {combination[3]} {combination[5]} {combination[4]} {ports} {grouped_gpus_string} {args.model} {combination[6]} {args.dataset_name} {combination[7]} {combination[8]} {combination[9]} {seq_profile} {combination[10]} "
                f"2>&1 | tee {raw_file_path}")
            seq_profile = args.sequence_profile_path or ""
            os.system(
                f"bash benchmark_pipeline.sh {args.model_path} {args.dataset_path} {args.request_rate} {args.num_requests} {args.pressure_test} {0} {combination[0]} {combination[1]} {combination[2]} {combination[3]} {combination[5]} {combination[4]} {ports} {grouped_gpus_string} {args.model} {combination[6]} {args.dataset_name} {combination[7]} {combination[8]} {combination[9]} {seq_profile} {combination[10]} "
                f"2>&1 | tee {raw_file_path}")
            break
        except Exception:
            logging.error(f'init error: {traceback.format_exc()}')


def _rec_to_combination(rec: pd.DataFrame, max_sequence_length: int = 4096) -> Tuple:
    """将 rec（可能只含部分调优参数）转为完整 combination，未调优参数用默认值"""
    def get(col, default):
        if col in rec.columns and len(rec[col]) > 0:
            return rec[col].tolist()[0]
        return default
    mnbt_default = max(32768, max_sequence_length * 2)
    sdf_raw = get('scheduler_delay_factor', 0)  # step_int: 0,2,...,20 -> 除以10得 0.0,0.2,...,2.0
    sdf_val = float(sdf_raw) / 10.0
    return (
        int(get('tp', 2)),
        int(get('pipeline_parallel_size', 1)),
        int(get('max_num_seqs', 64)),
        int(get('max_num_batched_tokens', mnbt_default)),
        int(get('block_size', 16)),
        sdf_val,
        "True",   # enable_chunked_prefill (固定)
        "True",   # enable_prefix_caching (固定)
        "True",   # disable_custom_all_reduce (固定)
        "True",   # use_v2_block_manager (固定)
        str(get('enable_expert_parallel', False)),
    )


def obj(rec: pd.DataFrame, gpu_nums, res_dir_path, args, min_world_size: int = 1) -> Union[None, np.ndarray]:
    # 固定为 True：enable_chunked_prefill, enable_prefix_caching, disable_custom_all_reduce, use_v2_block_manager
    try:
        max_seq_len = get_ref_config('max_sequence_length')
    except Exception:
        max_seq_len = 4096
    combination = _rec_to_combination(rec, max_seq_len)

    # 清理残留进程
    try:
        os.system(f'pgrep -f "clients.api_server" | xargs kill -9 2>/dev/null || true')
    except Exception as e:
        print("Kill Whole Process Error:", e)
    time.sleep(5)

    # 寻找空闲端口（若 8000 被占用则自动选用其他端口）
    num_replicas = int(gpu_nums / min_world_size)
    try:
        base_port = find_available_base_port(num_replicas, start=8000, end=30000)
    except RuntimeError:
        os.system(f'pgrep -f "clients.api_server" | xargs kill -9 2>/dev/null || true')
        time.sleep(10)
        base_port = find_available_base_port(num_replicas, start=8000, end=30000)
    print(f'Using base_port={base_port} (ports {base_port}-{base_port + num_replicas - 1})')
    logging.info(f'Using base_port={base_port} for benchmark')
    time.sleep(2)
    run_benchmark_pipeline(combination, gpu_nums, args, base_port=base_port)

    # 取最新的 vllm 结果文件（benchmark 失败时可能无新文件或取到旧配置）
    vllm_files = []
    for root, _, files in os.walk(res_dir_path):
        for name in files:
            if name.startswith("vllm"):
                vllm_files.append(os.path.join(root, name))
    if not vllm_files:
        logging.warning(f"No vllm result file in {res_dir_path}, benchmark may have failed")
        return None
    file_path = max(vllm_files, key=os.path.getmtime)
    with open(file_path, 'r') as f:
        act_result = json.load(f)

    # check wether the result from the latest file ("act_result") is indeed the output of actual env from the "rec"
    # 固定配置均为 True；pp 对应 pipeline_parallel_size
    act_pp = act_result.get('pp', act_result.get('pipeline_parallel_size', 1))
    act_enable_expert = act_result.get('enable_expert_parallel', 'False')
    if combination == (int(act_result['tp']),
                       int(act_pp),
                       int(act_result['max_num_seqs']),
                       int(act_result['max_num_batched_tokens']),
                       int(act_result['block_size']),
                       float(act_result['scheduler_delay_factor']),
                       str(act_result['enable_chunked_prefill']),
                       str(act_result['enable_prefix_caching']),
                       str(act_result['disable_custom_all_reduce']),
                       str(act_result['use_v2_block_manager']),
                       str(act_enable_expert)):
        # return np.array(
        #     [[
        #         -1 * act_result["request_throughput"],
        #         act_result["mean_ttft_ms"],
        #         act_result["mean_tpot_ms"]
        #     ]]
        # )

        # 将多目标合并为单目标：加权求和
        # 权重分配：吞吐量(0.5), TTFT(0.3), TPOT(0.2)
        throughput_weight = 1
        ttft_weight = 0
        tpot_weight = 0
        
        # 标准化各目标值到相近范围
        normalized_throughput = -1 * act_result["request_throughput"] / 100.0  # 假设吞吐量通常在几十到几百
        normalized_ttft = act_result["mean_ttft_ms"] / 1000.0  # 转换为秒
        normalized_tpot = act_result["mean_tpot_ms"] / 100.0   # 假设通常在几十毫秒
        
        # 计算综合目标值
        combined_objective = (throughput_weight * normalized_throughput + 
                            ttft_weight * normalized_ttft + 
                            tpot_weight * normalized_tpot)
        
        return np.array([[combined_objective]])
    else:
        return None


def read_rec_history(rec_history_file_path):
    if os.path.exists(rec_history_file_path):
        with open(rec_history_file_path, 'r') as fp:
            rec_history_data = json.load(fp)
            failed_num = len([item for item in rec_history_data if item['obj'] is None])
            return failed_num, rec_history_data
    else:
        return 0, []


def obtain_random_forest_train_set(rec_history, param_names=None):
    """param_names: 调优参数列名，用于构建特征；不传则使用全部参数（兼容旧格式）"""
    default_names = ['tp', 'pipeline_parallel_size', 'max_num_seqs', 'max_num_batched_tokens',
                    'block_size', 'scheduler_delay_factor', 'enable_expert_parallel']
    default_vals = [2, 1, 64, 32768, 16, 0, 0]
    name_to_idx = {k: i for i, k in enumerate(default_names)}
    x = []
    y = []
    for data_item in rec_history:
        rec0 = data_item['rec'][0]
        names = param_names if param_names else default_names
        row = []
        for n in names:
            if n == 'enable_expert_parallel':
                row.append(1 if rec0.get(n, False) else 0)
            else:
                row.append(rec0.get(n, default_vals[name_to_idx.get(n, 0)]))
        x.append(row)
        y.append(0 if data_item['obj'] is None else 1)
    return {'train_x': x, 'train_y': y}


def random_forest_regressor(train_set):
    x = train_set['train_x']
    y = train_set['train_y']
    if x and y and len(x) >= (len(x[0]) - 1):
        rfr = RandomForestRegressor()
        rfr.fit(x, y)
        return rfr
    else:
        return None


def compute_delta_and_continuous_right(rec_history):
    delta = 0.5
    continuous_right = 0
    if rec_history:
        init_train_size = len(rec_history[0]['rec'][0])
        if len(rec_history) >= init_train_size:  
            for i in range(init_train_size, len(rec_history)):
                if rec_history[i]['obj'] is None:
                    delta = round(min(0.75, max(delta + 0.05, 0.5)), 3)
                    continuous_right = 0
                else:
                    continuous_right += 1
                    if continuous_right > 5:
                        continuous_right = continuous_right - 5
                        delta = round(max(0.25, delta - 0.05), 3)
    return delta, continuous_right


def _ensure_ports_cleared(gpu_nums: int, min_world_size: int, max_retries: int = 8) -> None:
    """启动前确保端口已释放，避免上一轮残留进程导致 RuntimeError"""
    num_ports = max(2, int(gpu_nums / min_world_size))
    ports = [8000 + i for i in range(num_ports)]
    for attempt in range(max_retries):
        os.system('pgrep -f "clients.api_server" | xargs kill -9 2>/dev/null || true')
        time.sleep(5)
        busy = []
        for p in ports:
            if check_port(p):
                busy.append(p)
                os.system(f'lsof -t -i:{p} | xargs kill -9 2>/dev/null || true')
        if not busy:
            return
        print(f'Ports {busy} still busy (attempt {attempt + 1}/{max_retries}), waiting...')
        logging.info(f'Ports {busy} still busy, retrying...')
        time.sleep(10)
    logging.warning('Some ports may still be in use; continuing anyway.')


def main(args):
    print(f'Input Tuning Arguments: {args}')
    gpu_nums = torch.cuda.device_count()
    assert gpu_nums % 2 == 0 or gpu_nums == 1 
    logging.basicConfig(
        filename=os.path.join(LOG_DIR, f'bo_{args.bo_batch_size}_{args.model}_{args.total_resource}_num_requests{args.num_requests}_request_rate{args.request_rate}_{args.dataset_name}.log'),
        level=logging.INFO)

    for e in range(args.exp_num):
        res_dir_path = gen_res_dir_path(args.model, args.request_rate, args.num_requests, args.total_resource,
                                        args.dataset_name, RES_DIR, exp=e, bo=True, dir_prefix=RES_DIR_PREFIX)
        print(res_dir_path)
        os.environ["RES_DIR_PATH"] = res_dir_path

        logging.info(f"New BO loop of experiment {e} begins!!!")

        min_world_size = get_ref_config('min_world_size')
        _ensure_ports_cleared(gpu_nums, min_world_size)

        # 1. observe historical data
        xx, yy = read_historical_data(res_dir_path)

        # 2. DesignSpace
        max_sequence_length = get_ref_config('max_sequence_length')

        logging.info(f"Min World Size: {min_world_size}, "
                     f"Max World Size: {gpu_nums}"
                    )
        # 固定为 True 的配置：enable_chunked_prefill, enable_prefix_caching, disable_custom_all_reduce, use_v2_block_manager
        # 根据 tune_params 决定哪些参数参与调优
        tune_set = set(p.strip() for p in (args.tune_params or "tp,pp,block_size").split(",") if p.strip())
        param_defs = {
            "tp": {"name": "tp", "type": "int_exponent", "lb": max(min_world_size, 2), "ub": gpu_nums, "base": 2},
            "pipeline_parallel_size": {"name": "pipeline_parallel_size", "type": "int_exponent", "lb": 1, "ub": gpu_nums, "base": 2},
            "max_num_seqs": {"name": "max_num_seqs", "type": "int_exponent", "lb": 64, "ub": 8192, "base": 2},
            "max_num_batched_tokens": {"name": "max_num_batched_tokens", "type": "pow_int", "lb": 64, "ub": max(32768, max_sequence_length * 2), "base": 2},
            "block_size": {"name": "block_size", "type": "int_exponent", "lb": 16, "ub": 64, "base": 2},
            "scheduler_delay_factor": {"name": "scheduler_delay_factor", "type": "step_int", "lb": 0, "ub": 20, "step": 2},
            "enable_expert_parallel": {"name": "enable_expert_parallel", "type": "bool"},
        }
        # pp 与 pipeline_parallel_size 视为同一参数
        if "pp" in tune_set:
            tune_set.add("pipeline_parallel_size")
        para_dict = []
        for k in ["tp", "pipeline_parallel_size", "max_num_seqs", "max_num_batched_tokens", "block_size", "scheduler_delay_factor", "enable_expert_parallel"]:
            if k in tune_set:
                para_dict.append(param_defs[k])
        logging.info(f"Tune params: {tune_set}, design space: {[p['name'] for p in para_dict]}")

        space = DesignSpace().parse(para_dict)

        # 3. BO optimizer
        # 3.1 sample ref point
        if len(xx) > 0 and len(yy) > 0:
            # The first run in the history is ref point
            ref_point = np.array(yy[0])
            logging.info(f'input for ref point is {xx[0]} ')
        else: 
            ref_rec = pd.DataFrame.from_dict(
                {
                "tp": [min_world_size],
                "pipeline_parallel_size": [1],
                "max_num_seqs": [256],
                "max_num_batched_tokens": [max(4096, max_sequence_length)],
                "block_size": [16],
                "scheduler_delay_factor": [0.0],
                "enable_expert_parallel": [False],
                # enable_chunked_prefill, enable_prefix_caching, disable_custom_all_reduce, use_v2_block_manager 固定为 True
                 })
            ref_point = obj(ref_rec, gpu_nums, res_dir_path, args, min_world_size=min_world_size)
            logging.info(f'ref config: {ref_rec}')
            logging.info(f'ref obj: {ref_point}')
            xx, yy = read_historical_data(res_dir_path)

        ref_max_seq_len = math.ceil(math.log2(max_sequence_length))
        if args.num_obj > 1:
            # multi objective optimization
            opt = GeneralBO(space=space,
                            num_obj=args.num_obj,
                            num_model_constr=0,
                            num_hard_constr=2,
                            num_hidden_constr=1,
                            ref_point=ref_point,
                            model_config={"optimizer": "adam", "base_model_name": "gpy", "space": space},
                            use_noise=True,
                            max_sequence_length=ref_max_seq_len
                            )
        elif args.num_obj == 1:

            opt = HEBOConstr( 
                space=space,
                num_model_constr=0,
                num_hard_constr=2,
                num_hidden_constr=1,
                max_sequence_length=ref_max_seq_len,
            )
        else:
            raise ValueError('Work with Positive Numbers Only')

        pbar = tqdm.tqdm(total=args.bo_loop)

        # 3.3 merge history data
        if len(xx) > 0 and len(yy) > 0:
            print(f'Tuning History Configurations: {xx}')
            print(f'Tuning History Objectives: {yy}')
            explored_x = pd.DataFrame(xx)
            space_cols = [p['name'] for p in para_dict]
            explored_x = explored_x[[c for c in space_cols if c in explored_x.columns]]
            explored_y = np.array(yy)
            opt.observe(explored_x, explored_y)

        # 4. bo loop execution
        succeed_num = len(xx)
        rec_history_file_path = os.path.join(res_dir_path, 'rec_history.json')
        failed_num, rec_history_list = read_rec_history(rec_history_file_path)

        space_param_names = [p['name'] for p in para_dict]
        train_set = obtain_random_forest_train_set(rec_history_list, param_names=space_param_names)
        random_forest = random_forest_regressor(train_set)
        delta, continuous_right = compute_delta_and_continuous_right(rec_history_list)
        i = failed_num + succeed_num  
        pbar.update(i)
        while i < args.bo_loop: 
            s_time = time.time()
            rec = opt.suggest(n_suggestions=1, rf_with_thres=(random_forest, delta))
            e_time = time.time()
            rec_time = e_time - s_time

            s_time = time.time()
            try:
                y = obj(rec, gpu_nums, res_dir_path, args, min_world_size=min_world_size)
            except Exception as e:
                logging.error(f'Config evaluation failed, continuing to next: {traceback.format_exc()}')
                y = None
            e_time = time.time()
            run_time = e_time - s_time

            y_none = False
            if y is None:
                y_none = True
                failed_num += 1
                if i > len(para_dict): 
                    delta = round(min(0.75, max(delta + 0.05, 0.5)), 3)  # round operation is to avoid the error caused by float calculation
                    continuous_right = 0
            else:
                opt.observe(rec, y)  # update surrogate model
                succeed_num += 1
                if i > len(para_dict):  # i = len(rec_history)+1
                    continuous_right += 1
                    if continuous_right >= 5:
                        continuous_right = continuous_right - 5
                        delta = round(max(0.25, delta - 0.05), 3) 

            pbar.update(1)
            rec_history = {
                'rec': rec.to_dict(orient='records'),
                'obj': None if y_none else y.tolist(),
                'rec_time': rec_time,
                'run_time': run_time
            }
            rec_history_list.append(rec_history)
            random_forest = random_forest_regressor(obtain_random_forest_train_set(rec_history_list, param_names=space_param_names))

            with open(rec_history_file_path, 'w') as fp:
                json.dump(rec_history_list, fp)

            i += 1
            logging.info(f'total_tune_num: {i}, succeed_num: {succeed_num}, failed_num: {failed_num}')
            logging.info(f'For the {i}th iteration, BO time cost is {rec_time}')
            logging.info(f'For the {i}th iteration, rec config is {rec}')
            logging.info(f'For the {i}th iteration, rec obj is {y}')
            logging.info(f'After {i} iterations, best config is {opt.best_x}')
            logging.info(f'After {i} iterations, best obj is {opt.best_y}')
            logging.info(f'After {i} iterations, the threshold delta is {delta}')
            logging.info(f'After {i} iterations, the continuous right number is {continuous_right}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Script for raw data generation !!')
    parser = add_args(parser)
    arguments = parser.parse_args()
    main(arguments)
