import socket
import json
import os


def gen_res_dir_path(model, request_rate, num_requests, total_resource, dataset_name, res_dir, bo_bs=1, bo=False, exp=0,
                     dir_prefix='origin'):
    
    if not bo:
        res_dir_path = os.path.join(f'{res_dir}/{dir_prefix}',
                                    f'{model}_qps{request_rate}_prompts{num_requests}_{total_resource}_{dataset_name}',
                                    f'exp{exp}')
    else:
        res_dir_path = os.path.join(f'{res_dir}/{dir_prefix}',
                                    f'bo_{model}_qps{request_rate}_prompts{num_requests}_{total_resource}_{dataset_name}_bo_bs{bo_bs}',
                                    f'exp{exp}')
    os.makedirs(res_dir_path, exist_ok=True)
    return res_dir_path


def check_port(port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex((f'{"localhost"}', port))
    sock.close()
    return result == 0


def get_ref_config(key):
    with open('tuner_conf/conf.json', 'r') as f:
        tuner_conf = json.load(f)
        if key not in tuner_conf:
            raise ValueError(f'{key} is not found! Please check the tuner_conf/conf.json file!')
        else:
            ref_value = tuner_conf[key]
    return ref_value

def read_historical_data(res_dir_path):
    xx = []
    yy = []
    for root, dirs, files in os.walk(res_dir_path):
        for name in files:
            file_path = os.path.join(root, name)
            if not file_path.split('/')[-1].startswith("vllm"):
                continue
            with open(file_path, 'r') as f:
                res = json.load(f)

            xx.append({"tp": res['tp'],
                       "max_num_seqs": res["max_num_seqs"],
                       "max_num_batched_tokens": res["max_num_batched_tokens"],
                       "block_size": res["block_size"],
                       "enable_chunked_prefill": [True if res["enable_chunked_prefill"] == 'True' else False][0],
                       "scheduler_delay_factor": int(res["scheduler_delay_factor"] * 10),
                       "enable_prefix_caching": [True if res["enable_prefix_caching"] == 'True' else False][0],
                       "disable_custom_all_reduce": [True if res["disable_custom_all_reduce"] == 'True' else False][0],
                       "use_v2_block_manager": [True if res["use_v2_block_manager"] == 'True' else False][0]}
                    )
            # 多目标优化
            # yy.append([-1 * res["request_throughput"],
            #            res["mean_ttft_ms"],
            #            res["mean_tpot_ms"]
            #            ])

            # 将多目标合并为单目标：加权求和（与bo_scoot.py中的obj函数保持一致）
            # 权重分配：吞吐量(0.5), TTFT(0.3), TPOT(0.2)
            throughput_weight = 1.0
            ttft_weight = 0.0
            tpot_weight = 0.0
            
            # 标准化各目标值到相近范围
            normalized_throughput = -1 * res["request_throughput"] / 100.0
            normalized_ttft = res["mean_ttft_ms"] / 1000.0
            normalized_tpot = res["mean_tpot_ms"] / 100.0
            
            # 计算综合目标值
            combined_objective = (throughput_weight * normalized_throughput + 
                                ttft_weight * normalized_ttft + 
                                tpot_weight * normalized_tpot)
            
            yy.append([combined_objective])
    return xx, yy