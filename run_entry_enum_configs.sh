#!/bin/bash
# 枚举配置测试入口脚本
# 用法:
#   bash run_entry_enum_configs.sh <model_path> <model_name> <dataset_path> <dataset_name> <request_rate> <request_num> <gpu_num> <gpu_type> [config_file|--auto_enum] [max_configs]
#
# 示例:
#   # 从 JSON 文件加载配置
#   bash run_entry_enum_configs.sh /path/to/model Qwen3_32B /path/to/dataset gpqa_diamond 198 198 8 A800 configs.json
#
#   # 自动枚举（最多 30 个配置）
#   nohup bash run_entry_enum_configs.sh /research/d1/gds/ytyang/kwchen/hf_models/Qwen/Qwen3-32B/models--Qwen--Qwen3-32B/snapshots/9216db5781bf21249d130ec9da846c4624c16137 Qwen3_32B /research/d1/gds/ytyang/kwchen/hetero_rollout/results/gpqa_test/classification_by_range/sequence_profile_bucket_0.json gpqa_diamond_bucket_0 29 29 4 A800 --auto_enum 30 > run_entry_enum_configs_qwen3_32b_bucket_0.log 2>&1 &

#   nohup bash run_entry_enum_configs.sh /research/d1/gds/ytyang/kwchen/hf_models/Qwen/Qwen3-32B/models--Qwen--Qwen3-32B/snapshots/9216db5781bf21249d130ec9da846c4624c16137 Qwen3_32B /research/d1/gds/ytyang/kwchen/hetero_rollout/results/gpqa_test/classification_by_range/sequence_profile_bucket_3.json gpqa_diamond_bucket_0 163 163 8 A800 --auto_enum 30 > run_entry_enum_configs_qwen3_32b_bucket_3.log 2>&1 &

model_path=$1
model_name=$2
dataset_path=$3
dataset_name=$4
request_rate=$5
request_num=$6
gpu_num=$7
gpu_type=$8

# 设置可见GPU范围
if [ "${gpu_num}" == "4" ]; then
    export CUDA_VISIBLE_DEVICES="4,5,6,7"
elif [ "${gpu_num}" == "8" ]; then
    export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
elif [ "${gpu_num}" == "2" ]; then
    export CUDA_VISIBLE_DEVICES="6,7"
fi

echo "设置 CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "实际可用GPU数量: $(nvidia-smi -L | wc -l)"
config_mode=${9:-"--auto_enum"}  # config_file 路径 或 --auto_enum
max_configs=${10:-""}            # auto_enum 时的最大配置数（可选）

export VLLM_TORCH_COMPILE_LEVEL=NONE
export TORCHINDUCTOR_DISABLE=1
export TORCH_DYNAMO_DISABLE=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_GRAPH_RESERVED_MEM=0.1
export VLLM_GRAPH_COMPILATION=0
export no_proxy=localhost,127.0.0.1,192.168.50.186

# 获取 tuner 配置
cd tuner_conf
bash tuner_conf.sh ${model_path}
cd ..

if [ "${config_mode}" == "--auto_enum" ]; then
    if [ -n "${max_configs}" ]; then
        python enum_configs.py --auto_enum --max_configs ${max_configs} \
            --model_path ${model_path} --dataset_path ${dataset_path} --dataset_name ${dataset_name} \
            --model ${model_name} --total_resource ${gpu_num}${gpu_type} \
            --request_rate ${request_rate} --num_requests ${request_num}
    else
        python enum_configs.py --auto_enum \
            --model_path ${model_path} --dataset_path ${dataset_path} --dataset_name ${dataset_name} \
            --model ${model_name} --total_resource ${gpu_num}${gpu_type} \
            --request_rate ${request_rate} --num_requests ${request_num}
    fi
else
    python enum_configs.py --config_file ${config_mode} \
        --model_path ${model_path} --dataset_path ${dataset_path} --dataset_name ${dataset_name} \
        --model ${model_name} --total_resource ${gpu_num}${gpu_type} \
        --request_rate ${request_rate} --num_requests ${request_num}
fi
