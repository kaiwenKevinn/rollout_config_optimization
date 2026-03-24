#!/bin/bash
# export CUDA_VISIBLE_DEVICES="0,1,2,5"  # 注释掉CUDA设置，NPU不需要此环境变量


# 枚举配置测试入口脚本 (NPU 910B 版本)
# 用法:
#   bash run_entry_enum_configs.sh <model_path> <model_name> <dataset_path> <dataset_name> <request_rate> <request_num> <npu_num> <npu_type> [config_file|--auto_enum] [max_configs]
#
# 示例:
#   # 从 JSON 文件加载配置
#   bash run_entry_enum_configs.sh /path/to/model Qwen3_32B /path/to/dataset gpqa_diamond 198 198 8 910B configs.json
#
#   # 自动枚举（最多 30 个配置）
#   nohup bash run_entry_enum_configs.sh /research/d1/gds/ytyang/kwchen/hf_models/Qwen/Qwen3-32B/models--Qwen--Qwen3-32B/snapshots/9216db5781bf21249d130ec9da846c4624c16137 Qwen3_32B /research/d1/gds/ytyang/kwchen/hetero_rollout/results/gpqa_test/classification_by_range/sequence_profile_bucket_0.json gpqa_diamond_bucket_0 29 29 4 910B --auto_enum 30 > run_entry_enum_configs_qwen3_32b_bucket_0_6_params_npu.log 2>&1 &
#   nohup bash run_entry_enum_configs.sh /research/d1/gds/ytyang/kwchen/hf_models/Qwen/Qwen3-32B/models--Qwen--Qwen3-32B/snapshots/9216db5781bf21249d130ec9da846c4624c16137 Qwen3_32B /research/d1/gds/ytyang/kwchen/hetero_rollout/results/gpqa_test/classification_by_range/sequence_profile_bucket_0_enlarge.json gpqa_diamond_bucket_0_enlarge 232 232 4 910B --auto_enum 30 > run_entry_enum_configs_qwen3_32b_bucket_0_3_tp_pp_mns_params_npu.log 2>&1 &
#   nohup bash run_entry_enum_configs.sh /research/d1/gds/ytyang/kwchen/hf_models/Qwen/Qwen3-32B/models--Qwen--Qwen3-32B/snapshots/9216db5781bf21249d130ec9da846c4624c16137 Qwen3_32B /research/d1/gds/ytyang/kwchen/hetero_rollout/results/gpqa_test/classification_by_range/sequence_profile_bucket_3_small.json gpqa_diamond_bucket_3_small 20 20 8 910B --auto_enum 30 resume > run_entry_enum_configs_qwen3_32b_bucket_3_small_npu.log 2>&1 &
#
# 断点续测（两种方式）：
#   方式 1 - 自动使用最新目录：第 11 参数为 "resume"
#     bash run_entry_enum_configs.sh ... 8 910B --auto_enum 30 resume
#   方式 2 - 指定目录续测：第 11 参数为 "resume"，第 12 参数为 output_dir 路径
#     bash run_entry_enum_configs.sh ... 8 910B --auto_enum 30 resume ./tune_res/enum_results/Qwen3_32B_gpqa_diamond_bucket_0_20260304_232428

model_path=$1
model_name=$2
dataset_path=$3
dataset_name=$4
request_rate=$5
request_num=$6
npu_num=$7
npu_type=$8

# NPU环境下不需要设置CUDA_VISIBLE_DEVICES
# 检查NPU设备状态
echo "检查NPU设备状态..."
if command -v npu-smi &> /dev/null; then
    echo "节点总NPU数量: $(npu-smi info | grep -c "Device ID")"
    npu-smi info
else
    echo "警告: 未找到npu-smi命令，请确认NPU驱动已正确安装"
fi
config_mode=${9:-"--auto_enum"}  # config_file 路径 或 --auto_enum
max_configs=${10:-""}            # auto_enum 时的最大配置数（可选）
use_resume=${11:-""}             # 设为 "resume" 启用断点续测
resume_output_dir=${12:-""}      # 断点续测时指定 output_dir（可选）；不指定则自动使用最新目录

# NPU环境下的环境变量设置
export VLLM_TORCH_COMPILE_LEVEL=NONE
export TORCHINDUCTOR_DISABLE=1
export TORCH_DYNAMO_DISABLE=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_GRAPH_RESERVED_MEM=0.1
export VLLM_GRAPH_COMPILATION=0
export no_proxy=localhost,127.0.0.1,192.168.50.186

# NPU特定环境变量（如果需要）
# export ASCEND_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"  # 根据实际NPU数量调整
# export DEVICE_TYPE=npu

# 获取 tuner 配置
cd tuner_conf
bash tuner_conf.sh ${model_path}
cd ..

resume_arg=""
if [ "${use_resume}" == "resume" ]; then
    resume_arg="--resume"
    if [ -n "${resume_output_dir}" ]; then
        resume_arg="${resume_arg} --output_dir ${resume_output_dir}"
        echo "启用断点续测 (--resume)，指定目录: ${resume_output_dir}"
    else
        echo "启用断点续测 (--resume)，将自动使用最新同 model_dataset 目录"
    fi
fi

# 调优参数类型：枚举 tp, pp, block_size, scheduler_delay_factor（其他参数使用默认值）
# TUNE_PARAMS="tp,pp,block_size,scheduler_delay_factor,max_num_seqs,max_num_batched_tokens"
TUNE_PARAMS="tp,pp,max_num_seqs"

if [ "${config_mode}" == "--auto_enum" ]; then
    if [ -n "${max_configs}" ]; then
        python enum_configs.py --auto_enum --max_configs ${max_configs} --tune_params ${TUNE_PARAMS} ${resume_arg} \
            --model_path ${model_path} --dataset_path ${dataset_path} --dataset_name ${dataset_name} \
            --model ${model_name} --total_resource ${npu_num}${npu_type} \
            --request_rate ${request_rate} --num_requests ${request_num}
    else
        python enum_configs.py --auto_enum --tune_params ${TUNE_PARAMS} ${resume_arg} \
            --model_path ${model_path} --dataset_path ${dataset_path} --dataset_name ${dataset_name} \
            --model ${model_name} --total_resource ${npu_num}${npu_type} \
            --request_rate ${request_rate} --num_requests ${request_num}
    fi
else
    python enum_configs.py --config_file ${config_mode} ${resume_arg} \
        --model_path ${model_path} --dataset_path ${dataset_path} --dataset_name ${dataset_name} \
        --model ${model_name} --total_resource ${npu_num}${npu_type} \
        --request_rate ${request_rate} --num_requests ${request_num}
fi

# 指定目录续测示例：
#   bash run_entry_enum_configs.sh ... 8 910B --auto_enum 30 resume ./tune_res/enum_results/Qwen3_32B_gpqa_diamond_bucket_0_20260304_232428

# 注意事项：
# 1. 在NPU环境下运行前，请确保：
#    - 已安装Ascend CANN toolkit
#    - 已正确配置NPU驱动
#    - Python环境中已安装NPU版本的PyTorch
#    - vLLM已编译支持NPU后端
# 2. 参数说明：
#    npu_num: NPU设备数量 (如 4, 8)
#    npu_type: NPU型号 (如 910B)
# 3. 如果遇到内存不足问题，可以适当减少max_num_seqs等参数值
