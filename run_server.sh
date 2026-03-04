#!/bin/bash
# conda activate /research/d1/gds/ytyang/kwchen/hetero_rollout/hetero_env
export no_proxy=localhost,127.0.0.1,192.168.50.186
model_path=$1
port=$2
tp_size=$3
pp_size=$4
max_num_seqs=$5
max_num_batched_tokens=$6
scheduler_delay_factor=$7
block_size=$8
enable_chunked_prefill=$9
enable_prefix_caching=${10}
disable_custom_all_reduce=${11}
use_v2_block_manager=${12}
enable_expert_parallel=${13:-False}
int_port=$((port + 0))
additional_options=""

if [ "${enable_chunked_prefill}" == "True" ]; then
    additional_options="--enable-chunked-prefill "
fi

if [ "${enable_prefix_caching}" == "True" ]; then
    additional_options+="--enable-prefix-caching "
fi

if [ "${disable_custom_all_reduce}" == "True" ]; then
    additional_options+="--disable-custom-all-reduce "
fi

if [ "${use_v2_block_manager}" == "True" ]; then
    additional_options+="--use-v2-block-manager "
fi

if [ "${enable_expert_parallel}" == "True" ]; then
    additional_options+="--enable-expert-parallel "
fi

echo run_server.sh
echo tp_size ${tp_size}
echo pp_size ${pp_size}
echo enable_chunked_prefill ${enable_chunked_prefill}
echo enable_prefix_caching ${enable_prefix_caching}
echo disable_custom_all_reduce ${disable_custom_all_reduce}
echo use_v2_block_manager ${use_v2_block_manager}
echo enable_expert_parallel ${enable_expert_parallel}
 # --max-num-batched-tokens 32768 \

echo  python -m clients.api_server \
    --model ${model_path} \
    --disable-log-requests \
    --max-num-batched-tokens 32768\
    --max-num-seqs ${max_num_seqs} \
    --scheduler-delay-factor ${scheduler_delay_factor} \
    --port ${int_port} \
    --tensor-parallel-size ${tp_size} \
    --pipeline-parallel-size ${pp_size} \
    --block-size ${block_size} \
    --gpu-memory-utilization 0.9\
    --trust-remote-code\
    --enforce-eager\
    --disable-async-output-proc\
    --rope-scaling '{"rope_type": "linear", "factor": 1.0}'\
    $additional_options
#     

python -m clients.api_server \
    --model ${model_path} \
    --disable-log-requests \
    --max-num-batched-tokens 32768 \
    --max-num-seqs ${max_num_seqs} \
    --scheduler-delay-factor ${scheduler_delay_factor} \
    --port ${int_port} \
    --tensor-parallel-size ${tp_size} \
    --pipeline-parallel-size ${pp_size} \
    --block-size ${block_size} \
    --gpu-memory-utilization 0.9\
    --trust-remote-code\
    --enforce-eager\
    --disable-async-output-proc\
    --rope-scaling '{"rope_type": "linear", "factor": 1.0}'\
    $additional_options