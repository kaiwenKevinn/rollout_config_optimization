#!/bin/bash
# args
model_path=$1
dataset_path=$2
request_rate=$3
num_requests=$4
pressure_test=$5
max_concurrent_requests=$6

# parameter combination
tp_size=$7
pp_size=$8
max_num_seqs=$9
max_num_batched_tokens=${10}
scheduler_delay_factor=${11}
block_size=${12}
port=${13}
device_group=${14}
model=${15}
enable_chunked_prefill=${16}
dataset_name=${17}
enable_prefix_caching=${18}
disable_custom_all_reduce=${19}
use_v2_block_manager=${20}
sequence_profile_path=${21:-}
enable_expert_parallel=${22:-False}
OLD_IFS="$IFS"
IFS=','
read -ra ADDR <<< "$port"
IFS="$OLD_IFS"

OLD_IFS="$IFS"
IFS='#'
read -ra GPU_ADDR <<< "$device_group"
IFS="$OLD_IFS"

echo benchmark_pipeine.sh
echo model_path=$1 ${model_path}
echo dataset_path=$2 ${dataset_path}
echo request_rate=$3 ${request_rate}
echo num_requests=$4 ${num_requests}
echo pressure_test=$5 ${pressure_test}
echo max_concurrent_requests=$6 ${max_concurrent_requests}
echo tp_size=$7 ${tp_size}
echo pp_size=$8 ${pp_size}
echo max_num_seqs=$9 ${max_num_seqs}
echo max_num_batched_tokens=${10} ${max_num_batched_tokens}
echo scheduler_delay_factor=${11} ${scheduler_delay_factor}
echo block_size=${12} ${block_size}
echo port=${13} ${port}
echo device_group=${14} ${device_group}
echo model=${15} ${model}
echo enable_chunked_prefill=${16} ${enable_chunked_prefill}
echo dataset_name=${17} ${dataset_name}
echo enable_prefix_caching=${18} ${enable_prefix_caching}
echo disable_custom_all_reduce=${19} ${disable_custom_all_reduce}
echo use_v2_block_manager=${20} ${use_v2_block_manager}
echo sequence_profile_path=${21} ${sequence_profile_path}
echo enable_expert_parallel=${22} ${enable_expert_parallel}

# 启动前清理目标端口，避免 "address already in use" 错误
echo "Pre-cleaning ports: ${port}"
OLD_IFS="$IFS"
IFS=','
read -ra PRE_CLEAN_PORTS <<< "$port"
IFS="$OLD_IFS"
for p in "${PRE_CLEAN_PORTS[@]}"; do
    if [ -n "$p" ] && [ "$p" -gt 0 ] 2>/dev/null; then
        pids=$(lsof -t -i:${p} 2>/dev/null)
        if [ -n "$pids" ]; then
            echo "Pre-killing processes on port $p (PIDs: $pids) to avoid port conflict"
            echo "$pids" | xargs kill -9 2>/dev/null || true
            sleep 2
        fi
    fi
done
# 额外清理可能残留的 api_server 进程
for pid in $(pgrep -f "clients.api_server" 2>/dev/null); do
    echo "Pre-killing stale api_server pid: $pid"
    kill -9 $pid 2>/dev/null || true
done
sleep 2
echo "Port pre-clean done."

echo "server start!"
for ((i=0; i<${#ADDR[@]}; i++)); do
    device=${GPU_ADDR[i]}
    echo device:${device}
    echo addr:${ADDR[i]}
    CUDA_VISIBLE_DEVICES=${device} bash run_server.sh ${model_path} ${ADDR[i]} ${tp_size} ${pp_size} ${max_num_seqs} ${max_num_batched_tokens} ${scheduler_delay_factor} ${block_size} ${enable_chunked_prefill} ${enable_prefix_caching} ${disable_custom_all_reduce} ${use_v2_block_manager} ${enable_expert_parallel}&
done
echo "finish server start!"

echo "client start!"
bash run_client.sh ${model_path} ${dataset_path} ${request_rate} ${num_requests} ${pressure_test} ${max_concurrent_requests} ${tp_size} ${pp_size} ${max_num_seqs} ${max_num_batched_tokens} ${scheduler_delay_factor} ${block_size} ${port} ${model} ${dataset_name} ${enable_chunked_prefill} ${enable_prefix_caching} ${disable_custom_all_reduce} ${use_v2_block_manager} ${sequence_profile_path} ${enable_expert_parallel}
echo "finish client start!"

# 等待 client 完全退出，确保端口可被下一轮复用
sleep 5

# Kill the whole process and then kill each port to ensure that the engine process has been fully killed
echo "Killing vLLM API server processes..."
for pid in $(pgrep -f "clients.api_server" 2>/dev/null); do
    echo "Killing vLLM pid: $pid"
    kill -9 $pid 2>/dev/null || true
done

# 按实际使用的端口杀进程（使用传入的 port 变量，避免 sudo）
OLD_IFS="$IFS"
IFS=','
read -ra KILL_PORTS <<< "$port"
IFS="$OLD_IFS"
for p in "${KILL_PORTS[@]}"; do
    if [ -n "$p" ] && [ "$p" -gt 0 ] 2>/dev/null; then
        pids=$(lsof -t -i:${p} 2>/dev/null)
        if [ -n "$pids" ]; then
            echo "Killing processes on port $p: $pids"
            echo "$pids" | xargs kill -9 2>/dev/null || true
        fi
    fi
done

# 再次等待端口完全释放（TIME_WAIT 等）
sleep 10
echo "Done."