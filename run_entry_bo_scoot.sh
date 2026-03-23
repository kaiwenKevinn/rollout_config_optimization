model_path=$1
model_name=$2
dataset_path=$3
dataset_name=$4
request_rate=$5
request_num=$6
gpu_num=$7
gpu_type=$8

# 启动前清理残留的 vLLM 进程，避免端口占用导致 RuntimeError
echo "Pre-cleaning: killing any stale clients.api_server processes..."
pkill -9 -f "clients.api_server" 2>/dev/null || true
sleep 5
# 若 8000 等端口仍被占用（如 TIME_WAIT），多等一会
for port in 8000 8001 8010 8011 8020 8021; do
    if lsof -i:${port} 2>/dev/null | grep -q LISTEN; then
        echo "Port ${port} still in use, force killing..."
        lsof -t -i:${port} 2>/dev/null | xargs kill -9 2>/dev/null || true
        sleep 3
    fi
done
sleep 10
echo "Pre-clean done."

# install requirements
# pip install -r requirements.txt
export VLLM_TORCH_COMPILE_LEVEL=NONE
export TORCHINDUCTOR_DISABLE=1
export TORCH_DYNAMO_DISABLE=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_GRAPH_RESERVED_MEM=0.1
export VLLM_GRAPH_COMPILATION=0
export no_proxy=localhost,127.0.0.1,192.168.50.186
export CUDA_VISIBLE_DEVICES="4,5,6,7"
# obtain the default tp, and update max_sequence_length
cd tuner_conf
bash tuner_conf.sh ${model_path}
cd ..


echo submit.sh
echo model_path=${model_path}
echo model_name=${model_name}
echo dataset_path=${dataset_path}
echo dataset_name=${dataset_name}
echo request_rate=${request_rate}
echo request_num=${request_num}
echo gpu_num=${gpu_num}
echo gpu_type=${gpu_type}


# 调优参数类型：仅调优 tp, pp, block_size（其他参数使用默认值）
TUNE_PARAMS="tp,pp,block_size"

python bo_scoot.py --model_path ${model_path}\
                    --dataset_path ${dataset_path}\
                    --dataset_name ${dataset_name}\
                    --model ${model_name}\
                    --total_resource ${gpu_num}${gpu_type}_mobo\
                    --request_rate ${request_rate}\
                    --bo_loop 30\
                    --exp_num 1\
                    --num_requests ${request_num}\
                    --num_obj 1\
                    --tune_params ${TUNE_PARAMS}