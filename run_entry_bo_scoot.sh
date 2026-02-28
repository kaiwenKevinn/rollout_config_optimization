model_path=$1
model_name=$2
dataset_path=$3
dataset_name=$4
request_rate=$5
request_num=$6
gpu_num=$7
gpu_type=$8

# install requirements
# pip install -r requirements.txt
export VLLM_TORCH_COMPILE_LEVEL=NONE
export TORCHINDUCTOR_DISABLE=1
export TORCH_DYNAMO_DISABLE=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_GRAPH_RESERVED_MEM=0.1
export VLLM_GRAPH_COMPILATION=0
export no_proxy=localhost,127.0.0.1,192.168.50.186
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
                    --sequence_profile_path /research/d1/gds/ytyang/kwchen/hetero_rollout/results/gpqa_test/sequence_profile.json\