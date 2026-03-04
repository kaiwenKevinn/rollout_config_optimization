yes | cp ./clients/api_server.py {path to vllm/entrypoints/}
bash run_entry_bo_scoot.sh {path to model weights and configs} {model_name} {path to datasets} {dataset name} {request rate IN client} {number of requests} {GPU num} {gpu gpu_type}
# bash run_entry_bo_scoot.sh ./LLaMA2-7B-fp16 llama2_7b_scoot ./sharegpt.json sharegpt 20 1000 2 L20

# Qwen-2.5-3B-Instruct  AIME 25
# nohup bash run_entry_bo_scoot.sh /research/d1/gds/ytyang/kwchen/hf_models/Qwen/Qwen2.5-3B-Instruct/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1 Qwen2.5_3B_Instruct /research/d1/gds/ytyang/kwchen/hetero_rollout/datasets/aime25_AIME2025-I_test_processed.json aime25 20 15 8 A800 > run_entry_bo_scoot.log 2>&1 &

# Qwen-3-32B GPQA Diamond
# nohup bash run_entry_bo_scoot.sh /research/d1/gds/ytyang/kwchen/hf_models/Qwen/Qwen3-32B/models--Qwen--Qwen3-32B/snapshots/9216db5781bf21249d130ec9da846c4624c16137 Qwen3_32B /research/d1/gds/ytyang/kwchen/hetero_rollout/results/gpqa_test/classification_by_range/sequence_profile_bucket_0.json gpqa_diamond_bucket_0 29 29 4 A800 > run_entry_bo_scoot_qwen3_32b_bucket_0.log 2>&1 &

pkill -f "bo_scoot.py"
pkill -f "run_entry_bo_scoot.sh"
pkill -f "api_server.py"
pkill -f "benchmark_serving"
pkill -f "enum_configs.py"


pkill -9 -f "clients.api_server"


pkill -f "bo_scoot.py"                                                                          1
pkill -f "run_entry_bo_scoot.sh"
pkill -f "api_server.py"
pkill -f "benchmark_serving"
pkill -9 -f "clients.api_server"
pkill -9 -f "benchmark_serving"
pkill -9 -f "benchmark_pipeline"
pkill -9 -f "run_server.sh"
pkill -9 -f "run_client.sh"
pkill -f nohup