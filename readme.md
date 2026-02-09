# 1. 安装依赖
pip install -r requirements.txt

# 2. 运行Profiling阶段
python scripts/run_profiling.py

python3 scripts/run_profiling.py --with-inference --tp 2 --gpus 0,1,3

export NO_PROXY="localhost,127.0.0.1,192.168.50.186"
nohup python3 scripts/run_profiling.py --with-inference --config config/config_gpqa.yaml --dataset-type gpqa --output results/gpqa_test --tp 2 --gpus 1,2 --log-file ./logs/gpqa_benchmark.log  & 

export NO_PROXY="localhost,127.0.0.1,192.168.50.186"

nohup python scripts/run_profiling.py --with-inference --config config/config_aime25.yaml --dataset-type aime25 --output results/aime25_test --tp 2 --gpus 1,2 --log-file ./logs/aime25_benchmark.log  & 


# 3. 运行同构测试 (TP=1, 2, 4)

# aime
NO_PROXY="localhost,127.0.0.1,192.168.50.186" \
conda activate /research/d1/gds/ytyang/kwchen/hetero_rollout/hetero_env && \
nohup python scripts/run_homogeneous.py \
    --config config/config_aime25.yaml \
    --profiling-dir results/aime25_test \
    --runs 2 \
    --exclude-tp1 \
    --tp 2 4 \
    --output results/test_homo_aime \
    --log-level INFO > aime25_homo_test.out 2>&1 &

# gpqa
NO_PROXY="localhost,127.0.0.1,192.168.50.186" \
conda activate /research/d1/gds/ytyang/kwchen/hetero_rollout/hetero_env && \
nohup python scripts/run_homogeneous.py \
    --config config/config_gpqa.yaml \
    --profiling-dir results/gpqa_test \
    --runs 2 \
    --exclude-tp1 \
    --tp 2 4 \
    --output results/test_homo_gpqa \
    --log-level INFO > aime25_homo_test.out 2>&1 &

# 4. 运行异构测试

# aime
NO_PROXY="localhost,127.0.0.1,192.168.50.186" \
nohup python scripts/run_heterogeneous.py \
    --config config/config_aime25.yaml \
    --output results/heterogeneous_aime \
    --use-config-hetero \
    --profiling-dir ./results/aime25_test \
    --scenarios mix1 mix2 \
    --runs 2 \
    --output results/test_hetero \
    --log-level INFO > aime25_hetero_test.out 2>&1 &

# gpqa
NO_PROXY="localhost,127.0.0.1,192.168.50.186" \
nohup python scripts/run_heterogeneous.py \
    --config config/config_gpqa.yaml \
    --output results/heterogeneous_gpqa\
    --use-config-hetero \
    --profiling-dir ./results/gpqa_test \
    --scenarios mix1 mix2 \
    --runs 2 \
    --output results/test_hetero \
    --log-level INFO > hetero_test_gpqa.out 2>&1 &

pkill -f "vllm.entrypoints.openai.api_server" && sleep 3 && netstat -tulnp | grep -E "8000|8001|8002|8003"

# 5. 运行完整测试套件
python scripts/run_full_benchmark.py --runs 3