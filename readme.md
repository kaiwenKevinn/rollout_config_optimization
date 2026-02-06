# 1. 安装依赖
pip install -r requirements.txt

# 2. 运行Profiling阶段
python scripts/run_profiling.py

python3 scripts/run_profiling.py --with-inference --tp 2 --gpus 0,1,3

export NO_PROXY="localhost,127.0.0.1,192.168.50.186"
nohup python3 scripts/run_profiling.py --with-inference --config config/config_gpqa.yaml --dataset-type gpqa --output results/gpqa_test --tp 2 --gpus 1,2 --log-file ./logs/gpqa_benchmark.log  & 

export NO_PROXY="localhost,127.0.0.1,192.168.50.186"

nohup python scripts/run_profiling.py --with-inference --config config/config_aime25.yaml --dataset-type aime25 --output results/aime25_test --tp 2 --gpus 1,2 --log-file ./logs/aime25_benchmark.log  & 


# 3. 运行同构测试 (TP=1, 2, 4, 8)
python scripts/run_homogeneous.py --tp 1 2 4 --runs 1 --log-file ./logs/aime25_homogeneous

nohup python scripts/run_homogeneous.py \
    --config config/config_aime25.yaml \
    --output results/homogeneous_adaptive \
    --profiling-dir results/aime25_test \
    --runs 2 \
    --output results/test_homo \
    --tp 2 4 \
    --exclude-tp1 \
    --log-level INFO > aime25_homo_test.out 2>&1 &
# 4. 运行异构测试
python scripts/run_heterogeneous.py --scenarios all --runs 3 --output results/test_hetero

# 5. 运行完整测试套件
python scripts/run_full_benchmark.py --runs 3