# 1. 安装依赖
pip install -r requirements.txt

# 2. 运行Profiling阶段
python scripts/run_profiling.py

python3 scripts/run_profiling.py --with-inference --tp 2 --gpus 0,1,3


# 3. 运行同构测试 (TP=1, 2, 4, 8)
python scripts/run_homogeneous.py --tp 1 2 4 --runs 3

# 4. 运行异构测试
python scripts/run_heterogeneous.py --scenarios all --runs 3

# 5. 运行完整测试套件
python scripts/run_full_benchmark.py --runs 3