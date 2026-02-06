#!/bin/bash
# 同构配置自适应测试示例脚本

echo "=== 同构配置自适应测试示例 ==="

# 设置工作目录
WORK_DIR="/research/d1/gds/ytyang/kwchen/hetero_rollout"
cd $WORK_DIR

# 1. 首先运行profiling阶段获取序列长度分布
echo "步骤1: 运行profiling阶段..."
python scripts/run_profiling.py \
    --config config/config.yaml \
    --output results/profiling_gpqa \
    --dataset-type gpqa \
    --with-inference \
    --tp 4 \
    --gpus 0,1,2,3 \
    --max-concurrent 4

# 2. 基于profiling结果运行自适应同构测试
echo "步骤2: 运行自适应同构配置测试..."
python scripts/run_homogeneous.py \
    --config config/config.yaml \
    --output results/homogeneous_adaptive \
    --profiling-dir results/profiling_gpqa \
    --runs 3 \
    --log-level INFO

echo "测试完成！结果保存在: results/homogeneous_adaptive"