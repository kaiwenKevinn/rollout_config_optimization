#!/bin/bash
# SCOOT 序列分桶自动化工作流

set -e

# 配置变量
MODEL_PATH=${1:-"/path/to/your/model"}
DATASET_PATH=${2:-"/path/to/dataset"}
PROFILING_OUTPUT="./profiling_results"
BUCKET_CONFIG="./bucket_config.json"
VISUALIZATION_OUTPUT="./bucket_analysis.png"

echo "🚀 Starting SCOOT Sequence Bucketing Workflow"
echo "==========================================="

# 步骤1: 运行序列 profiling
echo "📋 Step 1: Running sequence profiling..."
python -m profiler.sequence_profiler \
    --model-path $MODEL_PATH \
    --dataset-path $DATASET_PATH \
    --output-dir $PROFILING_OUTPUT

# 步骤2: 分析序列长度分布并优化分桶
echo "📊 Step 2: Analyzing sequence distribution and optimizing buckets..."
python bucket_optimizer.py \
    --profiling-dir $PROFILING_OUTPUT \
    --method percentile \
    --output-config $BUCKET_CONFIG \
    --output-plot $VISUALIZATION_OUTPUT

# 步骤3: 验证分桶配置
echo "✅ Step 3: Validating bucket configuration..."
python -c "
import json
with open('$BUCKET_CONFIG', 'r') as f:
    config = json.load(f)
print('Generated Bucket Configuration:')
print(json.dumps(config, indent=2))
"

# 步骤4: 启动带有优化分桶的 SCOOT 调优
echo "🎯 Step 4: Starting SCOOT tuning with optimized buckets..."

# 更新 tuner_conf 以包含新的分桶配置
python -c "
import json
# 读取现有配置
with open('./tuner_conf/conf.json', 'r') as f:
    base_config = json.load(f)

# 读取分桶配置
with open('$BUCKET_CONFIG', 'r') as f:
    bucket_config = json.load(f)

# 合并配置
base_config.update(bucket_config)
with open('./tuner_conf/conf.json', 'w') as f:
    json.dump(base_config, f, indent=2)

print('Configuration updated successfully')
"

# 运行 SCOOT 调优
echo "Starting BO tuning with optimized sequence buckets..."
python bo_scoot.py \
    --model_path $MODEL_PATH \
    --dataset_path $DATASET_PATH \
    --dataset_name "optimized_buckets" \
    --model "your_model_name" \
    --total_resource "8gpu_mobo" \
    --request_rate 20 \
    --num_requests 1000 \
    --bo_loop 30 \
    --exp_num 1 \
    --num_obj 3

echo "🎉 Workflow completed successfully!"
echo "Results:"
echo "- Profiling data: $PROFILING_OUTPUT"
echo "- Bucket configuration: $BUCKET_CONFIG"  
echo "- Analysis visualization: $VISUALIZATION_OUTPUT"