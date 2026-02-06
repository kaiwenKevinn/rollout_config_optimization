#!/bin/bash
# 完整工作流示例：从profiling到对比分析

echo "=== 完整异构TP配置测试工作流 ==="

# 设置变量
WORK_DIR="/research/d1/gds/ytyang/kwchen/hetero_rollout"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
DATASET_TYPE="gpqa"  # 或 "aime25"

cd $WORK_DIR

echo "开始时间: $(date)"
echo "使用的数据集: $DATASET_TYPE"

# 1. Profiling阶段 - 获取序列长度分布
echo ""
echo "步骤1: 运行Profiling阶段..."
PROFILING_OUTPUT="results/profiling_${DATASET_TYPE}_${TIMESTAMP}"
python scripts/run_profiling.py \
    --config config/config.yaml \
    --output "$PROFILING_OUTPUT" \
    --dataset-type "$DATASET_TYPE" \
    --with-inference \
    --tp 4 \
    --gpus 0,1,2,3 \
    --max-concurrent 4 \
    --log-level INFO

if [ $? -ne 0 ]; then
    echo "错误: Profiling阶段失败"
    exit 1
fi

echo "Profiling结果保存在: $PROFILING_OUTPUT"

# 2. 同构配置测试
echo ""
echo "步骤2: 运行同构配置测试..."
HOMOGENEOUS_OUTPUT="results/homogeneous_${DATASET_TYPE}_${TIMESTAMP}"
python scripts/run_homogeneous.py \
    --config config/config.yaml \
    --output "$HOMOGENEOUS_OUTPUT" \
    --profiling-dir "$PROFILING_OUTPUT" \
    --runs 3 \
    --log-level INFO

if [ $? -ne 0 ]; then
    echo "警告: 同构配置测试部分失败"
fi

echo "同构测试结果保存在: $HOMOGENEOUS_OUTPUT"

# 3. 异构配置测试
echo ""
echo "步骤3: 运行异构配置测试..."
HETEROGENEOUS_OUTPUT="results/heterogeneous_${DATASET_TYPE}_${TIMESTAMP}"
python scripts/run_heterogeneous.py \
    --config config/config.yaml \
    --output "$HETEROGENEOUS_OUTPUT" \
    --profiling-dir "$PROFILING_OUTPUT" \
    --runs 3 \
    --log-level INFO

if [ $? -ne 0 ]; then
    echo "警告: 异构配置测试部分失败"
fi

echo "异构测试结果保存在: $HETEROGENEOUS_OUTPUT"

# 4. 对比分析
echo ""
echo "步骤4: 执行对比分析..."
ANALYSIS_OUTPUT="results/comparison_${DATASET_TYPE}_${TIMESTAMP}"
python examples/comparison_analysis.py \
    --homogeneous-results "$HOMOGENEOUS_OUTPUT" \
    --heterogeneous-results "$HETEROGENEOUS_OUTPUT" \
    --output "$ANALYSIS_OUTPUT"

if [ $? -ne 0 ]; then
    echo "警告: 对比分析失败"
fi

echo "对比分析结果保存在: $ANALYSIS_OUTPUT"

# 5. 生成总结报告
echo ""
echo "步骤5: 生成总结报告..."
SUMMARY_FILE="$ANALYSIS_OUTPUT/summary_$(date +%Y%m%d).md"

cat > "$SUMMARY_FILE" << EOF
# 异构TP配置测试总结报告

**测试时间**: $(date)
**数据集类型**: $DATASET_TYPE
**测试标识**: $TIMESTAMP

## 目录结构
- Profiling结果: $PROFILING_OUTPUT
- 同构测试结果: $HOMOGENEOUS_OUTPUT  
- 异构测试结果: $HETEROGENEOUS_OUTPUT
- 对比分析结果: $ANALYSIS_OUTPUT

## 关键发现

*(在此处添加您的观察和结论)*

## 建议

*(在此处添加基于测试结果的建议)*

---
*本报告由自动化测试工作流生成*
EOF

echo "总结报告已生成: $SUMMARY_FILE"

echo ""
echo "=== 工作流完成 ==="
echo "结束时间: $(date)"
echo "所有结果已保存在 results/ 目录下"