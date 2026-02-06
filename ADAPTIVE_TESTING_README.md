# 自适应异构TP配置测试框架

## 概述

本框架实现了基于profiling结果的自适应测试配置，能够根据实际序列长度分布智能地配置同构和异构测试场景。

## 主要特性

### 1. Profiling驱动的配置优化
- **真实序列分析**: 通过实际模型推理获取准确的序列长度分布
- **自适应阈值**: 基于统计数据动态调整长度分类阈值
- **智能资源配置**: 根据序列分布特点自动推荐最优的TP配置

### 2. 增强的同构测试 (场景1)
- 支持基于profiling结果的自适应TP配置
- 针对不同长度类别优化实例数量
- 提供基准性能指标用于后续对比

### 3. 智能异构测试 (场景2)
- 基于序列长度分布的智能路由策略
- 动态分配不同TP度的实例组合
- 实现负载均衡和资源利用最优化

## 使用方法

### 快速开始

```bash
# 给脚本添加执行权限
chmod +x examples/*.sh

# 运行完整工作流
./examples/full_workflow_example.sh
```

### 分步执行

#### 1. Profiling阶段
```bash
# 获取序列长度分布（必须步骤）
python scripts/run_profiling.py \
    --config config/config.yaml \
    --output results/profiling_gpqa \
    --dataset-type gpqa \
    --with-inference \
    --tp 4 \
    --gpus 0,1,2,3
```

#### 2. 同构配置测试
```bash
# 使用profiling结果进行自适应配置
python scripts/run_homogeneous.py \
    --config config/config.yaml \
    --output results/homogeneous_test \
    --profiling-dir results/profiling_gpqa \
    --runs 3
```

#### 3. 异构配置测试
```bash
# 使用profiling结果进行自适应配置
python scripts/run_heterogeneous.py \
    --config config/config.yaml \
    --output results/heterogeneous_test \
    --profiling-dir results/profiling_gpqa \
    --runs 3
```

#### 4. 结果对比分析
```bash
python examples/comparison_analysis.py \
    --homogeneous-results results/homogeneous_test \
    --heterogeneous-results results/heterogeneous_test \
    --output results/comparison
```

## 核心组件

### Profiling结果加载器 (`src/utils/profiling_loader.py`)

提供以下功能：
- 加载和解析profiling阶段生成的JSON结果
- 分析序列长度分布统计
- 基于实际数据优化长度阈值
- 选择代表性测试序列
- 生成自适应测试配置

### 自适应场景生成

系统会根据profiling结果自动创建：

**同构场景**：
- 针对主要序列长度类别优化TP配置
- 动态调整实例数量以匹配序列分布
- 例如：如果短序列占主导，则优先配置更多TP=1实例

**异构场景**：
- 混合不同TP度的实例组合
- 基于序列分布智能分配GPU资源
- 实现负载均衡和性能优化

## 输出结果

### Profiling分析报告示例
```
PROFILING RESULTS ANALYSIS REPORT
================================================================================

Dataset Summary:
  Total Questions: 150
  Average Question Length: 145.2 chars
  Length Range: 53 - 847 chars

Sequence Analysis (Total-based, including output):
  Total Thresholds: short<=5384, medium<=12168, long<=15152

Distribution by Total Category (Input + Output):
  short       :  89 (59.3%)
  medium      :  45 (30.0%)
  long        :  16 (10.7%)

Actual Total Token Statistics (Input + Output):
  Min:    156  Max:  18745  Mean:  4231.7
  P50:   2847  P90:  10234  P99:  16789
================================================================================
```

### 自动生成的配置建议
```
Created 3 adaptive homogeneous configurations
  - TP=1: 4 instances - Optimized for short sequences (89 sequences)
  - TP=2: 2 instances - Optimized for medium sequences (45 sequences)  
  - TP=4: 1 instances - Optimized for long sequences (16 sequences)

Created 2 adaptive heterogeneous configurations
  - adaptive_hetero_1: Adaptive heterogeneous configuration based on sequence distribution
  - balanced_hetero: Balanced heterogeneous configuration
```

## 配置选项

### 命令行参数

**Profiling脚本**：
- `--with-inference`: 启用实际推理获取准确输出长度
- `--tp`: 推理实例的TP度数
- `--gpus`: 指定使用的GPU ID
- `--max-concurrent`: 最大并发请求数

**测试脚本**：
- `--profiling-dir`: 指定profiling结果目录
- `--tp`: 手动指定TP度数（当不使用profiling时）
- `--runs`: 每个场景的运行次数

### 配置文件关键参数

```yaml
# 在config.yaml中可以调整的关键参数
scheduling:
  # 基础长度阈值（会被profiling结果优化）
  length_thresholds:
    short: 5000
    medium: 10000  
    long: 15000
  
  # 路由规则
  routing_rules:
    short: [1, 2]    # 短序列优先使用TP=1或TP=2
    medium: [2, 4]   # 中等序列优先使用TP=2或TP=4
    long: [4]        # 长序列优先使用TP=4

benchmark:
  max_concurrent_requests: 32  # 并发请求数
  generation:
    max_new_tokens: 20000      # 最大生成token数
```

## 最佳实践

### 1. Profiling阶段建议
- 使用足够大的数据集样本（建议≥100个问题）
- 选择合适的TP度数进行推理（通常TP=4是较好的平衡点）
- 控制并发数量避免OOM错误

### 2. 测试配置建议
- 同构测试：重点关注主要序列长度类别的TP配置
- 异构测试：确保GPU资源分配合理，避免资源浪费
- 运行多次取平均值以获得稳定结果

### 3. 结果分析建议
- 对比同构和异构配置的整体性能提升
- 分析不同长度序列的处理效率差异
- 评估资源利用率和成本效益

## 故障排除

### 常见问题

1. **内存不足错误**
   ```
   解决方案：减少max_concurrent_requests参数值
   ```

2. **GPU分配冲突**
   ```
   解决方案：检查GPU可用性，调整--gpus参数
   ```

3. **Profiling结果加载失败**
   ```
   解决方案：确保profiling阶段成功完成，检查输出目录结构
   ```

### 日志调试
```bash
# 启用详细日志
--log-level DEBUG --log-file debug.log
```

## 扩展开发

### 添加新的自适应策略

在 `src/utils/profiling_loader.py` 中扩展 `create_adaptive_scenarios_from_profiling` 函数：

```python
def create_custom_adaptive_strategy(profiling_dir: str, strategy_name: str):
    # 实现自定义的场景生成逻辑
    pass
```

### 自定义路由规则

修改 `config/scheduling/routing_rules` 或在运行时动态设置。

## 性能基准

典型的性能提升预期：
- **吞吐量**: 异构配置相比同构配置提升20-40%
- **延迟**: 针对不同长度序列优化，整体延迟降低15-30%
- **资源利用率**: GPU利用率提升至85%以上

---

*本文档最后更新: 2026年2月*