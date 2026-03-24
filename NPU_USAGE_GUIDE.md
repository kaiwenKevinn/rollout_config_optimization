# NPU 910B 使用指南

本文档说明如何在华为昇腾NPU 910B环境下使用配置枚举脚本。

## 环境准备

### 1. 硬件要求
- 华为昇腾910B NPU设备
- 支持的服务器平台

### 2. 软件依赖
```bash
# 安装Ascend CANN工具包
# 请参考华为官方文档安装对应版本的CANN

# 安装NPU版本的PyTorch
pip install torch==2.1.0+ascend -f https://developer.huaweicloud.com/repo/ascend-pytorch/

# 安装NPU版本的其他依赖
pip install torch-npu
```

### 3. 环境变量设置
```bash
# 设置NPU相关环境变量（通常由CANN自动设置）
export ASCEND_HOME=/usr/local/Ascend
export PATH=$ASCEND_HOME/bin:$PATH
export LD_LIBRARY_PATH=$ASCEND_HOME/lib64:$LD_LIBRARY_PATH
export PYTHONPATH=$ASCEND_HOME/python/site-packages:$PYTHONPATH
```

## 使用方法

### 基本用法
```bash
# 自动枚举配置（NPU 910B环境）
bash run_entry_enum_configs.sh \
    /path/to/model \
    Qwen3_32B \
    /path/to/dataset.json \
    dataset_name \
    29 \        # request_rate
    29 \        # num_requests  
    4 \         # npu_num (NPU设备数量)
    910B \      # npu_type (NPU型号)
    --auto_enum \
    30          # max_configs (最大配置数)
```

### 实际运行示例
```bash
# 后台运行示例
nohup bash run_entry_enum_configs.sh \
    /research/d1/gds/ytyang/kwchen/hf_models/Qwen/Qwen3-32B/models--Qwen--Qwen3-32B/snapshots/9216db5781bf21249d130ec9da846c4624c16137 \
    Qwen3_32B \
    /research/d1/gds/ytyang/kwchen/hetero_rollout/results/gpqa_test/classification_by_range/sequence_profile_bucket_0.json \
    gpqa_diamond_bucket_0 \
    29 \
    29 \
    4 \
    910B \
    --auto_enum \
    30 \
    > run_npu_qwen3_32b.log 2>&1 &
```

### 断点续测
```bash
# 自动续测（使用最新结果目录）
bash run_entry_enum_configs.sh ... 4 910B --auto_enum 30 resume

# 指定目录续测
bash run_entry_enum_configs.sh ... 4 910B --auto_enum 30 resume ./tune_res/enum_results/Qwen3_32B_gpqa_diamond_bucket_0_20260304_232428
```

## 参数说明

| 参数 | 说明 | 示例 |
|------|------|------|
| model_path | 模型路径 | `/path/to/model` |
| model_name | 模型名称 | `Qwen3_32B` |
| dataset_path | 数据集路径 | `/path/to/dataset.json` |
| dataset_name | 数据集名称 | `gpqa_diamond` |
| request_rate | 请求速率 | `29` |
| num_requests | 请求总数 | `29` |
| npu_num | NPU设备数量 | `4` |
| npu_type | NPU型号 | `910B` |
| config_mode | 配置模式 | `--auto_enum` 或配置文件路径 |
| max_configs | 最大配置数 | `30` |

## 性能调优建议

### 1. 内存管理
```bash
# 设置NPU内存预留比例
export ASCEND_GLOBAL_LOG_LEVEL=3
export ASCEND_SLOG_PRINT_TO_STDOUT=0
```

### 2. 并行策略
针对NPU 910B的特点，建议的并行配置：
- TP (Tensor Parallel): 2, 4, 8
- PP (Pipeline Parallel): 1, 2, 4
- 根据实际显存大小调整max_num_seqs

### 3. 常见问题解决

**问题1：内存不足**
```bash
# 减少batch size相关参数
# 在脚本中调整TUNE_PARAMS或直接修改枚举范围
```

**问题2：设备检测失败**
```bash
# 检查NPU驱动状态
npu-smi info

# 重新安装驱动
# 参考华为官方文档
```

**问题3：性能不佳**
```bash
# 尝试不同的block_size值
# 调整scheduler_delay_factor参数
```

## 结果分析

运行完成后，结果保存在：
- `./tune_res/enum_results/{model}_{dataset}_{timestamp}/enum_results.csv`
- `./tune_res/enum_results/{model}_{dataset}_{timestamp}/enum_results.json`

可以通过以下方式查看最佳配置：
```python
import pandas as pd
df = pd.read_csv('enum_results.csv')
best_config = df.loc[df['request_throughput'].idxmax()]
print(best_config)
```

## 注意事项

1. 确保NPU驱动和CANN版本兼容
2. 监控NPU温度和功耗
3. 根据实际硬件配置调整参数范围
4. 建议先用小规模测试验证环境配置