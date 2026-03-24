#!/bin/bash
# NPU环境测试脚本

echo "=== NPU环境配置测试 ==="

# 1. 检查NPU设备状态
echo "1. 检查NPU设备:"
if command -v npu-smi &> /dev/null; then
    echo "✓ npu-smi 命令可用"
    echo "NPU设备信息:"
    npu-smi info
    npu_count=$(npu-smi info | grep -c "Device ID" 2>/dev/null || echo "0")
    echo "检测到 $npu_count 个NPU设备"
else
    echo "✗ 未找到 npu-smi 命令"
    echo "请确认已安装Ascend驱动和工具包"
fi

echo ""

# 2. 检查Python环境
echo "2. 检查Python环境:"
python_version=$(python --version 2>&1)
echo "Python版本: $python_version"

# 检查PyTorch
if python -c "import torch; print('✓ PyTorch版本:', torch.__version__)" 2>/dev/null; then
    # 检查是否有NPU支持
    if python -c "import torch; hasattr(torch, 'npu')" 2>/dev/null; then
        echo "✓ 检测到PyTorch NPU支持"
        npu_devices=$(python -c "import torch; print(torch.npu.device_count())" 2>/dev/null || echo "0")
        echo "可用NPU设备数: $npu_devices"
    else
        echo "⚠ PyTorch未启用NPU支持"
    fi
else
    echo "✗ 未安装PyTorch"
fi

echo ""

# 3. 检查必要的环境变量
echo "3. 环境变量检查:"
echo "ASCEND_HOME: ${ASCEND_HOME:-未设置}"
echo "PATH中包含Ascend: $(echo $PATH | grep -q ascend && echo '是' || echo '否')"
echo "LD_LIBRARY_PATH中包含Ascend: $(echo $LD_LIBRARY_PATH | grep -q ascend && echo '是' || echo '否')"

echo ""

# 4. 测试脚本语法
echo "4. 脚本语法检查:"
if bash -n run_entry_enum_configs.sh; then
    echo "✓ run_entry_enum_configs.sh 语法正确"
else
    echo "✗ run_entry_enum_configs.sh 存在语法错误"
fi

echo ""
echo "=== 测试完成 ==="
echo "如需运行实际测试，请使用:"
echo "bash run_entry_enum_configs.sh <model_path> <model_name> <dataset_path> <dataset_name> <request_rate> <request_num> <npu_num> <npu_type> --auto_enum <max_configs>"