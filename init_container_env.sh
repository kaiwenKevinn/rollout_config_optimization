#!/bin/bash
# NPU容器环境初始化脚本 (EulerOS 2.0版本)

set -e

echo "=== NPU容器环境初始化 (EulerOS) ==="

# 检查是否在容器内运行
if [ ! -f /.dockerenv ]; then
    echo "警告: 此脚本应在容器内运行"
    # exit 1
fi

# 检查是否为EulerOS环境
if [[ -f /etc/os-release ]]; then
    . /etc/os-release
    if [[ "$ID" == "openeuler" ]] || [[ "$NAME" == *"EulerOS"* ]]; then
        echo "✓ 检测到EulerOS环境: $NAME $VERSION"
    else
        echo "⚠ 非EulerOS环境: $NAME $VERSION"
    fi
fi

# 安装系统依赖（EulerOS使用yum）
yum update -y
yum install -y \
    mesa-libGL \
    libSM \
    libXext \
    libXrender \
    glibc-static \
    && yum clean all

# 设置Ascend环境变量
export ASCEND_HOME=/usr/local/Ascend
export PATH=$ASCEND_HOME/bin:$PATH
export LD_LIBRARY_PATH=$ASCEND_HOME/lib64:$LD_LIBRARY_PATH
export PYTHONPATH=$ASCEND_HOME/python/site-packages:$PYTHONPATH

# EulerOS特定环境变量
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8
export PYTHONIOENCODING=utf-8

# 验证Ascend安装
if [ -d "$ASCEND_HOME" ]; then
    echo "✓ Ascend CANN已安装"
    echo "版本信息:"
    if [ -f "$ASCEND_HOME/version.info" ]; then
        cat $ASCEND_HOME/version.info
    fi
else
    echo "⚠ Ascend CANN未找到，请确保正确挂载"
fi

# 检查NPU设备
echo "检查NPU设备状态:"
if command -v npu-smi &> /dev/null; then
    npu-smi info
else
    echo "⚠ npu-smi命令不可用"
fi

# 安装Python依赖
echo "安装Python依赖..."
python3 -m pip install --upgrade pip setuptools wheel

# 安装NPU特定的PyTorch（如果wheel文件存在）
if [ -f "/workspace/torch_npu*.whl" ]; then
    echo "安装本地NPU PyTorch..."
    pip3 install /workspace/torch_npu*.whl
elif [ -f "/tmp/torch_npu*.whl" ]; then
    echo "安装临时目录中的NPU PyTorch..."
    pip3 install /tmp/torch_npu*.whl
else
    echo "⚠ 未找到NPU PyTorch wheel文件"
    echo "请从华为官方渠道下载并安装"
fi

# 安装其他Python包
cd /workspace
if [ -f "requirements.txt" ]; then
    echo "安装项目依赖..."
    pip3 install -r requirements.txt --no-cache-dir
fi

# 编译安装vllm-ascend（如果源码存在）
if [ -d "/workspace/vLLM" ] || [ -d "/tmp/vLLM" ]; then
    VLLM_DIR=""
    if [ -d "/workspace/vLLM" ]; then
        VLLM_DIR="/workspace/vLLM"
    else
        VLLM_DIR="/tmp/vLLM"
    fi
    
    echo "编译安装vllm-ascend..."
    cd $VLLM_DIR
    pip3 install -e . --no-cache-dir
    cd /workspace
else
    echo "⚠ 未找到vLLM源码目录"
    echo "请克隆Ascend版本的vLLM仓库"
fi

# 验证安装
echo "=== 验证安装 ==="
python3 /workspace/check_vllm_npu_support.py

echo "=== 环境初始化完成 ==="
echo "现在可以在EulerOS容器中运行NPU相关任务了"