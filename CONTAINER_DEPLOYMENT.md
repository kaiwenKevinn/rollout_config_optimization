# NPU 910B 容器化部署指南 (EulerOS 2.0版本)

## 概述

本文档介绍如何在EulerOS 2.0环境中部署和运行NPU 910B配置优化项目。

## 目录结构

```
rollout_config_optimization/
├── requirements.txt           # NPU环境依赖包
├── Dockerfile.npu            # EulerOS NPU容器构建文件
├── docker_deploy.sh          # 容器部署脚本(EulerOS优化版)
├── init_container_env.sh     # 容器环境初始化脚本(EulerOS版)
├── check_vllm_npu_support.py # NPU支持检查脚本
└── CONTAINER_DEPLOYMENT.md   # 本文件
```

## 环境要求

### 主机环境
- EulerOS 2.0 SP1 或更高版本
- Docker 18.09 或更高版本
- 华为Ascend 910B NPU设备及驱动
- Ascend CANN工具包

### 检查环境
```bash
# 检查操作系统
cat /etc/os-release

# 检查Docker版本
docker --version

# 检查NPU驱动
npu-smi info

# 检查Ascend CANN环境变量
echo $ASCEND_HOME
```

## 部署步骤

### 1. 环境准备

确保主机环境满足以下要求：

```bash
# 验证EulerOS版本
grep -i euler /etc/os-release

# 检查NPU设备
npu-smi info

# 验证Ascend CANN安装
ls -la $ASCEND_HOME/bin/npu-smi
```

### 2. 构建容器镜像

```bash
cd /mnt/data2/kwchen/rollout_config_optimization

# 构建EulerOS NPU容器镜像
./docker_deploy.sh build
```

### 3. 运行容器

#### 基本运行
```bash
# 运行容器（默认名称）
./docker_deploy.sh run

# 运行容器（指定名称和数据路径）
./docker_deploy.sh run my-npu-container /path/to/models /path/to/datasets
```

#### 高级运行选项
```bash
# 运行容器并指定额外参数
./docker_deploy.sh run my-container \
    /models/path \
    /datasets/path \
    "--shm-size=16g --ulimit memlock=-1"
```

### 4. 初始化容器环境

进入容器后初始化环境：

```bash
# 进入容器
./docker_deploy.sh exec bash

# 在容器内执行初始化
/workspace/init_container_env.sh
```

或者一步到位：
```bash
./docker_deploy.sh exec "bash /workspace/init_container_env.sh"
```

### 5. 验证环境

```bash
# 检查NPU支持
./docker_deploy.sh exec "python3 /workspace/check_vllm_npu_support.py"

# 查看容器日志
./docker_deploy.sh logs
```

## EulerOS特定配置

### 包管理差异
- 使用 `yum` 而非 `apt-get`
- Python命令为 `python3` 而非 `python`
- 系统库路径可能有所不同

### 设备访问
EulerOS环境下NPU设备路径可能为：
- `/dev/davinci*` (标准路径)
- `/dev/ascend/davinci*` (某些配置)

脚本已自动处理这两种情况。

## 容器使用

### 在容器中运行配置优化

```bash
# 进入容器
./docker_deploy.sh exec bash

# 运行配置枚举
cd /workspace
bash run_entry_enum_configs.sh \
    /models/Qwen3-32B \
    Qwen3_32B \
    /datasets/test.json \
    test_dataset \
    29 \
    29 \
    4 \
    910B \
    --auto_enum \
    30
```

### 容器管理命令

```bash
# 停止容器
./docker_deploy.sh stop [container_name]

# 查看运行中的容器
docker ps

# 查看所有容器
docker ps -a

# 删除容器
docker rm container_name

# 删除镜像
docker rmi npu-rollout-config:euleros
```

## 环境变量配置

容器内重要的环境变量：

```bash
# Ascend CANN路径
ASCEND_HOME=/usr/local/Ascend

# Python路径
PYTHONPATH=/workspace:/usr/local/Ascend/python/site-packages

# 库路径
LD_LIBRARY_PATH=/usr/local/Ascend/lib64

# EulerOS特定编码
LANG=en_US.UTF-8
LC_ALL=en_US.UTF-8
PYTHONIOENCODING=utf-8
```

## 故障排除

### EulerOS常见问题

1. **包管理器问题**
   ```bash
   # 更新yum缓存
   yum clean all && yum makecache
   
   # 安装EPEL源（如果需要）
   yum install -y epel-release
   ```

2. **Python版本问题**
   ```bash
   # 确保使用Python3
   which python3
   python3 --version
   
   # 检查pip3
   pip3 --version
   ```

3. **字符编码问题**
   ```bash
   # 设置正确的语言环境
   export LANG=en_US.UTF-8
   export LC_ALL=en_US.UTF-8
   locale
   ```

4. **NPU设备访问问题**
   ```bash
   # 检查设备权限
   ls -la /dev/davinci*
   ls -la /dev/ascend/
   
   # 检查设备映射
   docker inspect container_name | grep -A 10 Devices
   ```

### 日志查看

```bash
# 实时查看容器日志
./docker_deploy.sh logs

# 查看特定时间段的日志
docker logs --since 1h container_name

# 导出日志
docker logs container_name > container.log
```

## 性能优化建议

### EulerOS容器资源配置
```bash
# 推荐的Docker运行参数
docker run \
    --shm-size=16g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --cpuset-cpus=0-15 \
    --memory=64g \
    [其他参数]
```

### NPU资源管理
```bash
# 查看NPU使用情况
npu-smi info

# 设置可见设备
export ASCEND_VISIBLE_DEVICES=0,1,2,3
```

## 备份与恢复

### 备份容器数据
```bash
# 备份重要数据
docker cp container_name:/workspace/tune_res ./backup/

# 备份容器状态
docker commit container_name npu-rollout-config:euleros-backup
```

### 恢复运行
```bash
# 从备份镜像启动
docker run -it npu-rollout-config:euleros-backup
```

## 安全注意事项

1. 使用`--user`参数运行非root用户
2. 限制容器的capabilities
3. 定期更新基础镜像
4. 监控容器资源使用情况

## 参考资料

- [华为Ascend CANN文档](https://www.huawei.com/)
- [vLLM-Ascend GitHub](https://gitee.com/ascend/vLLM)
- [OpenEuler官方文档](https://openeuler.org/)
- [Docker官方文档](https://docs.docker.com/)