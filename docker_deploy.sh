#!/bin/bash
# NPU 910B 容器化部署脚本 (EulerOS 2.0版本)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_NAME="rollout-config-npu"
IMAGE_NAME="npu-rollout-config:euleros"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查前提条件
check_prerequisites() {
    log_info "检查前提条件..."
    
    # 检查Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker未安装，请先安装Docker"
        exit 1
    fi
    
    # 检查操作系统
    if [[ -f /etc/os-release ]]; then
        . /etc/os-release
        if [[ "$ID" == "euleros" ]] || [[ "$ID_LIKE" == *"euleros"* ]]; then
            log_info "检测到EulerOS环境"
        else
            log_warn "非EulerOS环境，可能存在兼容性问题"
        fi
    fi
    
    # 检查NPU驱动
    if ! command -v npu-smi &> /dev/null; then
        log_warn "未检测到npu-smi命令，NPU驱动可能未正确安装"
    else
        log_info "NPU驱动状态:"
        npu-smi info
    fi
    
    # 检查Ascend CANN
    if [ -z "$ASCEND_HOME" ]; then
        log_warn "ASCEND_HOME环境变量未设置"
    else
        log_info "ASCEND_HOME: $ASCEND_HOME"
    fi
}

# 构建Docker镜像
build_image() {
    log_info "构建EulerOS Docker镜像..."
    
    if [ ! -f "Dockerfile.npu" ]; then
        log_error "Dockerfile.npu 文件不存在"
        exit 1
    fi
    
    # 检查是否为EulerOS基础镜像
    if ! grep -q "openeuler" Dockerfile.npu; then
        log_warn "Dockerfile可能不是基于EulerOS，请确认"
    fi
    
    docker build -f Dockerfile.npu -t $IMAGE_NAME .
    log_info "镜像构建完成: $IMAGE_NAME"
}

# 运行容器
run_container() {
    local container_name=${1:-$PROJECT_NAME}
    local model_path=${2}
    local dataset_path=${3}
    local extra_args=${4:-""}
    
    log_info "启动EulerOS容器: $container_name"
    
    # 基础运行参数
    local docker_args="-d --name $container_name --privileged"
    
    # NPU设备映射（EulerOS环境）
    docker_args+=" --device /dev/davinci_manager"
    docker_args+=" --device /dev/devmm_svm"
    docker_args+=" --device /dev/hisi_hdc"
    
    # 显存设备（EulerOS可能有不同的设备命名）
    for i in {0..7}; do
        if [ -e "/dev/davinci$i" ]; then
            docker_args+=" --device /dev/davinci$i"
        fi
        # EulerOS可能使用不同的设备路径
        if [ -e "/dev/ascend/davinci$i" ]; then
            docker_args+=" --device /dev/ascend/davinci$i"
        fi
    done
    
    # IPC和网络
    docker_args+=" --ipc=host"
    docker_args+=" --network=host"
    
    # EulerOS特定挂载
    docker_args+=" -v /etc/localtime:/etc/localtime:ro"
    docker_args+=" -v /lib/modules:/lib/modules:ro"
    
    # 环境变量
    docker_args+=" -e ASCEND_HOME=/usr/local/Ascend"
    docker_args+=" -e PYTHONPATH=/workspace"
    docker_args+=" -e LANG=en_US.UTF-8"
    docker_args+=" -e LC_ALL=en_US.UTF-8"
    
    # 挂载卷
    docker_args+=" -v $SCRIPT_DIR:/workspace"
    docker_args+=" -v /usr/local/Ascend:/usr/local/Ascend:ro"
    docker_args+=" -v /var/log/npu:/var/log/npu"
    
    # 挂载模型和数据（如果提供）
    if [ -n "$model_path" ] && [ -d "$model_path" ]; then
        docker_args+=" -v $model_path:/models:ro"
    fi
    
    if [ -n "$dataset_path" ] && [ -d "$dataset_path" ]; then
        docker_args+=" -v $dataset_path:/datasets:ro"
    fi
    
    # 执行命令
    local cmd="docker run $docker_args $extra_args $IMAGE_NAME"
    log_info "执行命令: $cmd"
    
    eval $cmd
    
    # 等待容器启动
    sleep 5
    
    # 检查容器状态
    if docker ps | grep -q $container_name; then
        log_info "容器启动成功"
        docker logs $container_name
    else
        log_error "容器启动失败"
        docker logs $container_name
        exit 1
    fi
}

# 在容器中执行命令
exec_in_container() {
    local container_name=${1:-$PROJECT_NAME}
    local command=${2}
    
    if [ -z "$command" ]; then
        log_error "请提供要执行的命令"
        exit 1
    fi
    
    log_info "在容器 $container_name 中执行: $command"
    docker exec -it $container_name $command
}

# 停止并删除容器
stop_container() {
    local container_name=${1:-$PROJECT_NAME}
    
    log_info "停止容器: $container_name"
    
    if docker ps -a | grep -q $container_name; then
        docker stop $container_name 2>/dev/null || true
        docker rm $container_name 2>/dev/null || true
        log_info "容器已停止并删除"
    else
        log_warn "容器 $container_name 不存在"
    fi
}

# 显示容器日志
show_logs() {
    local container_name=${1:-$PROJECT_NAME}
    docker logs -f $container_name
}

# 主菜单
show_help() {
    echo "NPU 910B 容器化部署工具"
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  build              构建Docker镜像"
    echo "  run [name] [model_path] [dataset_path] [extra_docker_args]"
    echo "                     运行容器"
    echo "  exec [name] [command]"
    echo "                     在容器中执行命令"
    echo "  stop [name]        停止容器"
    echo "  logs [name]        查看容器日志"
    echo "  check              检查环境"
    echo "  help               显示帮助信息"
    echo ""
    echo "示例:"
    echo "  $0 build"
    echo "  $0 run my-container /path/to/models /path/to/datasets"
    echo "  $0 exec my-container 'python check_vllm_npu_support.py'"
    echo "  $0 stop my-container"
}

# 主程序
main() {
    case "${1:-help}" in
        "build")
            check_prerequisites
            build_image
            ;;
        "run")
            check_prerequisites
            run_container "$2" "$3" "$4" "$5"
            ;;
        "exec")
            exec_in_container "$2" "$3"
            ;;
        "stop")
            stop_container "$2"
            ;;
        "logs")
            show_logs "$2"
            ;;
        "check")
            check_prerequisites
            ;;
        "help"|*)
            show_help
            ;;
    esac
}

main "$@"