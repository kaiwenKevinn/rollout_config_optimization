#!/usr/bin/env python3
"""
vLLM NPU支持快速检查脚本
"""

import sys
import subprocess
import os

def run_command(cmd, description):
    """执行命令并返回结果"""
    print(f"\n🔍 {description}")
    print(f"执行: {cmd}")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ 成功")
            if result.stdout.strip():
                print(f"输出: {result.stdout.strip()}")
            return True, result.stdout
        else:
            print("❌ 失败")
            if result.stderr.strip():
                print(f"错误: {result.stderr.strip()}")
            return False, result.stderr
    except Exception as e:
        print(f"❌ 执行异常: {e}")
        return False, str(e)

def check_python_packages():
    """检查Python包"""
    print("\n=== Python环境检查 ===")
    
    packages = [
        ("torch", "PyTorch"),
        ("vllm", "vLLM"),
        ("torch_npu", "PyTorch NPU扩展")
    ]
    
    results = {}
    for pkg, name in packages:
        try:
            if pkg == "torch":
                import torch
                version = torch.__version__
                npu_support = hasattr(torch, 'npu')
                print(f"✅ {name}: {version} (NPU支持: {'是' if npu_support else '否'})")
                results[pkg] = {"version": version, "npu_support": npu_support}
            elif pkg == "vllm":
                import vllm
                version = vllm.__version__
                print(f"✅ {name}: {version}")
                results[pkg] = {"version": version}
            elif pkg == "torch_npu":
                import torch_npu
                version = getattr(torch_npu, '__version__', 'unknown')
                print(f"✅ {name}: {version}")
                results[pkg] = {"version": version}
        except ImportError as e:
            print(f"❌ {name}: 未安装 ({e})")
            results[pkg] = {"error": str(e)}
    
    return results

def check_vllm_npu_modules():
    """检查vLLM NPU模块"""
    print("\n=== vLLM NPU模块检查 ===")
    
    modules_to_check = [
        "vllm.worker.ascend_worker",
        "vllm.executor.ascend_executor",
        "vllm.model_executor.models.ascend",
    ]
    
    found_modules = []
    missing_modules = []
    
    for module in modules_to_check:
        try:
            __import__(module)
            print(f"✅ 找到: {module}")
            found_modules.append(module)
        except ImportError:
            print(f"❌ 缺少: {module}")
            missing_modules.append(module)
    
    return found_modules, missing_modules

def check_ascend_environment():
    """检查Ascend环境"""
    print("\n=== Ascend环境检查 ===")
    
    # 检查环境变量
    env_vars = ["ASCEND_HOME", "ASCEND_VISIBLE_DEVICES", "DEVICE_TYPE"]
    for var in env_vars:
        value = os.environ.get(var, "未设置")
        print(f"{var}: {value}")
    
    # 检查关键路径
    ascend_home = os.environ.get("ASCEND_HOME")
    if ascend_home:
        lib_path = os.path.join(ascend_home, "lib64")
        bin_path = os.path.join(ascend_home, "bin")
        
        if os.path.exists(lib_path):
            print(f"✅ ASCEND lib64存在: {lib_path}")
        else:
            print(f"❌ ASCEND lib64不存在: {lib_path}")
            
        if os.path.exists(bin_path):
            print(f"✅ ASCEND bin存在: {bin_path}")
        else:
            print(f"❌ ASCEND bin不存在: {bin_path}")
    
    # 检查npu-smi命令
    success, output = run_command("which npu-smi", "查找npu-smi命令")
    if success:
        run_command("npu-smi info", "获取NPU设备信息")

def main():
    print("🚀 vLLM NPU支持全面检查")
    print("=" * 50)
    
    # 1. Python包检查
    pkg_results = check_python_packages()
    
    # 2. Ascend环境检查
    check_ascend_environment()
    
    # 3. vLLM模块检查
    found_modules, missing_modules = check_vllm_npu_modules()
    
    # 4. 综合评估
    print("\n" + "=" * 50)
    print("📊 综合评估结果:")
    
    # 检查关键组件
    torch_ok = pkg_results.get("torch", {}).get("npu_support", False)
    vllm_ok = "vllm" in pkg_results and "error" not in pkg_results["vllm"]
    npu_modules_ok = len(found_modules) > 0
    
    if torch_ok and vllm_ok and npu_modules_ok:
        print("🎉 vLLM NPU支持配置完整！")
        print("   - PyTorch NPU支持: ✓")
        print("   - vLLM已安装: ✓") 
        print(f"   - NPU模块数量: {len(found_modules)}")
        return True
    else:
        print("❌ vLLM NPU支持不完整:")
        if not torch_ok:
            print("   - 缺少PyTorch NPU支持")
        if not vllm_ok:
            print("   - 缺少vLLM或安装有问题")
        if not npu_modules_ok:
            print("   - 缺少vLLM NPU模块")
        
        print("\n🔧 建议解决方案:")
        print("1. 安装vllm-ascend:")
        print("   git clone https://gitee.com/ascend/vLLM.git")
        print("   cd vLLM && pip install -e .")
        print("2. 确保Ascend CANN环境正确配置")
        print("3. 验证PyTorch NPU支持")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)