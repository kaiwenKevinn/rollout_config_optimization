### SCOOT: SLO-Oriented Performance Tuning for LLM Inference Engines

This is the implementations of the WWW2025 oral paper [SCOOT: SLO-Oriented Performance Tuning for LLM Inference Engines](https://arxiv.org/abs/2408.04323) 

![](SCOOT.jpg)

### Overview

SCOOT is an automatic performance tuning system to optimize SLOs for each LLM inference service by tuning the parameters of the inference engine. It jointly exploits single-objective and multiple-objective Bayesian optimization techniques to handle various optimization objectives via exploration and exploitation. Moreover, SCOOT prunes the searchb space with known constraints and adopts a random forest to learn hidden constraints during the tuning process to mitigate invalid exploration. It can improve the performance of the LLM inference engine efficiently.

### Quick Start

`bo_scoot.py` is the script invovling the whole pipeline.

The shell script `tune_entry.sh` is used to reproduce the main results in the paper.

The python scripts in the directory `clients` are forked form vllm, involving `api_server.py`, `backend_request_func.py` and `benchmark_serving.py`, which are used to initialize server, client and benchmarking requsting, respectively.

Also, we implement modules of handling hidden and hard constraints in the BO search based on HEBO, which is in `hebo` directory. Specifically, the modules of handling hidden and hard constraints are incorporated in acquisition functions and the optimizers, i.e., `/hebo/acquisitions/acq.py` and `/hebo/optimizers/util.py`.

### NPU 910B Support

This project has been adapted to run on Huawei Ascend 910B NPU devices. Key modifications include:

- Updated `requirements.txt` for NPU environment
- Modified scripts to detect and use NPU devices
- Added NPU-specific environment variable handling
- Created containerization support for easy deployment

### Container Deployment

For containerized deployment on NPU 910B:

1. **Build container image:**
   ```bash
   ./docker_deploy.sh build
   ```

2. **Run container:**
   ```bash
   ./docker_deploy.sh run container-name /path/to/models /path/to/datasets
   ```

3. **Initialize environment in container:**
   ```bash
   ./docker_deploy.sh exec "bash /workspace/init_container_env.sh"
   ```

See [CONTAINER_DEPLOYMENT.md](CONTAINER_DEPLOYMENT.md) for detailed container deployment instructions.

### NPU Usage

For direct NPU usage (without container):

1. **Check NPU support:**
   ```bash
   python check_vllm_npu_support.py
   ```

2. **Run configuration enumeration:**
   ```bash
   bash run_entry_enum_configs.sh \
       /path/to/model \
       model_name \
       /path/to/dataset.json \
       dataset_name \
       request_rate \
       num_requests \
       npu_count \
       910B \
       --auto_enum \
       max_configs
   ```

See [NPU_USAGE_GUIDE.md](NPU_USAGE_GUIDE.md) for detailed NPU usage instructions.

### Citation
```latex
@inproceedings{cheng2025scoot,
  title={SCOOT: SLO-Oriented Performance Tuning for LLM Inference Engines},
  author={Cheng, Ke and Wang, Zhi and Hu, Wen and Yang, Tiannuo and Li, Jianguo and Zhang, Sheng},
  booktitle={Proceedings of the ACM on Web Conference 2025},
  pages={829--839},
  year={2025},
  publisher={Association for Computing Machinery}
}
```


