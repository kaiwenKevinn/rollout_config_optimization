"""
vLLM Instance Module

Encapsulates a single vLLM server instance with management capabilities.
"""

import os
import sys
import json
import time
import logging
import asyncio
import subprocess
from typing import List, Dict, Any, Optional, AsyncGenerator
from dataclasses import dataclass, field
from enum import Enum

import aiohttp
import httpx

logger = logging.getLogger(__name__)


class InstanceState(Enum):
    STOPPED = "stopped"
    STARTING = "starting"
    READY = "ready"
    BUSY = "busy"
    ERROR = "error"
    STOPPING = "stopping"


@dataclass
class InstanceConfig:
    instance_id: str
    tp_degree: int
    gpu_ids: List[int]
    port: int
    host: str = "127.0.0.1"
    model_name: str = "Qwen/Qwen3-32B"
    max_model_len: int = 8192
    dtype: str = "auto"
    trust_remote_code: bool = True
    gpu_memory_utilization: float = 0.90
    api_type: str = "openai"
    enforce_eager: bool = True  # Force eager mode to avoid context length issues
    extra_args: Dict[str, Any] = field(default_factory=dict)
    log_dir: Optional[str] = None  # Directory to save vLLM logs
    
    def to_cmd_args(self) -> List[str]:
        args = [
            "--model", self.model_name,
            "--tensor-parallel-size", str(self.tp_degree),
            "--port", str(self.port),
            "--host", self.host,
            "--max-model-len", str(self.max_model_len),
            "--dtype", self.dtype,
            "--gpu-memory-utilization", str(self.gpu_memory_utilization),
        ]
        # Check if vLLM version is 0.7.2 or newer and if using Qwen2.5/Qwen3
        # In newer vLLM versions, trust-remote-code might be needed or handled differently
        # But for now we keep it as is unless specifically problematic
        if self.trust_remote_code:
            args.append("--trust-remote-code")
        if self.enforce_eager:
            args.append("--enforce-eager")
        for key, value in self.extra_args.items():
            if isinstance(value, bool):
                if value:
                    args.append(f"--{key}")
            else:
                args.extend([f"--{key}", str(value)])
        return args


@dataclass
class GenerationParams:
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    stop: Optional[List[str]] = None
    stream: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        params = {"max_tokens": self.max_tokens, "temperature": self.temperature,
                  "top_p": self.top_p, "stream": self.stream}
        if self.stop:
            params["stop"] = self.stop
        return params


@dataclass
class GenerationResult:
    request_id: str
    prompt: str
    generated_text: str
    input_tokens: int
    output_tokens: int
    finish_reason: str
    first_token_time: Optional[float] = None
    total_time: float = 0.0
    success: bool = True
    error_message: str = ""


class VLLMInstance:
    """Manages a single vLLM server instance."""
    
    def __init__(self, config: InstanceConfig):
        self.config = config
        self.instance_id = config.instance_id
        self.tp_degree = config.tp_degree
        self.gpu_ids = config.gpu_ids
        self.port = config.port
        self.host = config.host
        self._state = InstanceState.STOPPED
        self._process: Optional[subprocess.Popen] = None
        self._http_client: Optional[httpx.AsyncClient] = None
        self._total_requests = 0
        self._active_requests = 0
        self._failed_requests = 0
        self._base_url = f"http://{self.host}:{self.port}"
        self._health_url = f"{self._base_url}/health"
        self._completions_url = f"{self._base_url}/v1/completions"
        self._log_file: Optional[Any] = None  # File handle for vLLM logs
    
    @property
    def state(self) -> InstanceState:
        return self._state
    
    @property
    def is_ready(self) -> bool:
        return self._state == InstanceState.READY
    
    @property
    def active_requests(self) -> int:
        return self._active_requests
    
    @property
    def total_requests(self) -> int:
        return self._total_requests
    
    async def start(self, timeout: int = 300) -> bool:
        if self._state != InstanceState.STOPPED:
            logger.warning(f"Instance {self.instance_id} is not in STOPPED state, current: {self._state}")
            return False
        
        self._state = InstanceState.STARTING
        logger.info(f"Starting vLLM instance {self.instance_id} on port {self.port}")
        logger.info(f"Model: {self.config.model_name}, Max Model Len: {self.config.max_model_len}")
        
        try:
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in self.gpu_ids)
            # Note: vLLM 0.11.0 openai.api_server requires V1 engine (VLLM_USE_V1=1)
            # Do NOT set VLLM_USE_V1=0, it will cause AssertionError
            
            # Use python -m vllm.entrypoints.openai.api_server
            cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server"] + self.config.to_cmd_args()
            
            logger.info(f"Command: {' '.join(cmd)}")
            logger.info(f"CUDA_VISIBLE_DEVICES: {env['CUDA_VISIBLE_DEVICES']}")
            logger.info(f"Instance {self.instance_id} GPU assignment: {self.gpu_ids}")
            
            # Setup log file if log_dir is specified
            if self.config.log_dir:
                os.makedirs(self.config.log_dir, exist_ok=True)
                log_file_path = os.path.join(self.config.log_dir, f"vllm_{self.instance_id}.log")
                self._log_file = open(log_file_path, "w")
                logger.info(f"vLLM logs will be saved to: {log_file_path}")
                stdout_target = self._log_file
                stderr_target = subprocess.STDOUT  # Merge stderr into stdout
            else:
                stdout_target = subprocess.PIPE
                stderr_target = subprocess.PIPE
            
            self._process = subprocess.Popen(
                cmd, env=env, 
                stdout=stdout_target,
                stderr=stderr_target, 
                text=True
            )
            
            ready = await self._wait_for_ready(timeout)
            if ready:
                self._state = InstanceState.READY
                self._http_client = httpx.AsyncClient(
                    timeout=httpx.Timeout(
                        connect=30.0,      # 连接超时
                        read=6000.0,    # 读取超时
                        write=30.0,        # 写入超时
                        pool=30.0          # 连接池超时
                    )
                )
                logger.info(f"Instance {self.instance_id} is ready")
                return True
            else:
                self._state = InstanceState.ERROR
                # Capture stderr/stdout for debugging
                await self._log_process_output()
                await self.stop()
                return False
        except Exception as e:
            self._state = InstanceState.ERROR
            logger.error(f"Failed to start instance {self.instance_id}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    async def _log_process_output(self):
        """Log process stdout/stderr for debugging."""
        if self._process:
            try:
                # Non-blocking read of available output
                import select
                if self._process.stderr and select.select([self._process.stderr], [], [], 0)[0]:
                    stderr_output = self._process.stderr.read()
                    if stderr_output:
                        logger.error(f"vLLM stderr output:\n{stderr_output[:2000]}")
                if self._process.stdout and select.select([self._process.stdout], [], [], 0)[0]:
                    stdout_output = self._process.stdout.read()
                    if stdout_output:
                        logger.info(f"vLLM stdout output:\n{stdout_output[:2000]}")
            except Exception as e:
                logger.warning(f"Could not read process output: {e}")
    
    async def _wait_for_ready(self, timeout: int) -> bool:
        start_time = time.time()
        check_count = 0
        async with aiohttp.ClientSession() as session:
            while time.time() - start_time < timeout:
                check_count += 1
                try:
                    async with session.get(self._health_url, timeout=5) as response:
                        if response.status == 200:
                            logger.info(f"Health check passed after {check_count} attempts")
                            return True
                except Exception as e:
                    if check_count % 15 == 0:  # Log every 30 seconds
                        elapsed = time.time() - start_time
                        logger.info(f"Waiting for vLLM... ({elapsed:.0f}s elapsed, {check_count} checks)")
                
                # Check if process has died
                if self._process and self._process.poll() is not None:
                    exit_code = self._process.poll()
                    logger.error(f"vLLM process died with exit code: {exit_code}")
                    await self._log_process_output()
                    return False
                
                await asyncio.sleep(2.0)
        
        logger.error(f"vLLM startup timed out after {timeout}s")
        return False
    
    async def stop(self) -> None:
        if self._state == InstanceState.STOPPED:
            return
        self._state = InstanceState.STOPPING
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None
        if self._process:
            try:
                self._process.terminate()
                self._process.wait(timeout=10)
            except:
                self._process.kill()
                self._process.wait()
            self._process = None
        # Close log file if open
        if self._log_file:
            self._log_file.close()
            self._log_file = None
        self._state = InstanceState.STOPPED
    
    async def health_check(self) -> bool:
        if not self._http_client:
            return False
        try:
            response = await self._http_client.get(self._health_url)
            return response.status_code == 200
        except:
            return False
    
    async def generate(self, prompt: str, params: Optional[GenerationParams] = None,
                      request_id: Optional[str] = None) -> GenerationResult:
        # 记录收到请求的日志
        logger.info(f"Instance {self.instance_id} received generate request: {request_id}, prompt_len={len(prompt)}")
        
        if not self.is_ready:
            logger.warning(f"Instance {self.instance_id} is not ready (state: {self._state.value})")
            return GenerationResult(request_id=request_id or "", prompt=prompt,
                                   generated_text="", input_tokens=0, output_tokens=0,
                                   finish_reason="error", success=False,
                                   error_message="Instance not ready")
        
        params = params or GenerationParams()
        request_id = request_id or f"req_{time.time()}"
        self._active_requests += 1
        self._total_requests += 1
        
        start_time = time.time()
        first_token_time = None
        generated_text = ""
        output_tokens = 0
        
        try:
            if params.stream:
                async for chunk in self._stream_generate(prompt, params):
                    if first_token_time is None and chunk:
                        first_token_time = time.time()
                    generated_text += chunk
                    output_tokens += 1
            else:
                result = await self._non_stream_generate(prompt, params)
                generated_text = result.get("text", "")
                output_tokens = result.get("output_tokens", 0)
                first_token_time = start_time
            
            return GenerationResult(
                request_id=request_id, prompt=prompt, generated_text=generated_text,
                input_tokens=0, output_tokens=output_tokens, finish_reason="stop",
                first_token_time=first_token_time, total_time=time.time() - start_time,
                success=True
            )
        except Exception as e:
            self._failed_requests += 1
            return GenerationResult(
                request_id=request_id, prompt=prompt, generated_text="",
                input_tokens=0, output_tokens=0, finish_reason="error",
                total_time=time.time() - start_time, success=False, error_message=str(e)
            )
        finally:
            self._active_requests -= 1
    
    async def _stream_generate(self, prompt: str, params: GenerationParams) -> AsyncGenerator[str, None]:
        payload = {"model": self.config.model_name, "prompt": prompt, **params.to_dict()}
        async with self._http_client.stream("POST", self._completions_url, json=payload) as response:
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        if "choices" in chunk and chunk["choices"]:
                            text = chunk["choices"][0].get("text", "")
                            if text:
                                yield text
                    except json.JSONDecodeError:
                        continue
    
    async def _non_stream_generate(self, prompt: str, params: GenerationParams) -> Dict[str, Any]:
        params_dict = params.to_dict()
        params_dict["stream"] = False
        payload = {"model": self.config.model_name, "prompt": prompt, **params_dict}
        
        logger.debug(f"Request payload: model={self.config.model_name}, prompt_len={len(prompt)}, params={params_dict}")
        
        try:
            response = await self._http_client.post(self._completions_url, json=payload)
            
            if response.status_code != 200:
                error_text = response.text
                logger.error(f"vLLM API error: status={response.status_code}, response={error_text[:500]}")
                return {"text": "", "finish_reason": "error", "output_tokens": 0, "error": error_text}
            
            result = response.json()
            if "choices" in result and result["choices"]:
                choice = result["choices"][0]
                return {"text": choice.get("text", ""), "finish_reason": choice.get("finish_reason", "stop"),
                        "output_tokens": result.get("usage", {}).get("completion_tokens", 0)}
            return {"text": "", "finish_reason": "error", "output_tokens": 0}
        except httpx.TimeoutException as e:
            logger.error(f"Request timeout: {type(e).__name__} - {e}")
            return {"text": "", "finish_reason": "timeout", "output_tokens": 0, "error": f"Timeout: {e}"}
        except httpx.ConnectError as e:
            logger.error(f"Connection error: {type(e).__name__} - {e}")
            return {"text": "", "finish_reason": "error", "output_tokens": 0, "error": f"Connection error: {e}"}
        except Exception as e:
            import traceback
            logger.error(f"Request failed: {type(e).__name__}: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {"text": "", "finish_reason": "error", "output_tokens": 0, "error": f"{type(e).__name__}: {e}"}
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            "instance_id": self.instance_id, "tp_degree": self.tp_degree,
            "gpu_ids": self.gpu_ids, "port": self.port, "state": self._state.value,
            "total_requests": self._total_requests, "active_requests": self._active_requests,
            "failed_requests": self._failed_requests
        }
    
    def __repr__(self) -> str:
        return f"VLLMInstance(id={self.instance_id}, tp={self.tp_degree}, gpus={self.gpu_ids}, state={self._state.value})"


def create_instance_from_config(instance_config: Dict[str, Any], model_config: Dict[str, Any],
                                server_config: Dict[str, Any], port_offset: int = 0) -> VLLMInstance:
    config = InstanceConfig(
        instance_id=instance_config.get("instance_id", f"instance_{port_offset}"),
        tp_degree=instance_config.get("tp", 1),
        gpu_ids=instance_config.get("gpus", [0]),
        port=server_config.get("base_port", 8000) + port_offset,
        host=server_config.get("host", "127.0.0.1"),
        model_name=model_config.get("name", "Qwen/Qwen3-32B"),
        max_model_len=model_config.get("max_model_len", 8192),
        dtype=model_config.get("dtype", "auto"),
        trust_remote_code=model_config.get("trust_remote_code", True),
        gpu_memory_utilization=model_config.get("gpu_memory_utilization", 0.90),
        log_dir=server_config.get("log_dir", None)
    )
    return VLLMInstance(config)
