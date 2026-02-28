import json
import os
import sys
import time
import traceback
import asyncio
import aiohttp

from dataclasses import dataclass, field
from typing import List, Optional
from tqdm.asyncio import tqdm

# 流式推理可能很慢（高负载时），需足够长的 sock_read 避免单次 read 超时
AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=150000 * 60, sock_read=100000 * 60)

# Round-robin 负载均衡计数器
_rr_counter = 0

@dataclass
class RequestFuncInput:
    prompt: str
    api_url: str
    prompt_len: int
    output_len: int
    model: str
    best_of: int = 1
    use_beam_search: bool = False


@dataclass
class RequestFuncOutput:
    generated_text: str = ""
    success: bool = False
    latency: float = 0.0
    ttft: float = 0.0  # Time to first token
    itl: List[float] = field(
        default_factory=list)  # List of inter-token latencies
    prompt_len: int = 0
    gpu_hit_rate: List[float] = field(
        default_factory=list)  
    error: str = ""

def remove_prefix(text: str, prefix: str) -> str:
    if text.startswith(prefix):
        return text[len(prefix):]
    return text
async def fetch_stats(session, url):
    # 尝试多个可能的metrics端点
    metrics_endpoints = ['metrics', 'stats', 'monitoring']
    original_url = url
    
    for endpoint in metrics_endpoints:
        url = original_url.replace('generate', endpoint)
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status == 200:
                    try:
                        text_data = await response.text()
                        # 优先尝试解析 JSON 格式（clients.api_server 返回）
                        text_stripped = text_data.strip()
                        if text_stripped.startswith('{'):
                            data = json.loads(text_data)
                            return json.dumps({
                                "pending_queue_length": int(data.get("pending_queue_length", 0)),
                                "num_running": int(data.get("num_running", 0)),
                                "gpu_cache_usage": float(data.get("gpu_cache_usage", 0.0)),
                                "gpu_hit_rate": float(data.get("gpu_hit_rate", 0.0)),
                            })
                        # 否则尝试 Prometheus 文本格式
                        if 'gpu' in text_data.lower() or 'cache' in text_data.lower() or 'running' in text_data.lower():
                            lines = text_data.strip().split('\n')
                            metrics_dict = {}
                            for line in lines:
                                if line.startswith('#') or not line.strip():
                                    continue
                                if ' ' in line:
                                    parts = line.split(' ')
                                    if len(parts) >= 2:
                                        key, value = parts[0], parts[1]
                                        if any(k in key.lower() for k in ('gpu', 'cache', 'queue', 'running', 'waiting')):
                                            try:
                                                metrics_dict[key] = float(value)
                                            except Exception:
                                                pass
                            return json.dumps({
                                "pending_queue_length": int(metrics_dict.get('vllm:num_requests_waiting', metrics_dict.get('vllm:pending_queue_length', 0))),
                                "num_running": int(metrics_dict.get('vllm:num_requests_running', 0)),
                                "gpu_cache_usage": float(metrics_dict.get('vllm:gpu_cache_usage_perc', metrics_dict.get('vllm:gpu_cache_usage', 0.0))),
                                "gpu_hit_rate": float(metrics_dict.get('vllm:gpu_hit_rate', 0.0)),
                            })
                        return json.dumps({
                            "pending_queue_length": 0,
                            "num_running": 0,
                            "gpu_cache_usage": 0.0,
                            "gpu_hit_rate": 0.0,
                        })
                    except:
                        # 解析失败，返回默认值
                        return json.dumps({
                            "pending_queue_length": 0,
                            "num_running": 0,
                            "gpu_cache_usage": 0.0,
                            "gpu_hit_rate": 0.0
                        })
                # 如果状态码不是200，继续尝试下一个端点
        except Exception as e:
            # 继续尝试下一个端点
            continue
    
    # 所有端点都失败了，返回默认值
    if len(metrics_endpoints) > 1:  # 只在尝试了多个端点后才打印警告
        print(f"Warning: Failed to fetch GPU usage from {original_url}, all endpoints tried: {metrics_endpoints}")
    
    return json.dumps({
        "pending_queue_length": 0,
        "num_running": 0,
        "gpu_cache_usage": 0.0,
        "gpu_hit_rate": 0.0
    })

async def async_request_vllm(
    request_func_input: RequestFuncInput,
    pbar: Optional[tqdm] = None,
    ignore_eos: bool = True,
    **kwargs
) -> RequestFuncOutput:
    global _rr_counter
    api_url_list = request_func_input.api_url.split(',')
    
    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        payload = {
            "prompt": request_func_input.prompt,
            "n": 1,
            "best_of": request_func_input.best_of,
            "temperature": 0.0 if request_func_input.use_beam_search else 1.0,
            "top_p": 1.0,
            "max_tokens": request_func_input.output_len,
            "ignore_eos": ignore_eos,
            "stream": True,
        }
        output = RequestFuncOutput()
        output.prompt_len = request_func_input.prompt_len

        ttft = 0
        st = time.perf_counter()
        most_recent_timestamp = st
        try:
            
            if len(api_url_list) == 1:
                api_url = api_url_list[0]
            else:
                idx = _rr_counter % len(api_url_list)
                _rr_counter += 1
                api_url = api_url_list[idx]
            assert api_url.endswith("generate")
            
            async with session.post(url=api_url, json=payload) as response:
                if response.status == 200:
                    async for data in response.content.iter_any():
                        
                        timestamp = time.perf_counter()
                        # First token
                        if ttft == 0.0:
                            ttft = time.perf_counter() - st
                            output.ttft = ttft

                        # Decoding phase
                        else:
                            output.itl.append(timestamp -
                                              most_recent_timestamp)

                        most_recent_timestamp = timestamp
                    output.latency = time.perf_counter() - st

                    # When streaming, '\0' is appended to the end of the response.
                    body = data.decode("utf-8").strip("\0")
                    
                    # body=data.decode("utf-8").split("\0")[0].strip("\0")
                    try:
                        output.generated_text = json.loads(
                        body)["text"][0][len(request_func_input.prompt):]
                        
                        output.success = True
                    except:
                        output.success = False

                else:
                    output.success = False
        except (aiohttp.ClientOSError, aiohttp.ServerDisconnectedError,
                asyncio.TimeoutError, ConnectionError) as e:
            output.success = False
            output.error = type(e).__name__

        try:
            gpu_usages_waiting_len = await asyncio.gather(*[fetch_stats(session, url) for url in api_url_list])
            output.gpu_hit_rate = [json.loads(m)['gpu_hit_rate'] for m in gpu_usages_waiting_len]
        except Exception:
            output.gpu_hit_rate = [0.0] * len(api_url_list)

        if pbar:
            pbar.update(1)
        return output

ASYNC_REQUEST_FUNCS = {
    "vllm": async_request_vllm
}
