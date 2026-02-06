"""
Instance Pool Module

Manages a pool of vLLM instances with lifecycle management,
health monitoring, and dynamic scaling capabilities.
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum

from .vllm_instance import VLLMInstance, InstanceConfig, InstanceState, GenerationParams, GenerationResult
from .gpu_allocator import GPUAllocator, GPUAllocation

logger = logging.getLogger(__name__)


class PoolState(Enum):
    """States of the instance pool."""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"


@dataclass
class PoolConfig:
    """Configuration for the instance pool."""
    instances: List[Dict[str, Any]]  # List of instance configs
    model_config: Dict[str, Any]
    server_config: Dict[str, Any]
    
    # Health check settings
    health_check_interval: float = 30.0
    auto_restart: bool = True
    max_restart_attempts: int = 3
    
    # Startup settings
    startup_timeout: int = 300
    parallel_startup: bool = False  # Start instances sequentially by default


@dataclass
class PoolStats:
    """Statistics for the instance pool."""
    total_instances: int
    ready_instances: int
    busy_instances: int
    error_instances: int
    total_requests: int
    failed_requests: int
    
    # Per-TP statistics
    by_tp_degree: Dict[int, Dict[str, int]] = field(default_factory=dict)


class InstancePool:
    """
    Manages a pool of vLLM instances.
    
    Features:
    - Lifecycle management (start, stop, restart)
    - Health monitoring with automatic recovery
    - Instance grouping by TP degree
    - Dynamic instance addition/removal
    - Statistics collection
    """
    
    def __init__(
        self,
        config: PoolConfig,
        gpu_allocator: Optional[GPUAllocator] = None
    ):
        """
        Initialize the instance pool.
        
        Args:
            config: Pool configuration
            gpu_allocator: GPU allocator for resource management
        """
        self.config = config
        self.gpu_allocator = gpu_allocator or GPUAllocator()
        
        # Instance storage
        self._instances: Dict[str, VLLMInstance] = {}
        self._instances_by_tp: Dict[int, List[VLLMInstance]] = defaultdict(list)
        
        # State management
        self._state = PoolState.STOPPED
        self._health_check_task: Optional[asyncio.Task] = None
        self._restart_counts: Dict[str, int] = defaultdict(int)
        
        # Port management
        self._next_port_offset = 0
        self._used_ports: Set[int] = set()
    
    @property
    def state(self) -> PoolState:
        """Get current pool state."""
        return self._state
    
    @property
    def instances(self) -> List[VLLMInstance]:
        """Get all instances."""
        return list(self._instances.values())
    
    @property
    def ready_instances(self) -> List[VLLMInstance]:
        """Get all ready instances."""
        return [i for i in self._instances.values() if i.is_ready]
    
    def get_instances_by_tp(self, tp_degree: int) -> List[VLLMInstance]:
        """Get instances with specific TP degree."""
        return [i for i in self._instances_by_tp[tp_degree] if i.is_ready]
    
    def get_available_tp_degrees(self) -> List[int]:
        """Get list of available TP degrees."""
        return sorted(set(i.tp_degree for i in self.ready_instances))
    
    async def start(self) -> bool:
        """
        Start all instances in the pool.
        
        Returns:
            True if all instances started successfully
        """
        if self._state != PoolState.STOPPED:
            logger.warning("Pool is not in stopped state")
            return False
        
        self._state = PoolState.STARTING
        logger.info("Starting instance pool")
        
        try:
            # Create instances from config
            await self._create_instances()
            
            # Start instances
            if self.config.parallel_startup:
                success = await self._start_parallel()
            else:
                success = await self._start_sequential()
            
            if success:
                self._state = PoolState.RUNNING
                # Start health check background task
                self._start_health_check()
                logger.info(f"Instance pool started with {len(self.ready_instances)} ready instances")
            else:
                self._state = PoolState.ERROR
                logger.error("Failed to start all instances")
            
            return success
            
        except Exception as e:
            self._state = PoolState.ERROR
            logger.error(f"Error starting pool: {e}")
            return False
    
    async def _create_instances(self) -> None:
        """Create VLLMInstance objects from configuration."""
        base_port = self.config.server_config.get("base_port", 8000)
        
        for idx, inst_config in enumerate(self.config.instances):
            instance_id = inst_config.get("instance_id", f"instance_{idx}")
            tp_degree = inst_config.get("tp", 1)
            gpu_ids = inst_config.get("gpus", [idx % 8])
            
            # Allocate port
            port = base_port + self._next_port_offset
            self._used_ports.add(port)
            self._next_port_offset += 1
            
            # Create instance config
            config = InstanceConfig(
                instance_id=instance_id,
                tp_degree=tp_degree,
                gpu_ids=gpu_ids,
                port=port,
                host=self.config.server_config.get("host", "127.0.0.1"),
                model_name=self.config.model_config.get("name", "Qwen/Qwen3-32B"),
                max_model_len=self.config.model_config.get("max_model_len", 8192),
                dtype=self.config.model_config.get("dtype", "auto"),
                trust_remote_code=self.config.model_config.get("trust_remote_code", True),
                gpu_memory_utilization=self.config.model_config.get("gpu_memory_utilization", 0.90),
                log_dir=self.config.server_config.get("log_dir", None)
            )
            
            # Allocate GPUs
            if self.gpu_allocator:
                allocation = self.gpu_allocator.allocate(
                    instance_id=instance_id,
                    gpu_ids=gpu_ids
                )
                if not allocation:
                    logger.warning(f"GPU allocation failed for {instance_id}")
            
            # Create instance
            instance = VLLMInstance(config)
            self._instances[instance_id] = instance
            self._instances_by_tp[tp_degree].append(instance)
            
            logger.debug(f"Created instance {instance_id}: TP={tp_degree}, GPUs={gpu_ids}, Port={port}")
    
    async def _start_sequential(self) -> bool:
        """Start instances sequentially."""
        success_count = 0
        
        for instance_id, instance in self._instances.items():
            logger.info(f"Starting instance {instance_id}")
            
            if await instance.start(timeout=self.config.startup_timeout):
                success_count += 1
            else:
                logger.error(f"Failed to start instance {instance_id}")
        
        return success_count == len(self._instances)
    
    async def _start_parallel(self) -> bool:
        """Start instances in parallel."""
        tasks = [
            instance.start(timeout=self.config.startup_timeout)
            for instance in self._instances.values()
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        success_count = sum(1 for r in results if r is True)
        return success_count == len(self._instances)
    
    async def stop(self) -> None:
        """Stop all instances in the pool."""
        if self._state == PoolState.STOPPED:
            return
        
        self._state = PoolState.STOPPING
        logger.info("Stopping instance pool")
        
        # Stop health check
        self._stop_health_check()
        
        # Stop all instances
        stop_tasks = [instance.stop() for instance in self._instances.values()]
        await asyncio.gather(*stop_tasks, return_exceptions=True)
        
        # Release GPU allocations
        if self.gpu_allocator:
            for instance_id in self._instances:
                self.gpu_allocator.release(instance_id)
        
        self._state = PoolState.STOPPED
        logger.info("Instance pool stopped")
    
    def _start_health_check(self) -> None:
        """Start the background health check task."""
        if self._health_check_task is not None:
            return
        
        self._health_check_task = asyncio.create_task(self._health_check_loop())
        logger.debug("Health check started")
    
    def _stop_health_check(self) -> None:
        """Stop the background health check task."""
        if self._health_check_task:
            self._health_check_task.cancel()
            self._health_check_task = None
    
    async def _health_check_loop(self) -> None:
        """Background health check loop."""
        while self._state == PoolState.RUNNING:
            try:
                await self._check_all_instances()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check error: {e}")
            
            await asyncio.sleep(self.config.health_check_interval)
    
    async def _check_all_instances(self) -> None:
        """Check health of all instances and restart if needed."""
        for instance_id, instance in self._instances.items():
            if instance.state == InstanceState.READY:
                healthy = await instance.health_check()
                
                if not healthy:
                    logger.warning(f"Instance {instance_id} is unhealthy")
                    
                    if self.config.auto_restart:
                        await self._try_restart(instance_id)
    
    async def _try_restart(self, instance_id: str) -> bool:
        """Try to restart a failed instance."""
        if self._restart_counts[instance_id] >= self.config.max_restart_attempts:
            logger.error(f"Max restart attempts reached for {instance_id}")
            return False
        
        instance = self._instances.get(instance_id)
        if not instance:
            return False
        
        self._restart_counts[instance_id] += 1
        logger.info(f"Restarting instance {instance_id} (attempt {self._restart_counts[instance_id]})")
        
        await instance.stop()
        success = await instance.start(timeout=self.config.startup_timeout)
        
        if success:
            self._restart_counts[instance_id] = 0
        
        return success
    
    async def add_instance(self, config: Dict[str, Any]) -> Optional[VLLMInstance]:
        """
        Dynamically add a new instance to the pool.
        
        Args:
            config: Instance configuration
            
        Returns:
            The new instance if successful, None otherwise
        """
        instance_id = config.get("instance_id", f"dynamic_{len(self._instances)}")
        
        if instance_id in self._instances:
            logger.warning(f"Instance {instance_id} already exists")
            return None
        
        tp_degree = config.get("tp", 1)
        gpu_ids = config.get("gpus", [])
        
        # Allocate port
        base_port = self.config.server_config.get("base_port", 8000)
        port = base_port + self._next_port_offset
        self._used_ports.add(port)
        self._next_port_offset += 1
        
        # Allocate GPUs
        if self.gpu_allocator:
            allocation = self.gpu_allocator.allocate(instance_id, gpu_ids)
            if not allocation:
                logger.error(f"GPU allocation failed for {instance_id}")
                return None
        
        # Create instance
        inst_config = InstanceConfig(
            instance_id=instance_id,
            tp_degree=tp_degree,
            gpu_ids=gpu_ids,
            port=port,
            host=self.config.server_config.get("host", "127.0.0.1"),
            model_name=self.config.model_config.get("name", "Qwen/Qwen3-32B"),
            max_model_len=self.config.model_config.get("max_model_len", 8192),
            dtype=self.config.model_config.get("dtype", "auto"),
            trust_remote_code=self.config.model_config.get("trust_remote_code", True),
            gpu_memory_utilization=self.config.model_config.get("gpu_memory_utilization", 0.90)
        )
        
        instance = VLLMInstance(inst_config)
        
        # Start if pool is running
        if self._state == PoolState.RUNNING:
            if not await instance.start(timeout=self.config.startup_timeout):
                logger.error(f"Failed to start instance {instance_id}")
                if self.gpu_allocator:
                    self.gpu_allocator.release(instance_id)
                return None
        
        # Add to pool
        self._instances[instance_id] = instance
        self._instances_by_tp[tp_degree].append(instance)
        
        logger.info(f"Added instance {instance_id} to pool")
        return instance
    
    async def remove_instance(self, instance_id: str) -> bool:
        """
        Remove an instance from the pool.
        
        Args:
            instance_id: ID of the instance to remove
            
        Returns:
            True if successfully removed
        """
        instance = self._instances.get(instance_id)
        if not instance:
            return False
        
        # Stop the instance
        await instance.stop()
        
        # Release GPU allocation
        if self.gpu_allocator:
            self.gpu_allocator.release(instance_id)
        
        # Remove from storage
        del self._instances[instance_id]
        self._instances_by_tp[instance.tp_degree].remove(instance)
        
        logger.info(f"Removed instance {instance_id} from pool")
        return True
    
    def get_instance(self, instance_id: str) -> Optional[VLLMInstance]:
        """Get an instance by ID."""
        return self._instances.get(instance_id)
    
    def get_least_loaded_instance(
        self,
        tp_degree: Optional[int] = None
    ) -> Optional[VLLMInstance]:
        """
        Get the instance with the fewest active requests.
        
        Args:
            tp_degree: Optional TP degree filter
            
        Returns:
            The least loaded ready instance
        """
        if tp_degree is not None:
            candidates = self.get_instances_by_tp(tp_degree)
        else:
            candidates = self.ready_instances
        
        if not candidates:
            return None
        
        return min(candidates, key=lambda i: i.active_requests)
    
    def get_stats(self) -> PoolStats:
        """Get pool statistics."""
        by_tp: Dict[int, Dict[str, int]] = defaultdict(lambda: {
            "total": 0,
            "ready": 0,
            "requests": 0
        })
        
        total_requests = 0
        failed_requests = 0
        ready_count = 0
        busy_count = 0
        error_count = 0
        
        for instance in self._instances.values():
            tp = instance.tp_degree
            by_tp[tp]["total"] += 1
            total_requests += instance.total_requests
            failed_requests += instance._failed_requests
            
            if instance.state == InstanceState.READY:
                ready_count += 1
                by_tp[tp]["ready"] += 1
            elif instance.state == InstanceState.BUSY:
                busy_count += 1
            elif instance.state == InstanceState.ERROR:
                error_count += 1
            
            by_tp[tp]["requests"] += instance.total_requests
        
        return PoolStats(
            total_instances=len(self._instances),
            ready_instances=ready_count,
            busy_instances=busy_count,
            error_instances=error_count,
            total_requests=total_requests,
            failed_requests=failed_requests,
            by_tp_degree=dict(by_tp)
        )
    
    def print_status(self) -> None:
        """Print pool status."""
        stats = self.get_stats()
        
        print("\n" + "=" * 60)
        print("Instance Pool Status")
        print("=" * 60)
        print(f"State: {self._state.value}")
        print(f"Total Instances: {stats.total_instances}")
        print(f"Ready: {stats.ready_instances}, Busy: {stats.busy_instances}, Error: {stats.error_instances}")
        print(f"Total Requests: {stats.total_requests}, Failed: {stats.failed_requests}")
        
        print("\nBy TP Degree:")
        for tp, tp_stats in sorted(stats.by_tp_degree.items()):
            print(f"  TP={tp}: {tp_stats['ready']}/{tp_stats['total']} ready, {tp_stats['requests']} requests")
        
        print("\nInstance Details:")
        for instance in self._instances.values():
            print(f"  {instance}")
        print("=" * 60)
    
    async def __aenter__(self):
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop()


def create_pool_from_config(config: Dict[str, Any]) -> InstancePool:
    """
    Create an instance pool from configuration dictionary.
    
    Args:
        config: Full configuration dictionary
        
    Returns:
        Configured InstancePool
    """
    # Determine instance configurations based on scenario
    tp_configs = config.get("tp_configs", {})
    
    # Default to heterogeneous configuration
    instances = tp_configs.get("heterogeneous", [])
    if not instances:
        # Fall back to first homogeneous config
        homogeneous = tp_configs.get("homogeneous", [])
        if homogeneous:
            first_config = homogeneous[0]
            tp = first_config.get("tp", 1)
            num_instances = first_config.get("instances", 1)
            gpu_per_instance = tp
            
            instances = []
            for i in range(num_instances):
                start_gpu = i * gpu_per_instance
                instances.append({
                    "instance_id": f"homo_tp{tp}_{i}",
                    "tp": tp,
                    "gpus": list(range(start_gpu, start_gpu + gpu_per_instance))
                })
    
    pool_config = PoolConfig(
        instances=instances,
        model_config=config.get("model", {}),
        server_config=config.get("vllm_server", {}),
        health_check_interval=config.get("vllm_server", {}).get("health_check_interval", 30.0),
        startup_timeout=config.get("vllm_server", {}).get("startup_timeout", 300)
    )
    
    # Create GPU allocator
    gpu_config = config.get("gpu", {})
    gpu_allocator = GPUAllocator(
        total_gpus=gpu_config.get("total_gpus", 8),
        available_gpus=gpu_config.get("available_gpus")
    )
    
    return InstancePool(pool_config, gpu_allocator)
