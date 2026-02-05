"""
GPU Allocator Module

Manages GPU resource allocation for vLLM instances.
Prevents GPU conflicts and tracks resource utilization.
"""

import logging
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field
from threading import Lock

logger = logging.getLogger(__name__)


@dataclass
class GPUAllocation:
    """Represents a GPU allocation for an instance."""
    instance_id: str
    gpu_ids: List[int]
    tp_degree: int
    allocated_at: float = 0.0
    
    def __repr__(self) -> str:
        return f"GPUAllocation(instance={self.instance_id}, gpus={self.gpu_ids}, tp={self.tp_degree})"


@dataclass
class GPUInfo:
    """Information about a single GPU."""
    gpu_id: int
    total_memory_gb: float = 0.0
    is_available: bool = True
    allocated_to: Optional[str] = None


class GPUAllocator:
    """
    Manages GPU allocation for multiple vLLM instances.
    
    Features:
    - Track GPU availability
    - Prevent double allocation
    - Support for TP (Tensor Parallel) configurations
    - Thread-safe operations
    """
    
    def __init__(
        self,
        total_gpus: int = 8,
        available_gpus: Optional[List[int]] = None
    ):
        """
        Initialize the GPU allocator.
        
        Args:
            total_gpus: Total number of GPUs in the system
            available_gpus: List of GPU IDs available for allocation (default: all)
        """
        self.total_gpus = total_gpus
        self.available_gpus = set(available_gpus if available_gpus else range(total_gpus))
        
        # Initialize GPU tracking
        self._gpu_info: Dict[int, GPUInfo] = {}
        for gpu_id in range(total_gpus):
            self._gpu_info[gpu_id] = GPUInfo(
                gpu_id=gpu_id,
                is_available=gpu_id in self.available_gpus
            )
        
        # Allocation tracking
        self._allocations: Dict[str, GPUAllocation] = {}
        self._gpu_to_instance: Dict[int, str] = {}
        
        # Thread safety
        self._lock = Lock()
        
        logger.info(f"GPU Allocator initialized: {total_gpus} GPUs, {len(self.available_gpus)} available")
    
    def allocate(
        self,
        instance_id: str,
        gpu_ids: List[int],
        tp_degree: Optional[int] = None
    ) -> Optional[GPUAllocation]:
        """
        Allocate GPUs to an instance.
        
        Args:
            instance_id: Unique identifier for the instance
            gpu_ids: List of GPU IDs to allocate
            tp_degree: TP degree (default: len(gpu_ids))
            
        Returns:
            GPUAllocation if successful, None otherwise
        """
        with self._lock:
            # Check if instance already has allocation
            if instance_id in self._allocations:
                logger.warning(f"Instance {instance_id} already has GPU allocation")
                return self._allocations[instance_id]
            
            # Validate GPU IDs
            for gpu_id in gpu_ids:
                if gpu_id not in self.available_gpus:
                    logger.error(f"GPU {gpu_id} is not available for allocation")
                    return None
                
                if gpu_id in self._gpu_to_instance:
                    logger.error(f"GPU {gpu_id} is already allocated to {self._gpu_to_instance[gpu_id]}")
                    return None
            
            # Create allocation
            import time
            allocation = GPUAllocation(
                instance_id=instance_id,
                gpu_ids=gpu_ids,
                tp_degree=tp_degree or len(gpu_ids),
                allocated_at=time.time()
            )
            
            # Update tracking
            self._allocations[instance_id] = allocation
            for gpu_id in gpu_ids:
                self._gpu_to_instance[gpu_id] = instance_id
                self._gpu_info[gpu_id].is_available = False
                self._gpu_info[gpu_id].allocated_to = instance_id
            
            logger.info(f"Allocated GPUs {gpu_ids} to instance {instance_id}")
            return allocation
    
    def release(self, instance_id: str) -> bool:
        """
        Release GPU allocation for an instance.
        
        Args:
            instance_id: Instance ID to release
            
        Returns:
            True if successfully released
        """
        with self._lock:
            allocation = self._allocations.get(instance_id)
            if not allocation:
                logger.warning(f"No allocation found for instance {instance_id}")
                return False
            
            # Release GPUs
            for gpu_id in allocation.gpu_ids:
                if gpu_id in self._gpu_to_instance:
                    del self._gpu_to_instance[gpu_id]
                self._gpu_info[gpu_id].is_available = True
                self._gpu_info[gpu_id].allocated_to = None
            
            # Remove allocation
            del self._allocations[instance_id]
            
            logger.info(f"Released GPUs {allocation.gpu_ids} from instance {instance_id}")
            return True
    
    def get_allocation(self, instance_id: str) -> Optional[GPUAllocation]:
        """Get allocation for an instance."""
        with self._lock:
            return self._allocations.get(instance_id)
    
    def get_free_gpus(self) -> List[int]:
        """Get list of unallocated GPU IDs."""
        with self._lock:
            return [
                gpu_id for gpu_id in self.available_gpus
                if gpu_id not in self._gpu_to_instance
            ]
    
    def get_contiguous_free_gpus(self, count: int) -> Optional[List[int]]:
        """
        Find contiguous free GPUs for TP allocation.
        
        Args:
            count: Number of contiguous GPUs needed
            
        Returns:
            List of contiguous GPU IDs if available, None otherwise
        """
        free_gpus = sorted(self.get_free_gpus())
        
        if len(free_gpus) < count:
            return None
        
        # Find contiguous sequence
        for i in range(len(free_gpus) - count + 1):
            sequence = free_gpus[i:i + count]
            if sequence == list(range(sequence[0], sequence[0] + count)):
                return sequence
        
        return None
    
    def can_allocate(self, gpu_ids: List[int]) -> bool:
        """Check if specified GPUs can be allocated."""
        with self._lock:
            for gpu_id in gpu_ids:
                if gpu_id not in self.available_gpus:
                    return False
                if gpu_id in self._gpu_to_instance:
                    return False
            return True
    
    def get_allocation_for_tp(self, tp_degree: int) -> Optional[List[int]]:
        """
        Find available GPUs for a given TP degree.
        
        Args:
            tp_degree: Required TP degree
            
        Returns:
            List of GPU IDs if available, None otherwise
        """
        # First try to find contiguous GPUs
        contiguous = self.get_contiguous_free_gpus(tp_degree)
        if contiguous:
            return contiguous
        
        # Fall back to any available GPUs
        free_gpus = self.get_free_gpus()
        if len(free_gpus) >= tp_degree:
            return free_gpus[:tp_degree]
        
        return None
    
    def get_all_allocations(self) -> Dict[str, GPUAllocation]:
        """Get all current allocations."""
        with self._lock:
            return dict(self._allocations)
    
    def get_gpu_status(self) -> Dict[int, Dict[str, Any]]:
        """Get status of all GPUs."""
        with self._lock:
            return {
                gpu_id: {
                    "available": info.is_available,
                    "allocated_to": info.allocated_to,
                    "in_pool": gpu_id in self.available_gpus
                }
                for gpu_id, info in self._gpu_info.items()
            }
    
    def reset(self) -> None:
        """Reset all allocations."""
        with self._lock:
            self._allocations.clear()
            self._gpu_to_instance.clear()
            
            for gpu_id in self._gpu_info:
                self._gpu_info[gpu_id].is_available = gpu_id in self.available_gpus
                self._gpu_info[gpu_id].allocated_to = None
            
            logger.info("GPU allocator reset")
    
    def print_status(self) -> None:
        """Print current GPU allocation status."""
        print("\n" + "=" * 50)
        print("GPU Allocation Status")
        print("=" * 50)
        print(f"Total GPUs: {self.total_gpus}")
        print(f"Available Pool: {sorted(self.available_gpus)}")
        print(f"Free GPUs: {self.get_free_gpus()}")
        
        print("\nGPU Details:")
        for gpu_id, info in sorted(self._gpu_info.items()):
            status = "free" if info.is_available else f"allocated to {info.allocated_to}"
            pool_status = "in pool" if gpu_id in self.available_gpus else "excluded"
            print(f"  GPU {gpu_id}: {status} ({pool_status})")
        
        print("\nAllocations:")
        if self._allocations:
            for alloc in self._allocations.values():
                print(f"  {alloc}")
        else:
            print("  No active allocations")
        print("=" * 50)
    
    def __repr__(self) -> str:
        free_count = len(self.get_free_gpus())
        allocated_count = len(self._allocations)
        return f"GPUAllocator(total={self.total_gpus}, free={free_count}, allocations={allocated_count})"


class GPUAllocationPlanner:
    """
    Plans GPU allocations for different scenarios.
    
    Helps determine optimal GPU assignments for homogeneous
    and heterogeneous configurations.
    """
    
    def __init__(self, total_gpus: int = 8):
        """
        Initialize the allocation planner.
        
        Args:
            total_gpus: Total number of GPUs available
        """
        self.total_gpus = total_gpus
    
    def plan_homogeneous(self, tp_degree: int) -> List[Dict[str, Any]]:
        """
        Plan allocations for homogeneous configuration.
        
        Args:
            tp_degree: TP degree for all instances
            
        Returns:
            List of instance configurations
        """
        if self.total_gpus % tp_degree != 0:
            raise ValueError(f"Cannot evenly divide {self.total_gpus} GPUs with TP={tp_degree}")
        
        num_instances = self.total_gpus // tp_degree
        allocations = []
        
        for i in range(num_instances):
            start_gpu = i * tp_degree
            gpus = list(range(start_gpu, start_gpu + tp_degree))
            
            allocations.append({
                "instance_id": f"homo_tp{tp_degree}_{i}",
                "tp": tp_degree,
                "gpus": gpus,
                "description": f"Homogeneous TP={tp_degree} instance {i}"
            })
        
        return allocations
    
    def plan_heterogeneous(
        self,
        config: List[Dict[str, int]]
    ) -> List[Dict[str, Any]]:
        """
        Plan allocations for heterogeneous configuration.
        
        Args:
            config: List of {tp: degree, count: num_instances}
            
        Returns:
            List of instance configurations
        """
        # Calculate total GPUs needed
        total_needed = sum(c["tp"] * c.get("count", 1) for c in config)
        if total_needed > self.total_gpus:
            raise ValueError(f"Configuration requires {total_needed} GPUs, only {self.total_gpus} available")
        
        allocations = []
        current_gpu = 0
        
        for cfg in config:
            tp = cfg["tp"]
            count = cfg.get("count", 1)
            
            for i in range(count):
                gpus = list(range(current_gpu, current_gpu + tp))
                
                allocations.append({
                    "instance_id": f"hetero_tp{tp}_{i}",
                    "tp": tp,
                    "gpus": gpus,
                    "description": f"Heterogeneous TP={tp} instance {i}"
                })
                
                current_gpu += tp
        
        return allocations
    
    def plan_default_heterogeneous(self) -> List[Dict[str, Any]]:
        """
        Plan default heterogeneous configuration.
        
        Default: TP=4 (4 GPUs) + TP=2 (2 GPUs) + TP=1 x 2 (2 GPUs)
        
        Returns:
            List of instance configurations
        """
        return self.plan_heterogeneous([
            {"tp": 4, "count": 1},  # 1 instance with TP=4 (GPUs 0-3)
            {"tp": 2, "count": 1},  # 1 instance with TP=2 (GPUs 4-5)
            {"tp": 1, "count": 2},  # 2 instances with TP=1 (GPUs 6, 7)
        ])
    
    def validate_allocation(self, allocations: List[Dict[str, Any]]) -> bool:
        """
        Validate that allocations don't have conflicts.
        
        Args:
            allocations: List of instance configurations
            
        Returns:
            True if allocations are valid
        """
        used_gpus: Set[int] = set()
        
        for alloc in allocations:
            gpus = alloc.get("gpus", [])
            
            # Check for overlaps
            for gpu in gpus:
                if gpu in used_gpus:
                    logger.error(f"GPU {gpu} is used by multiple allocations")
                    return False
                if gpu < 0 or gpu >= self.total_gpus:
                    logger.error(f"GPU {gpu} is out of range [0, {self.total_gpus})")
                    return False
                used_gpus.add(gpu)
            
            # Validate TP degree matches GPU count
            tp = alloc.get("tp", 1)
            if len(gpus) != tp:
                logger.error(f"TP degree {tp} doesn't match GPU count {len(gpus)}")
                return False
        
        return True
