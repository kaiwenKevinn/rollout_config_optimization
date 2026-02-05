"""
Heterogeneous Scheduler Module

Intelligent scheduler for heterogeneous TP configurations.
Routes requests based on sequence length to appropriate TP instances.
"""

import time
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

from .base_scheduler import (
    BaseScheduler,
    SchedulerRequest,
    SchedulingResult,
    SchedulingStrategy,
    LoadBalancer,
    SchedulerStatistics
)

logger = logging.getLogger(__name__)


@dataclass
class RoutingRule:
    """Defines a routing rule for sequence categories."""
    category: str  # "short", "medium", "long", "extra_long"
    preferred_tp_degrees: List[int]  # In order of preference
    fallback_tp_degrees: List[int]  # Fallback options if preferred unavailable


@dataclass
class HeterogeneousStatistics(SchedulerStatistics):
    """Extended statistics for heterogeneous scheduling."""
    
    # Routing statistics
    preferred_routes: int = 0  # Routed to preferred TP
    fallback_routes: int = 0  # Routed to fallback TP
    
    # Per-category routing
    category_to_tp: Dict[str, Dict[int, int]] = field(default_factory=dict)
    
    @property
    def preferred_rate(self) -> float:
        """Rate of routing to preferred instances."""
        total = self.preferred_routes + self.fallback_routes
        if total == 0:
            return 1.0
        return self.preferred_routes / total
    
    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            "preferred_routes": self.preferred_routes,
            "fallback_routes": self.fallback_routes,
            "preferred_rate": self.preferred_rate,
            "category_to_tp": self.category_to_tp
        })
        return base


class HeterogeneousScheduler(BaseScheduler):
    """
    Intelligent scheduler for heterogeneous TP configurations.
    
    Routes requests based on sequence length:
    - Short sequences (<=256 tokens) -> TP=1 or TP=2 instances
    - Medium sequences (257-512 tokens) -> TP=2 or TP=4 instances
    - Long sequences (>512 tokens) -> TP=4 or TP=8 instances
    
    Features:
    - Length-based intelligent routing
    - Load-aware instance selection
    - Adaptive threshold adjustment
    - Fallback routing when preferred instances are busy
    """
    
    DEFAULT_ROUTING_RULES = {
        "short": RoutingRule(
            category="short",
            preferred_tp_degrees=[1, 2],
            fallback_tp_degrees=[4, 8]
        ),
        "medium": RoutingRule(
            category="medium",
            preferred_tp_degrees=[2, 4],
            fallback_tp_degrees=[1, 8]
        ),
        "long": RoutingRule(
            category="long",
            preferred_tp_degrees=[4, 8],
            fallback_tp_degrees=[2, 1]
        ),
        "extra_long": RoutingRule(
            category="extra_long",
            preferred_tp_degrees=[8, 4],
            fallback_tp_degrees=[2, 1]
        )
    }
    
    def __init__(
        self,
        routing_rules: Optional[Dict[str, List[int]]] = None,
        length_thresholds: Optional[Dict[str, int]] = None,
        load_balance_strategy: SchedulingStrategy = SchedulingStrategy.LEAST_CONNECTIONS,
        max_queue_length: int = 100,
        enable_fallback: bool = True,
        adaptive_routing: bool = True
    ):
        """
        Initialize the heterogeneous scheduler.
        
        Args:
            routing_rules: Custom routing rules {category: [tp_degrees]}
            length_thresholds: Custom length thresholds
            load_balance_strategy: Strategy for balancing within TP groups
            max_queue_length: Maximum queue length before using fallback
            enable_fallback: Whether to use fallback routing
            adaptive_routing: Enable adaptive threshold adjustment
        """
        super().__init__(name="heterogeneous")
        
        # Configure routing rules
        self._routing_rules = self._build_routing_rules(routing_rules)
        
        # Length thresholds
        self._thresholds = length_thresholds or {
            "short": 256,
            "medium": 512,
            "long": 1024
        }
        
        # Load balancing
        self._load_balancer = LoadBalancer(load_balance_strategy)
        self._max_queue_length = max_queue_length
        self._enable_fallback = enable_fallback
        self._adaptive_routing = adaptive_routing
        
        # Extended statistics
        self._hetero_stats = HeterogeneousStatistics()
        
        # Performance tracking for adaptive routing
        self._performance_history: Dict[Tuple[str, int], List[float]] = defaultdict(list)
        
        logger.info(f"Initialized HeterogeneousScheduler with thresholds: {self._thresholds}")
    
    def _build_routing_rules(
        self,
        custom_rules: Optional[Dict[str, List[int]]]
    ) -> Dict[str, RoutingRule]:
        """Build routing rules from configuration."""
        rules = dict(self.DEFAULT_ROUTING_RULES)
        
        if custom_rules:
            for category, tp_degrees in custom_rules.items():
                if category in rules:
                    rules[category].preferred_tp_degrees = tp_degrees
                else:
                    # Create new rule
                    rules[category] = RoutingRule(
                        category=category,
                        preferred_tp_degrees=tp_degrees,
                        fallback_tp_degrees=[]
                    )
        
        return rules
    
    async def route_request(
        self,
        request: SchedulerRequest,
        available_instances: List[Any]
    ) -> SchedulingResult:
        """
        Route a request based on sequence length.
        
        Args:
            request: Request to schedule
            available_instances: List of available vLLM instances
            
        Returns:
            SchedulingResult
        """
        start_time = time.time()
        
        # Record category
        self.record_category(request.sequence_category)
        
        # Get routing rule for this category
        rule = self._routing_rules.get(
            request.sequence_category,
            self._routing_rules.get("medium")  # Default fallback
        )
        
        # Try preferred TP degrees first
        selected = await self._try_route_to_preferred(
            request, available_instances, rule.preferred_tp_degrees
        )
        
        is_fallback = False
        
        # If preferred routing failed, try fallback
        if selected is None and self._enable_fallback:
            selected = await self._try_route_to_preferred(
                request, available_instances, rule.fallback_tp_degrees
            )
            is_fallback = True
        
        # Build result
        if selected is None:
            result = SchedulingResult(
                request_id=request.request_id,
                instance_id="",
                tp_degree=0,
                success=False,
                queue_time=time.time() - request.created_at,
                reason="No available instances for routing"
            )
            self._hetero_stats.failed_schedules += 1
        else:
            # Update request metadata
            request.scheduled_at = time.time()
            request.target_instance_id = selected.instance_id
            request.target_tp_degree = selected.tp_degree
            
            result = SchedulingResult(
                request_id=request.request_id,
                instance_id=selected.instance_id,
                tp_degree=selected.tp_degree,
                success=True,
                queue_time=request.get_queue_time(),
                reason="fallback" if is_fallback else "preferred"
            )
            
            # Update statistics
            if is_fallback:
                self._hetero_stats.fallback_routes += 1
            else:
                self._hetero_stats.preferred_routes += 1
            
            # Track category->TP mapping
            cat = request.sequence_category
            tp = selected.tp_degree
            if cat not in self._hetero_stats.category_to_tp:
                self._hetero_stats.category_to_tp[cat] = {}
            self._hetero_stats.category_to_tp[cat][tp] = \
                self._hetero_stats.category_to_tp[cat].get(tp, 0) + 1
            
            logger.debug(
                f"Scheduled {request.request_id} ({cat}, {request.input_tokens} tokens) "
                f"-> {selected.instance_id} (TP={tp})"
            )
        
        self.record_result(result)
        return result
    
    async def _try_route_to_preferred(
        self,
        request: SchedulerRequest,
        available_instances: List[Any],
        tp_preferences: List[int]
    ) -> Optional[Any]:
        """
        Try to route to instances with preferred TP degrees.
        
        Args:
            request: Request to route
            available_instances: Available instances
            tp_preferences: List of preferred TP degrees
            
        Returns:
            Selected instance or None
        """
        for tp_degree in tp_preferences:
            # Get instances with this TP degree
            candidates = [
                inst for inst in available_instances
                if inst.tp_degree == tp_degree and inst.is_ready
            ]
            
            if not candidates:
                continue
            
            # Check queue lengths
            if self._max_queue_length > 0:
                candidates = [
                    inst for inst in candidates
                    if inst.active_requests < self._max_queue_length
                ]
            
            if not candidates:
                continue
            
            # Select using load balancer
            selected = self._load_balancer.select_instance(candidates)
            if selected:
                return selected
        
        return None
    
    def get_preferred_tp_degrees(self, request: SchedulerRequest) -> List[int]:
        """
        Get preferred TP degrees for a request.
        
        Args:
            request: The request to evaluate
            
        Returns:
            List of preferred TP degrees
        """
        rule = self._routing_rules.get(
            request.sequence_category,
            self._routing_rules.get("medium")
        )
        
        return rule.preferred_tp_degrees if rule else [2, 4]
    
    def get_statistics(self) -> HeterogeneousStatistics:
        """Get extended statistics."""
        # Merge base stats
        self._hetero_stats.total_requests = self._statistics.total_requests
        self._hetero_stats.successful_schedules = self._statistics.successful_schedules
        self._hetero_stats.avg_queue_time = self._statistics.avg_queue_time
        self._hetero_stats.max_queue_time = self._statistics.max_queue_time
        self._hetero_stats.by_category = self._statistics.by_category
        self._hetero_stats.by_tp_degree = self._statistics.by_tp_degree
        self._hetero_stats.by_instance = self._statistics.by_instance
        
        return self._hetero_stats
    
    def update_routing_rule(
        self,
        category: str,
        preferred_tp_degrees: List[int],
        fallback_tp_degrees: Optional[List[int]] = None
    ) -> None:
        """
        Update routing rule for a category.
        
        Args:
            category: Sequence category
            preferred_tp_degrees: New preferred TP degrees
            fallback_tp_degrees: New fallback TP degrees
        """
        if category in self._routing_rules:
            self._routing_rules[category].preferred_tp_degrees = preferred_tp_degrees
            if fallback_tp_degrees:
                self._routing_rules[category].fallback_tp_degrees = fallback_tp_degrees
        else:
            self._routing_rules[category] = RoutingRule(
                category=category,
                preferred_tp_degrees=preferred_tp_degrees,
                fallback_tp_degrees=fallback_tp_degrees or []
            )
        
        logger.info(f"Updated routing rule for {category}: preferred={preferred_tp_degrees}")
    
    def update_thresholds(self, thresholds: Dict[str, int]) -> None:
        """
        Update length thresholds.
        
        Args:
            thresholds: New thresholds
        """
        self._thresholds.update(thresholds)
        logger.info(f"Updated thresholds: {self._thresholds}")
    
    def record_performance(
        self,
        category: str,
        tp_degree: int,
        latency: float
    ) -> None:
        """
        Record performance for adaptive routing.
        
        Args:
            category: Sequence category
            tp_degree: TP degree used
            latency: Observed latency
        """
        key = (category, tp_degree)
        self._performance_history[key].append(latency)
        
        # Keep only recent history
        max_history = 100
        if len(self._performance_history[key]) > max_history:
            self._performance_history[key] = self._performance_history[key][-max_history:]
        
        # Adaptive routing adjustment
        if self._adaptive_routing:
            self._adjust_routing_adaptively()
    
    def _adjust_routing_adaptively(self) -> None:
        """Adjust routing rules based on observed performance."""
        # This is a simplified adaptive algorithm
        # In production, more sophisticated approaches would be used
        
        for category in ["short", "medium", "long"]:
            category_performance: Dict[int, float] = {}
            
            for (cat, tp), latencies in self._performance_history.items():
                if cat == category and len(latencies) >= 10:
                    avg_latency = sum(latencies) / len(latencies)
                    category_performance[tp] = avg_latency
            
            if len(category_performance) >= 2:
                # Sort TP degrees by performance (lower latency = better)
                sorted_tps = sorted(category_performance.items(), key=lambda x: x[1])
                best_tps = [tp for tp, _ in sorted_tps[:2]]
                
                # Update routing rule if significantly different
                current_preferred = self._routing_rules[category].preferred_tp_degrees
                if best_tps != current_preferred[:len(best_tps)]:
                    logger.debug(
                        f"Adaptive routing: {category} updated from "
                        f"{current_preferred} to {best_tps}"
                    )
    
    def reset_statistics(self) -> None:
        """Reset all statistics."""
        super().reset_statistics()
        self._hetero_stats = HeterogeneousStatistics()
        self._performance_history.clear()
    
    def print_statistics(self) -> None:
        """Print detailed statistics."""
        stats = self.get_statistics()
        
        print("\n" + "=" * 60)
        print("Heterogeneous Scheduler Statistics")
        print("=" * 60)
        print(f"Total Requests: {stats.total_requests}")
        print(f"Successful: {stats.successful_schedules}")
        print(f"Failed: {stats.failed_schedules}")
        print(f"Success Rate: {stats.success_rate:.2%}")
        
        print(f"\nPreferred Routes: {stats.preferred_routes}")
        print(f"Fallback Routes: {stats.fallback_routes}")
        print(f"Preferred Rate: {stats.preferred_rate:.2%}")
        
        print(f"\nAvg Queue Time: {stats.avg_queue_time*1000:.2f} ms")
        print(f"Max Queue Time: {stats.max_queue_time*1000:.2f} ms")
        
        print("\nCategory -> TP Distribution:")
        for cat, tp_counts in sorted(stats.category_to_tp.items()):
            print(f"  {cat}:")
            for tp, count in sorted(tp_counts.items()):
                print(f"    TP={tp}: {count}")
        
        print("\nRouting Rules:")
        for cat, rule in sorted(self._routing_rules.items()):
            print(f"  {cat}: preferred={rule.preferred_tp_degrees}, "
                  f"fallback={rule.fallback_tp_degrees}")
        
        print("=" * 60)


class AdaptiveHeterogeneousScheduler(HeterogeneousScheduler):
    """
    Extended heterogeneous scheduler with advanced adaptive features.
    
    Additional features:
    - Real-time threshold adjustment
    - Performance-based TP preference updates
    - Queue pressure awareness
    """
    
    def __init__(
        self,
        **kwargs
    ):
        """Initialize with adaptive features enabled."""
        kwargs['adaptive_routing'] = True
        super().__init__(**kwargs)
        
        # Additional tracking
        self._queue_pressure: Dict[int, float] = defaultdict(float)
        self._throughput_history: Dict[int, List[float]] = defaultdict(list)
    
    async def route_request(
        self,
        request: SchedulerRequest,
        available_instances: List[Any]
    ) -> SchedulingResult:
        """Route with queue pressure awareness."""
        # Update queue pressure metrics
        self._update_queue_pressure(available_instances)
        
        # Use base routing logic
        result = await super().route_request(request, available_instances)
        
        return result
    
    def _update_queue_pressure(self, instances: List[Any]) -> None:
        """Update queue pressure metrics per TP degree."""
        tp_queues: Dict[int, List[int]] = defaultdict(list)
        
        for inst in instances:
            if inst.is_ready:
                tp_queues[inst.tp_degree].append(inst.active_requests)
        
        for tp, queues in tp_queues.items():
            if queues:
                avg_pressure = sum(queues) / len(queues) / max(self._max_queue_length, 1)
                self._queue_pressure[tp] = avg_pressure
    
    def get_queue_pressure(self) -> Dict[int, float]:
        """Get current queue pressure per TP degree."""
        return dict(self._queue_pressure)


def create_heterogeneous_scheduler(
    config: Dict[str, Any]
) -> HeterogeneousScheduler:
    """
    Create a heterogeneous scheduler from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured HeterogeneousScheduler
    """
    scheduling_config = config.get("scheduling", {})
    
    # Build routing rules from config
    routing_rules = scheduling_config.get("routing_rules", {})
    
    # Get thresholds
    thresholds = scheduling_config.get("length_thresholds", {
        "short": 256,
        "medium": 512,
        "long": 1024
    })
    
    # Get load balance strategy
    strategy_name = scheduling_config.get("load_balance_strategy", "least_connections")
    strategy_map = {
        "round_robin": SchedulingStrategy.ROUND_ROBIN,
        "least_connections": SchedulingStrategy.LEAST_CONNECTIONS,
        "weighted": SchedulingStrategy.WEIGHTED
    }
    strategy = strategy_map.get(strategy_name, SchedulingStrategy.LEAST_CONNECTIONS)
    
    return HeterogeneousScheduler(
        routing_rules=routing_rules,
        length_thresholds=thresholds,
        load_balance_strategy=strategy,
        max_queue_length=scheduling_config.get("max_queue_length", 100),
        enable_fallback=True,
        adaptive_routing=True
    )
