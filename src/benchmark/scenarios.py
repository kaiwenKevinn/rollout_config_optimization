"""
Benchmark Scenarios Module

Defines test scenarios for homogeneous and heterogeneous configurations.
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class ScenarioType(Enum):
    """Types of benchmark scenarios."""
    HOMOGENEOUS = "homogeneous"
    HETEROGENEOUS = "heterogeneous"
    PROFILING = "profiling"


@dataclass
class InstanceAllocation:
    """GPU allocation for a single instance."""
    instance_id: str
    tp_degree: int
    gpu_ids: List[int]
    description: str = ""


@dataclass
class ScenarioConfig:
    """Configuration for a benchmark scenario."""
    name: str
    scenario_type: ScenarioType
    description: str
    
    # Instance allocations
    instances: List[InstanceAllocation]
    
    # Benchmark settings
    num_runs: int = 3
    warmup_requests: int = 5
    max_concurrent_requests: int = 32
    
    # Scheduling settings (for heterogeneous)
    use_length_routing: bool = False
    length_thresholds: Dict[str, int] = field(default_factory=dict)
    routing_rules: Dict[str, List[int]] = field(default_factory=dict)
    
    def get_total_gpus(self) -> int:
        """Get total number of GPUs used."""
        gpus = set()
        for inst in self.instances:
            gpus.update(inst.gpu_ids)
        return len(gpus)
    
    def get_instance_count(self) -> int:
        """Get number of instances."""
        return len(self.instances)
    
    def get_tp_distribution(self) -> Dict[int, int]:
        """Get distribution of TP degrees."""
        dist: Dict[int, int] = {}
        for inst in self.instances:
            dist[inst.tp_degree] = dist.get(inst.tp_degree, 0) + 1
        return dist
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "type": self.scenario_type.value,
            "description": self.description,
            "instances": [
                {
                    "instance_id": i.instance_id,
                    "tp_degree": i.tp_degree,
                    "gpu_ids": i.gpu_ids,
                    "description": i.description
                }
                for i in self.instances
            ],
            "num_runs": self.num_runs,
            "warmup_requests": self.warmup_requests,
            "max_concurrent_requests": self.max_concurrent_requests,
            "total_gpus": self.get_total_gpus(),
            "instance_count": self.get_instance_count(),
            "tp_distribution": self.get_tp_distribution()
        }


# Predefined Homogeneous Scenarios
HOMOGENEOUS_TP1 = ScenarioConfig(
    name="homogeneous_tp1",
    scenario_type=ScenarioType.HOMOGENEOUS,
    description="8 instances with TP=1 (one GPU each)",
    instances=[
        InstanceAllocation(f"homo_tp1_{i}", 1, [i], f"TP=1 instance on GPU {i}")
        for i in range(8)
    ],
    use_length_routing=False
)

HOMOGENEOUS_TP2 = ScenarioConfig(
    name="homogeneous_tp2",
    scenario_type=ScenarioType.HOMOGENEOUS,
    description="4 instances with TP=2 (two GPUs each)",
    instances=[
        InstanceAllocation(
            f"homo_tp2_{i}", 2,
            [i*2, i*2+1],
            f"TP=2 instance on GPUs {i*2}-{i*2+1}"
        )
        for i in range(4)
    ],
    use_length_routing=False
)

HOMOGENEOUS_TP4 = ScenarioConfig(
    name="homogeneous_tp4",
    scenario_type=ScenarioType.HOMOGENEOUS,
    description="2 instances with TP=4 (four GPUs each)",
    instances=[
        InstanceAllocation(
            f"homo_tp4_{i}", 4,
            list(range(i*4, i*4+4)),
            f"TP=4 instance on GPUs {i*4}-{i*4+3}"
        )
        for i in range(2)
    ],
    use_length_routing=False
)

HOMOGENEOUS_TP8 = ScenarioConfig(
    name="homogeneous_tp8",
    scenario_type=ScenarioType.HOMOGENEOUS,
    description="1 instance with TP=8 (all GPUs)",
    instances=[
        InstanceAllocation(
            "homo_tp8_0", 8,
            list(range(8)),
            "TP=8 instance on all GPUs"
        )
    ],
    use_length_routing=False
)

# Predefined Heterogeneous Scenarios
HETEROGENEOUS_MIX_1 = ScenarioConfig(
    name="heterogeneous_mix_1",
    scenario_type=ScenarioType.HETEROGENEOUS,
    description="Mixed TP: 1x TP=4 + 1x TP=2 + 2x TP=1",
    instances=[
        InstanceAllocation("hetero_tp4_0", 4, [0, 1, 2, 3], "TP=4 for long sequences"),
        InstanceAllocation("hetero_tp2_0", 2, [4, 5], "TP=2 for medium sequences"),
        InstanceAllocation("hetero_tp1_0", 1, [6], "TP=1 for short sequences"),
        InstanceAllocation("hetero_tp1_1", 1, [7], "TP=1 for short sequences"),
    ],
    use_length_routing=True,
    length_thresholds={"short": 256, "medium": 512, "long": 1024},
    routing_rules={
        "short": [1, 2],
        "medium": [2, 4],
        "long": [4],
        "extra_long": [4]
    }
)

HETEROGENEOUS_MIX_2 = ScenarioConfig(
    name="heterogeneous_mix_2",
    scenario_type=ScenarioType.HETEROGENEOUS,
    description="Mixed TP: 1x TP=4 + 2x TP=2",
    instances=[
        InstanceAllocation("hetero_tp4_0", 4, [0, 1, 2, 3], "TP=4 for long sequences"),
        InstanceAllocation("hetero_tp2_0", 2, [4, 5], "TP=2 for medium/short sequences"),
        InstanceAllocation("hetero_tp2_1", 2, [6, 7], "TP=2 for medium/short sequences"),
    ],
    use_length_routing=True,
    length_thresholds={"short": 256, "medium": 512, "long": 1024},
    routing_rules={
        "short": [2, 4],
        "medium": [2, 4],
        "long": [4, 2],
        "extra_long": [4, 2]
    }
)

HETEROGENEOUS_MIX_3 = ScenarioConfig(
    name="heterogeneous_mix_3",
    scenario_type=ScenarioType.HETEROGENEOUS,
    description="Mixed TP: 2x TP=2 + 4x TP=1",
    instances=[
        InstanceAllocation("hetero_tp2_0", 2, [0, 1], "TP=2 for longer sequences"),
        InstanceAllocation("hetero_tp2_1", 2, [2, 3], "TP=2 for longer sequences"),
        InstanceAllocation("hetero_tp1_0", 1, [4], "TP=1 for short sequences"),
        InstanceAllocation("hetero_tp1_1", 1, [5], "TP=1 for short sequences"),
        InstanceAllocation("hetero_tp1_2", 1, [6], "TP=1 for short sequences"),
        InstanceAllocation("hetero_tp1_3", 1, [7], "TP=1 for short sequences"),
    ],
    use_length_routing=True,
    length_thresholds={"short": 256, "medium": 512, "long": 1024},
    routing_rules={
        "short": [1],
        "medium": [1, 2],
        "long": [2],
        "extra_long": [2]
    }
)

# All predefined scenarios
SCENARIOS: Dict[str, ScenarioConfig] = {
    # Homogeneous
    "homogeneous_tp1": HOMOGENEOUS_TP1,
    "homogeneous_tp2": HOMOGENEOUS_TP2,
    "homogeneous_tp4": HOMOGENEOUS_TP4,
    "homogeneous_tp8": HOMOGENEOUS_TP8,
    # Heterogeneous
    "heterogeneous_mix_1": HETEROGENEOUS_MIX_1,
    "heterogeneous_mix_2": HETEROGENEOUS_MIX_2,
    "heterogeneous_mix_3": HETEROGENEOUS_MIX_3,
}


def get_scenario(name: str) -> Optional[ScenarioConfig]:
    """Get a predefined scenario by name."""
    return SCENARIOS.get(name)


def list_scenarios() -> List[str]:
    """List all available scenario names."""
    return list(SCENARIOS.keys())


def get_homogeneous_scenarios() -> List[ScenarioConfig]:
    """Get all homogeneous scenarios."""
    return [s for s in SCENARIOS.values() if s.scenario_type == ScenarioType.HOMOGENEOUS]


def get_heterogeneous_scenarios() -> List[ScenarioConfig]:
    """Get all heterogeneous scenarios."""
    return [s for s in SCENARIOS.values() if s.scenario_type == ScenarioType.HETEROGENEOUS]


def create_custom_homogeneous_scenario(
    tp_degree: int,
    total_gpus: int = 8,
    num_runs: int = 3,
    warmup_requests: int = 5,
    max_concurrent_requests: int = 32
) -> ScenarioConfig:
    """
    Create a custom homogeneous scenario.
    
    Args:
        tp_degree: TP degree for all instances
        total_gpus: Total number of GPUs available
        num_runs: Number of benchmark runs
        warmup_requests: Number of warmup requests
        max_concurrent_requests: Maximum concurrent requests
        
    Returns:
        ScenarioConfig for the homogeneous configuration
    """
    if total_gpus % tp_degree != 0:
        raise ValueError(f"Cannot evenly divide {total_gpus} GPUs with TP={tp_degree}")
    
    num_instances = total_gpus // tp_degree
    
    instances = [
        InstanceAllocation(
            f"custom_homo_tp{tp_degree}_{i}",
            tp_degree,
            list(range(i * tp_degree, (i + 1) * tp_degree)),
            f"TP={tp_degree} instance {i}"
        )
        for i in range(num_instances)
    ]
    
    return ScenarioConfig(
        name=f"custom_homogeneous_tp{tp_degree}",
        scenario_type=ScenarioType.HOMOGENEOUS,
        description=f"Custom homogeneous: {num_instances} instances with TP={tp_degree}",
        instances=instances,
        num_runs=num_runs,
        warmup_requests=warmup_requests,
        max_concurrent_requests=max_concurrent_requests,
        use_length_routing=False
    )


def create_custom_heterogeneous_scenario(
    instance_configs: List[Dict[str, Any]],
    name: str = "custom_heterogeneous",
    description: str = "Custom heterogeneous configuration",
    length_thresholds: Optional[Dict[str, int]] = None,
    routing_rules: Optional[Dict[str, List[int]]] = None,
    num_runs: int = 3,
    warmup_requests: int = 5,
    max_concurrent_requests: int = 32
) -> ScenarioConfig:
    """
    Create a custom heterogeneous scenario.
    
    Args:
        instance_configs: List of instance configurations
            Each dict should have: tp, gpus, (optional) instance_id
        name: Scenario name
        description: Scenario description
        length_thresholds: Custom length thresholds
        routing_rules: Custom routing rules
        num_runs: Number of benchmark runs
        warmup_requests: Number of warmup requests
        max_concurrent_requests: Maximum concurrent requests
        
    Returns:
        ScenarioConfig for the heterogeneous configuration
    """
    instances = []
    for idx, cfg in enumerate(instance_configs):
        tp = cfg.get("tp", 1)
        gpus = cfg.get("gpus", [idx])
        instance_id = cfg.get("instance_id", f"custom_hetero_{idx}")
        
        instances.append(InstanceAllocation(
            instance_id=instance_id,
            tp_degree=tp,
            gpu_ids=gpus,
            description=f"TP={tp} on GPUs {gpus}"
        ))
    
    return ScenarioConfig(
        name=name,
        scenario_type=ScenarioType.HETEROGENEOUS,
        description=description,
        instances=instances,
        num_runs=num_runs,
        warmup_requests=warmup_requests,
        max_concurrent_requests=max_concurrent_requests,
        use_length_routing=True,
        length_thresholds=length_thresholds or {"short": 256, "medium": 512, "long": 1024},
        routing_rules=routing_rules or {}
    )


def create_scenario_from_config(
    config: Dict[str, Any],
    scenario_name: str
) -> Optional[ScenarioConfig]:
    """
    Create a scenario from configuration dictionary.
    
    Args:
        config: Full configuration dictionary
        scenario_name: Name of scenario to create
        
    Returns:
        ScenarioConfig or None
    """
    tp_configs = config.get("tp_configs", {})
    benchmark_config = config.get("benchmark", {})
    scheduling_config = config.get("scheduling", {})
    
    # Check if it's a predefined scenario
    if scenario_name in SCENARIOS:
        scenario = SCENARIOS[scenario_name]
        # Update with config values
        scenario.num_runs = benchmark_config.get("num_runs", scenario.num_runs)
        scenario.warmup_requests = benchmark_config.get("warmup_requests", scenario.warmup_requests)
        scenario.max_concurrent_requests = benchmark_config.get("max_concurrent_requests", scenario.max_concurrent_requests)
        return scenario
    
    # Check for homogeneous configuration
    if scenario_name.startswith("homogeneous_tp"):
        try:
            tp = int(scenario_name.split("tp")[1])
            return create_custom_homogeneous_scenario(
                tp_degree=tp,
                total_gpus=config.get("gpu", {}).get("total_gpus", 8),
                num_runs=benchmark_config.get("num_runs", 3),
                warmup_requests=benchmark_config.get("warmup_requests", 5),
                max_concurrent_requests=benchmark_config.get("max_concurrent_requests", 32)
            )
        except (ValueError, IndexError):
            pass
    
    # Check for heterogeneous in config
    if scenario_name == "heterogeneous" or scenario_name.startswith("heterogeneous_"):
        hetero_config = tp_configs.get("heterogeneous", [])
        if hetero_config:
            return create_custom_heterogeneous_scenario(
                instance_configs=hetero_config,
                name=scenario_name,
                description="Heterogeneous from config",
                length_thresholds=scheduling_config.get("length_thresholds"),
                routing_rules=scheduling_config.get("routing_rules"),
                num_runs=benchmark_config.get("num_runs", 3),
                warmup_requests=benchmark_config.get("warmup_requests", 5),
                max_concurrent_requests=benchmark_config.get("max_concurrent_requests", 32)
            )
    
    return None


def print_scenario_info(scenario: ScenarioConfig) -> None:
    """Print detailed scenario information."""
    print("\n" + "=" * 60)
    print(f"Scenario: {scenario.name}")
    print("=" * 60)
    print(f"Type: {scenario.scenario_type.value}")
    print(f"Description: {scenario.description}")
    print(f"\nInstances ({len(scenario.instances)}):")
    
    for inst in scenario.instances:
        print(f"  - {inst.instance_id}: TP={inst.tp_degree}, GPUs={inst.gpu_ids}")
        if inst.description:
            print(f"    {inst.description}")
    
    print(f"\nTotal GPUs: {scenario.get_total_gpus()}")
    print(f"TP Distribution: {scenario.get_tp_distribution()}")
    
    print(f"\nBenchmark Settings:")
    print(f"  Runs: {scenario.num_runs}")
    print(f"  Warmup Requests: {scenario.warmup_requests}")
    print(f"  Max Concurrent: {scenario.max_concurrent_requests}")
    
    if scenario.use_length_routing:
        print(f"\nLength-based Routing: Enabled")
        print(f"  Thresholds: {scenario.length_thresholds}")
        print(f"  Rules: {scenario.routing_rules}")
    
    print("=" * 60)
