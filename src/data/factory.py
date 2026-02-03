"""
Adapter Factory.

Dynamic adapter selection based on configuration.
Enables dataset-agnostic training pipeline.
"""

from typing import Dict, Any

from .adapters.base import RiskDataAdapter
from .adapters.uci_credit import UCICreditAdapter
from .adapters.home_credit import HomeCreditAdapter
from .adapters.ieee_cis import IEEECISAdapter
from .adapters.lending_club import LendingClubAdapter


# Registry of available adapters
ADAPTER_REGISTRY: Dict[str, type] = {
    "uci_credit": UCICreditAdapter,
    "home_credit": HomeCreditAdapter,
    "ieee_cis": IEEECISAdapter,
    "lending_club": LendingClubAdapter,
}


def get_adapter(name: str, config: Dict[str, Any]) -> RiskDataAdapter:
    """
    Factory function to instantiate the appropriate adapter.
    
    Args:
        name: Adapter name (must be in ADAPTER_REGISTRY)
        config: Configuration dictionary for the adapter
        
    Returns:
        Instantiated RiskDataAdapter subclass
        
    Raises:
        ValueError: If adapter name is not recognized
        
    Example:
        >>> config = {"path": "data/raw/home_credit", "target_col": "TARGET"}
        >>> adapter = get_adapter("home_credit", config)
        >>> adapter.load_raw()
        >>> adapter.feature_engineer()
        >>> X, y = adapter.get_features_and_target()
    """
    if name not in ADAPTER_REGISTRY:
        available = ", ".join(ADAPTER_REGISTRY.keys())
        raise ValueError(
            f"Unknown adapter: '{name}'. Available adapters: {available}"
        )
    
    adapter_class = ADAPTER_REGISTRY[name]
    return adapter_class(config)


def list_adapters() -> list:
    """Return list of available adapter names."""
    return list(ADAPTER_REGISTRY.keys())


def register_adapter(name: str, adapter_class: type) -> None:
    """
    Register a custom adapter.
    
    Useful for adding new datasets without modifying core code.
    
    Args:
        name: Unique adapter name
        adapter_class: Class that inherits from RiskDataAdapter
    """
    if not issubclass(adapter_class, RiskDataAdapter):
        raise TypeError(
            f"Adapter must inherit from RiskDataAdapter, got {adapter_class}"
        )
    
    ADAPTER_REGISTRY[name] = adapter_class
