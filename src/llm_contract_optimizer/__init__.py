"""Energy-aware contract and workload optimization."""

from llm_contract_optimizer.config import AppConfig, load_config
from llm_contract_optimizer.domain import OptimizationResult
from llm_contract_optimizer.solve import solve_case

__all__ = ["AppConfig", "OptimizationResult", "load_config", "solve_case"]
__version__ = "0.5.0"
