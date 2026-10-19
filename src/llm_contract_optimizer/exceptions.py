class OptimizerError(Exception):
    """Base exception for expected project failures."""


class ConfigurationError(OptimizerError):
    """Raised when configuration is missing or inconsistent."""


class DataValidationError(OptimizerError):
    """Raised when an input file violates the solver schema."""


class NoFeasibleSolutionError(OptimizerError):
    """Raised when the model is proved infeasible by constraints or a solver."""


class SolveError(OptimizerError):
    """Raised when an optimization backend cannot resolve a required problem."""
