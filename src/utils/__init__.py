from .indicators import calculate_technical_indicators

# Optional utilities are imported lazily to avoid hard dependencies during tests
__all__ = ['calculate_technical_indicators']

try:
    from .hyperparameter_optimization import (
        HyperparameterOptimizer, ValidationFramework, SensitivityAnalysis,
        run_hyperparameter_optimization
    )
    __all__ += [
        'HyperparameterOptimizer',
        'ValidationFramework',
        'SensitivityAnalysis',
        'run_hyperparameter_optimization'
    ]
except Exception:  # pragma: no cover - optional dependency
    pass

try:
    from .benchmarking import (
        BaseStrategy, BuyAndHoldStrategy, RandomStrategy, MovingAverageCrossoverStrategy,
        RSIStrategy, GRPOStrategy, StrategyBenchmark, run_strategy_benchmark
    )
    __all__ += [
        'BaseStrategy', 'BuyAndHoldStrategy', 'RandomStrategy',
        'MovingAverageCrossoverStrategy', 'RSIStrategy', 'GRPOStrategy',
        'StrategyBenchmark', 'run_strategy_benchmark'
    ]
except Exception:  # pragma: no cover - optional dependency
    pass