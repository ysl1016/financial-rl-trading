# Financial RL Trading Model

This repository contains a Reinforcement Learning based Financial Trading Model implemented in Python. The model uses various technical indicators and a custom gym environment to train an RL agent for trading decisions.

## Features

* Custom OpenAI Gym environment for trading
* Multiple technical indicators implementation
* Risk management with stop-loss
* Transaction costs and slippage simulation
* Sharpe ratio based reward function
* GRPO (Generalized Reward-Penalty Optimization) agent implementation
* Comprehensive evaluation metrics
* Advanced testing & optimization tools
* Benchmarking framework for strategy comparison
* DeepSeek-R1 transformer-based model integration

## Installation

```bash
pip install -r requirements.txt
```

## Package Structure

```
financial-rl-trading/
├── src/
│   ├── data/           # Data processing modules
│   ├── models/         # Trading environment and RL agents
│   ├── utils/          # Utility functions and analysis tools
│   └── tests/          # Testing framework
├── examples/           # Example scripts
├── docs/               # Documentation
├── TESTING_GUIDE.md    # Guide for testing and optimization
└── requirements.txt
```

## Documentation

* [API Documentation](docs/api_documentation.md) - Detailed API usage guide for all components
* [Testing Guide](TESTING_GUIDE.md) - Guide for testing and optimization

## Basic Usage

### Data Processing

```python
from src.data.data_processor import process_data

# Download and process stock data
data = process_data('SPY', start_date='2020-01-01')
```

### Setting Up Trading Environment

```python
from src.models.trading_env import TradingEnv

# Create trading environment
env = TradingEnv(
    data=data,
    initial_capital=100000,
    trading_cost=0.0005,
    slippage=0.0001
)
```

### Creating and Training GRPO Agent

```python
from src.models.grpo_agent import GRPOAgent

# Create agent
agent = GRPOAgent(
    state_dim=env.observation_space.shape[0],
    action_dim=env.action_space.n,
    hidden_dim=128,
    lr=3e-4
)

# Basic training loop
for episode in range(100):
    state = env.reset()
    done = False
    
    while not done:
        action = agent.select_action(state)
        next_state, reward, done, info = env.step(action)
        agent.store_transition(state, action, reward, next_state, done)
        
        # Update every 10 steps
        if episode % 10 == 0:
            agent.update()
        
        state = next_state
```

### Using DeepSeek-R1 GRPO Agent

```python
from src.models.deepseek_grpo_agent import DeepSeekGRPOAgent

# Create DeepSeek-R1 based GRPO agent
agent = DeepSeekGRPOAgent(
    state_dim=env.observation_space.shape[0],
    action_dim=env.action_space.n,
    seq_length=30,  # For temporal context
    hidden_dim=256,
    lr=3e-4
)

# See API documentation for full usage details
```

### Using Pre-built Examples

```bash
# Basic example of environment usage
python examples/trading_example.py

# Training with GRPO
python examples/train_grpo.py

# Hyperparameter optimization and benchmarking
python examples/optimize_and_benchmark.py --symbol SPY --optimize --train --benchmark
```

## GRPO Model Details

The GRPO (Generalized Reward-Penalty Optimization) agent implements a policy optimization algorithm with the following features:

* Actor-Critic architecture with shared network layers
* Generalized Advantage Estimation (GAE)
* Proximal Policy Optimization (PPO) clipping
* Entropy regularization for exploration
* Value function loss with coefficient
* Gradient clipping for stability

The model uses the following hyperparameters by default:
* Learning rate: 3e-4
* Discount factor (gamma): 0.99
* GAE lambda: 0.95
* PPO clip ratio: 0.2
* Entropy coefficient: 0.01
* Value function coefficient: 0.5
* Max gradient norm: 0.5

## Advanced Features

### Testing and Optimization

The project includes comprehensive testing and optimization tools:

```bash
# Run all tests
python src/tests/run_tests.py --type all

# Optimize hyperparameters
python examples/optimize_and_benchmark.py --symbol SPY --optimize --n_iter 30

# Benchmark against traditional strategies
python examples/optimize_and_benchmark.py --symbol SPY --benchmark
```

For detailed information on testing and optimization, see [TESTING_GUIDE.md](TESTING_GUIDE.md).

### Hyperparameter Optimization

The project includes Bayesian Optimization for finding the best hyperparameters:

```python
from src.utils.hyperparameter_optimization import HyperparameterOptimizer

optimizer = HyperparameterOptimizer(train_data, val_data)
result = optimizer.optimize(n_iter=30)
```

### Strategy Benchmarking

Compare GRPO agent against traditional trading strategies:

```python
from src.utils.benchmarking import StrategyBenchmark

benchmark = StrategyBenchmark(test_data)
benchmark.create_standard_strategies()  # Add Buy & Hold, MA Crossover, RSI, etc.
benchmark.add_grpo_agent(agent, name="GRPO")
results = benchmark.run_all_benchmarks()
benchmark.plot_results()
```

## Training Process

The training process includes:

1. Data collection using the trading environment
2. Advantage estimation using GAE
3. Policy and value function updates
4. Regular evaluation of agent performance
5. Model checkpointing
6. Performance visualization

## Performance Metrics

The toolkit includes various performance metrics:

* **Profitability Metrics**: Total return, annualized return, win rate
* **Risk-Adjusted Metrics**: Sharpe ratio, Sortino ratio, Calmar ratio
* **Risk Metrics**: Maximum drawdown, volatility
* **Statistical Significance Testing**: t-test with effect size calculation

## License

MIT