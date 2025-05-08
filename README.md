# DeepSeek-R1 Based Financial Trading Reinforcement Learning Model

This project implements a reinforcement learning-based financial trading model utilizing the DeepSeek-R1 architecture. It integrates transformer-based time series processing with the GRPO (Generalized Reward-Penalty Optimization) algorithm to deliver superior trading performance.

## Key Features

* DeepSeek-R1 based transformer architecture
* Positional encoding and attention mechanisms for time series data
* Enhanced GRPO algorithm (KL divergence constraints, entropy regularization, etc.)
* Distributional value network for uncertainty modeling
* Market regime detection and adaptive parameter adjustment
* Comprehensive backtesting framework and performance analysis tools

## Installation

```bash
pip install -r requirements.txt
```

## Package Structure

```
financial-rl-trading/
├── src/
│   ├── data/           # Data processing modules
│   ├── models/         # Reinforcement learning models
│   └── utils/          # Utility functions
├── examples/           # Example scripts
└── requirements.txt
```

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

### Creating DeepSeek-R1 GRPO Agent

```python
from src.models.deepseek_grpo_agent import DeepSeekGRPOAgent

# Create agent
agent = DeepSeekGRPOAgent(
    state_dim=env.observation_space.shape[0],
    action_dim=env.action_space.n,
    seq_length=20,
    hidden_dim=256
)
```

### Training the Model

```python
# Use examples/train_deepseek_grpo.py script
python examples/train_deepseek_grpo.py --symbol SPY --start_date 2018-01-01 --end_date 2022-12-31
```

### Backtesting

```python
# Use examples/backtest_deepseek_grpo.py script
python examples/backtest_deepseek_grpo.py --model_path models/deepseek_grpo_model.pt --symbol SPY --start_date 2023-01-01
```

## DeepSeek-R1 GRPO Model Characteristics

### Transformer-Based Architecture

- **Positional Encoding**: Preserves temporal information in time series data
- **Multi-Head Attention**: Parallel processing of patterns across various time scales
- **Feature Attention**: Models relationships between different technical indicators

### Enhanced GRPO Algorithm

- **Reward-Penalty Separation**: Differentiated learning for positive/negative advantages
- **KL Divergence Constraints**: Improved policy update stability
- **Entropy Regularization**: Optimization of exploration-exploitation balance

### Market Adaptation Mechanisms

- **Market Regime Detection**: Recognition of various market conditions
- **Dynamic Parameter Adjustment**: Agent behavior adjustment based on market conditions
- **Exploration Temperature Control**: Exploration level adjustment according to market uncertainty

## Backtesting Framework

### Key Features

- **Testing in Various Market Conditions**: Normal markets, bull markets, bear markets, high volatility, etc.
- **Market Shock Simulation**: Robustness testing in extreme market conditions
- **Monte Carlo Simulation**: Probabilistic performance distribution analysis
- **Benchmark Comparison**: Comparison with various strategies such as Buy-and-Hold

### Performance Metrics

- **Profitability Metrics**: Total return, annualized return, win rate, etc.
- **Risk-Adjusted Metrics**: Sharpe ratio, Sortino ratio, Calmar ratio, etc.
- **Risk Metrics**: Maximum drawdown, VaR, CVaR, etc.
- **Statistical Significance Testing**: t-test, Wilcoxon test, etc.

### Visualization Tools

- **Portfolio Performance Visualization**: Cumulative returns, daily returns, drawdowns, etc.
- **Return Distribution Analysis**: Histograms, QQ plots, box plots, etc.
- **Rolling Metrics**: Rolling returns, volatility, Sharpe ratio, etc.
- **Regime-Based Performance Analysis**: Performance comparison by market regime

## License

MIT