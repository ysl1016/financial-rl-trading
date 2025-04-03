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

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Example
```python
from src.models.trading_env import TradingEnv
from src.data.data_processor import process_data

# Load and process data
data = process_data('SPY')  # Replace with your data

# Create environment
env = TradingEnv(data)

# Use the environment for training your RL agent
```

### Training with GRPO
```python
from src.models.grpo_agent import GRPOAgent

# Create GRPO agent
agent = GRPOAgent(
    state_dim=env.observation_space.shape[0],
    action_dim=env.action_space.n,
    hidden_dim=128,
    lr=3e-4
)

# Train the agent
python examples/train_grpo.py
```

## Project Structure

* `src/data/`: Data processing and handling
* `src/models/`: Trading environment and GRPO agent implementation
* `src/utils/`: Technical indicators and utility functions
* `examples/`: Usage examples
  * `trading_example.py`: Basic environment usage
  * `train_grpo.py`: GRPO agent training

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

## Training Process

The training process includes:

1. Data collection using the trading environment
2. Advantage estimation using GAE
3. Policy and value function updates
4. Regular evaluation of agent performance
5. Model checkpointing
6. Performance visualization

## License

MIT