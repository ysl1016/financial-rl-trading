# Financial RL Trading Model

This repository contains a Reinforcement Learning based Financial Trading Model implemented in Python. The model uses various technical indicators and a custom gym environment to train an RL agent for trading decisions.

## Features

- Custom OpenAI Gym environment for trading
- Multiple technical indicators implementation
- Risk management with stop-loss
- Transaction costs and slippage simulation
- Sharpe ratio based reward function

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Check the examples directory for usage examples:

```python
from src.models.trading_env import TradingEnv
from src.data.data_processor import process_data

# Load and process data
data = process_data('SPY')  # Replace with your data

# Create environment
env = TradingEnv(data)

# Use the environment for training your RL agent
```

## Project Structure

- `src/data/`: Data processing and handling
- `src/models/`: Trading environment implementation
- `src/utils/`: Technical indicators and utility functions
- `examples/`: Usage examples

## License

MIT