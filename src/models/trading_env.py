import gym
import numpy as np
from gym import spaces

class TradingEnv(gym.Env):
    """
    A custom OpenAI Gym environment for financial trading
    """
    
    def __init__(self, data, initial_capital=100000, trading_cost=0.0005, slippage=0.0001,
                 risk_free_rate=0.02, max_position_size=1.0, stop_loss_pct=0.02):
        super().__init__()
        self.data = data.reset_index(drop=True)
        self.n_steps = len(data)
        
        # Determine available feature columns dynamically
        self.feature_columns = [col for col in self.data.columns if col.endswith('_norm')]

        # Observation space: all features + position info + return info
        obs_dim = len(self.feature_columns) + 2  # position and portfolio return
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        
        # Action space: 0=hold, 1=buy, 2=sell
        self.action_space = spaces.Discrete(3)
        
        # Initialize parameters
        self.initial_capital = initial_capital
        self.trading_cost = trading_cost
        self.slippage = slippage
        self.risk_free_rate = risk_free_rate / 252  # Daily risk-free rate
        self.max_position_size = max_position_size
        self.stop_loss_pct = stop_loss_pct
        self.reset()
    
    def reset(self):
        """Reset the environment to initial state"""
        self.index = 0
        self.position = 0  # 0=no position, 1=long, -1=short
        self.cash = self.initial_capital
        self.holdings = 0
        self.last_price = float(self.data.loc[self.index, 'Close'])
        self.entry_price = self.last_price
        self.portfolio_values = [self.initial_capital]
        self.trades = []
        self.daily_returns = []
        self.position_duration = 0
        
        return self._get_observation()
    
    def _get_observation(self):
        """Get current state observation"""
        row = self.data.loc[self.index]

        # Calculate portfolio return
        portfolio_return = 0.0
        if len(self.portfolio_values) > 1:
            portfolio_return = (self.portfolio_values[-1] / self.portfolio_values[-2]) - 1

        features = [float(row[col]) for col in self.feature_columns]
        obs = np.array(features + [float(self.position), float(portfolio_return)], dtype=np.float32)

        return obs
    
    def _calculate_reward(self, old_value, new_value, action):
        """Calculate reward for the current step"""
        # Base return
        returns = (new_value / old_value) - 1
        
        # Add Sharpe Ratio component
        if len(self.daily_returns) > 0:
            returns_std = np.std(self.daily_returns) + 1e-9
            sharpe = (returns - self.risk_free_rate) / returns_std
        else:
            sharpe = 0
        
        # Position duration penalty
        holding_penalty = -0.0001 * self.position_duration if self.position != 0 else 0
        
        # Excessive trading penalty
        trading_penalty = -0.0002 if action != 0 else 0
        
        # Final reward
        reward = returns + 0.1 * sharpe + holding_penalty + trading_penalty
        
        return reward
    
    def step(self, action):
        """Execute one step in the environment"""
        # Get current price and portfolio value
        current_price = float(self.data.loc[self.index, 'Close'])
        old_portfolio_value = self.cash + (self.holdings * current_price)
        
        # Check stop loss
        if self.position != 0:
            price_change = (current_price - self.entry_price) / self.entry_price
            if (self.position == 1 and price_change < -self.stop_loss_pct) or \
               (self.position == -1 and price_change > self.stop_loss_pct):
                action = 2 if self.position == 1 else 1  # Close position
        
        # Apply slippage
        if action == 1:
            exec_price = current_price * (1 + self.slippage)
        elif action == 2:
            exec_price = current_price * (1 - self.slippage)
        else:
            exec_price = current_price
        
        # Execute trading action
        if action == 1:  # Buy
            if self.position == 0:
                self.position = 1
                self.holdings = self.max_position_size
                cost = exec_price * self.holdings
                self.cash -= cost * (1 + self.trading_cost)
                self.entry_price = exec_price
                self.trades.append(('buy', self.index, exec_price))
                self.position_duration = 0
            elif self.position == -1:
                profit = (self.entry_price - exec_price) * abs(self.holdings)
                self.cash += profit
                self.cash -= abs(self.holdings) * exec_price * self.trading_cost
                self.trades.append(('close_short', self.index, exec_price))
                
                self.position = 1
                self.holdings = self.max_position_size
                cost = exec_price * self.holdings
                self.cash -= cost * (1 + self.trading_cost)
                self.entry_price = exec_price
                self.trades.append(('buy', self.index, exec_price))
                self.position_duration = 0
        
        elif action == 2:  # Sell
            if self.position == 0:
                self.position = -1
                self.holdings = -self.max_position_size
                proceed = exec_price * abs(self.holdings)
                self.cash += proceed * (1 - self.trading_cost)
                self.entry_price = exec_price
                self.trades.append(('short', self.index, exec_price))
                self.position_duration = 0
            elif self.position == 1:
                profit = (exec_price - self.entry_price) * self.holdings
                self.cash += profit
                self.cash -= self.holdings * exec_price * self.trading_cost
                self.trades.append(('close_long', self.index, exec_price))
                
                self.position = -1
                self.holdings = -self.max_position_size
                proceed = exec_price * abs(self.holdings)
                self.cash += proceed * (1 - self.trading_cost)
                self.entry_price = exec_price
                self.trades.append(('short', self.index, exec_price))
                self.position_duration = 0
        
        # Update state
        self.index += 1
        if self.position != 0:
            self.position_duration += 1
        
        # Calculate new portfolio value and return
        new_portfolio_value = self.cash + (self.holdings * current_price)
        self.portfolio_values.append(new_portfolio_value)
        daily_return = (new_portfolio_value / old_portfolio_value) - 1
        self.daily_returns.append(daily_return)
        
        # Calculate reward
        reward = self._calculate_reward(old_portfolio_value, new_portfolio_value, action)
        
        # Check if episode is done
        done = self.index >= self.n_steps - 1
        
        return self._get_observation(), reward, done, {
            'portfolio_value': new_portfolio_value,
            'position': self.position,
            'trades': self.trades
        }