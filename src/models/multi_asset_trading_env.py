import gym
import numpy as np
import pandas as pd
from gym import spaces
from typing import Dict, List, Tuple, Optional, Union

class MultiAssetTradingEnv(gym.Env):
    """
    A custom OpenAI Gym environment for multi-asset financial trading
    
    This environment extends the single asset TradingEnv to support multiple assets
    with continuous action space for portfolio allocation.
    """
    
    def __init__(self, 
                 asset_data: Dict[str, pd.DataFrame], 
                 initial_capital: float = 100000, 
                 trading_cost: float = 0.0005, 
                 slippage: float = 0.0001,
                 risk_free_rate: float = 0.02, 
                 max_position_size: float = 1.0, 
                 stop_loss_pct: float = 0.02,
                 max_asset_weight: float = 0.5,
                 correlation_lookback: int = 30):
        """
        Initialize the multi-asset trading environment
        
        Args:
            asset_data: Dictionary of DataFrames containing OHLCV and indicator data for each asset
            initial_capital: Initial portfolio capital
            trading_cost: Transaction cost as a fraction of trade value
            slippage: Price slippage as a fraction of price
            risk_free_rate: Annual risk-free rate (used for Sharpe calculation)
            max_position_size: Maximum allowed position size multiplier (e.g., leverage)
            stop_loss_pct: Stop loss percentage (0.02 = 2% loss triggers exit)
            max_asset_weight: Maximum weight for a single asset (diversification)
            correlation_lookback: Number of days for correlation calculation
        """
        super().__init__()
        
        # Store parameters
        self.asset_data = asset_data
        self.assets = list(asset_data.keys())
        self.num_assets = len(self.assets)
        self.initial_capital = initial_capital
        self.trading_cost = trading_cost
        self.slippage = slippage
        self.risk_free_rate = risk_free_rate / 252  # Daily risk-free rate
        self.max_position_size = max_position_size
        self.stop_loss_pct = stop_loss_pct
        self.max_asset_weight = max_asset_weight
        self.correlation_lookback = correlation_lookback
        
        # Align all asset data to the same date range
        self._align_asset_data()
        
        # Calculate number of time steps
        self.n_steps = min([len(data) for data in self.asset_data.values()])
        
        # Define action and observation spaces
        # Continuous action space: portfolio weights for each asset (-1 = max short, 1 = max long)
        self.action_space = spaces.Box(
            low=-1, 
            high=1, 
            shape=(self.num_assets,), 
            dtype=np.float32
        )
        
        # Calculate observation space dimensions
        # For each asset: technical indicators + asset-specific features
        # Plus portfolio state features
        self.indicators_per_asset = 12  # Number of normalized indicators per asset
        asset_features = self.num_assets * self.indicators_per_asset
        portfolio_features = self.num_assets + 2  # Current weights, portfolio return, cash ratio
        correlation_features = self.num_assets * (self.num_assets - 1) // 2  # Upper triangular correlation matrix
        
        total_features = asset_features + portfolio_features + correlation_features
        
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(total_features,), 
            dtype=np.float32
        )
        
        # Initialize state
        self.reset()
    
    def _align_asset_data(self):
        """Align all asset data to the same date range"""
        # Find common date range
        all_dates = set()
        for asset, data in self.asset_data.items():
            if not data.index.is_all_dates:
                data.index = pd.to_datetime(data.index)
            all_dates = all_dates.union(set(data.index)) if not all_dates else all_dates.intersection(set(data.index))
        
        common_dates = sorted(list(all_dates))
        
        # Reindex all dataframes to common dates
        for asset in self.assets:
            self.asset_data[asset] = self.asset_data[asset].reindex(common_dates)
            
        # Forward fill missing values
        for asset in self.assets:
            self.asset_data[asset] = self.asset_data[asset].ffill()
    
    def reset(self):
        """Reset the environment to initial state"""
        self.index = 0
        self.cash = self.initial_capital
        
        # Portfolio holdings and weights
        self.holdings = {asset: 0.0 for asset in self.assets}
        self.weights = {asset: 0.0 for asset in self.assets}
        
        # Track prices and entry prices
        self.last_prices = {asset: float(self.asset_data[asset].iloc[self.index]['Close']) 
                            for asset in self.assets}
        self.entry_prices = {asset: self.last_prices[asset] for asset in self.assets}
        
        # Portfolio metrics
        self.portfolio_value = self.initial_capital
        self.portfolio_values = [self.initial_capital]
        self.asset_values = {asset: 0.0 for asset in self.assets}
        self.total_values = {asset: [0.0] for asset in self.assets}
        
        # Trading history
        self.trades = []
        self.daily_returns = []
        self.position_durations = {asset: 0 for asset in self.assets}
        
        # Calculate initial correlation matrix
        self._update_correlation_matrix()
        
        return self._get_observation()
    
    def _update_correlation_matrix(self):
        """Update the correlation matrix of asset returns"""
        start_idx = max(0, self.index - self.correlation_lookback)
        end_idx = self.index + 1
        
        returns_data = {}
        for asset in self.assets:
            asset_data = self.asset_data[asset].iloc[start_idx:end_idx]
            returns_data[asset] = asset_data['Close'].pct_change().dropna()
        
        returns_df = pd.DataFrame(returns_data)
        
        # Handle case when not enough data
        if len(returns_df) <= 1:
            self.correlation_matrix = np.eye(self.num_assets)
            return
        
        # Calculate correlation matrix
        self.correlation_matrix = returns_df.corr().values
        
        # Fill NaNs with 0
        self.correlation_matrix = np.nan_to_num(self.correlation_matrix)
    
    def _get_observation(self):
        """Get current state observation"""
        # Collect technical indicators for each asset
        asset_features = []
        for asset in self.assets:
            row = self.asset_data[asset].iloc[self.index]
            
            # Extract normalized indicators
            indicators = [
                float(row['RSI_norm']),
                float(row['ForceIndex2_norm']),
                float(row['%K_norm']),
                float(row['%D_norm']),
                float(row['MACD_norm']),
                float(row['MACDSignal_norm']),
                float(row['BBWidth_norm']),
                float(row['ATR_norm']),
                float(row['VPT_norm']),
                float(row['VPT_MA_norm']),
                float(row['OBV_norm']),
                float(row['ROC_norm'])
            ]
            asset_features.extend(indicators)
        
        # Portfolio state features
        current_weights = [float(self.weights[asset]) for asset in self.assets]
        
        # Portfolio return
        portfolio_return = 0.0
        if len(self.portfolio_values) > 1:
            portfolio_return = (self.portfolio_values[-1] / self.portfolio_values[-2]) - 1
        
        # Cash ratio
        cash_ratio = self.cash / self.portfolio_value if self.portfolio_value > 0 else 1.0
        
        portfolio_features = current_weights + [float(portfolio_return), float(cash_ratio)]
        
        # Correlation matrix (upper triangular only to avoid redundancy)
        correlation_features = []
        for i in range(self.num_assets):
            for j in range(i+1, self.num_assets):
                correlation_features.append(float(self.correlation_matrix[i, j]))
        
        # Combine all features
        obs = np.array(asset_features + portfolio_features + correlation_features, dtype=np.float32)
        
        return obs
    
    def _normalize_weights(self, weights):
        """
        Normalize weights to comply with constraints
        
        Args:
            weights: Raw portfolio weights from the action
            
        Returns:
            Normalized weights that comply with position limits
        """
        # Apply max asset weight constraint
        weights = np.clip(weights, -self.max_asset_weight, self.max_asset_weight)
        
        # Apply max position size (leverage) constraint
        total_abs_weight = np.