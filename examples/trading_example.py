import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.data_processor import process_data
from src.models.trading_env import TradingEnv
import matplotlib.pyplot as plt
import numpy as np

def main():
    # Download and process data
    data = process_data('SPY', start_date='2020-01-01')
    
    # Create environment
    env = TradingEnv(
        data=data,
        initial_capital=100000,
        trading_cost=0.0005,
        slippage=0.0001,
        risk_free_rate=0.02,
        max_position_size=1.0,
        stop_loss_pct=0.02
    )
    
    # Run simple example
    n_episodes = 1
    for episode in range(n_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            # Random action for demonstration
            action = env.action_space.sample()
            
            # Take step
            next_state, reward, done, info = env.step(action)
            total_reward += reward
            
            if done:
                print(f"Episode {episode + 1}/{n_episodes}")
                print(f"Final portfolio value: ${info['portfolio_value']:.2f}")
                print(f"Total reward: {total_reward:.2f}")
                print(f"Number of trades: {len(info['trades'])}\n")
                
                # Plot portfolio value
                plt.figure(figsize=(15, 5))
                plt.plot(env.portfolio_values)
                plt.title('Portfolio Value Over Time')
                plt.xlabel('Time Step')
                plt.ylabel('Portfolio Value ($)')
                plt.grid(True)
                plt.show()

if __name__ == '__main__':
    main()
