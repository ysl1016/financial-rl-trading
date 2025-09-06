import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.data_processor import process_data
from src.models.trading_env import TradingEnv
from src.models.grpo_agent import GRPOAgent
import numpy as np
import torch
import matplotlib.pyplot as plt

def train_grpo(env, agent, num_epochs=50, steps_per_epoch=1000, eval_episodes=10,
               update_interval=1000):
    """Train the GRPO agent"""
    # Training metrics
    epoch_rewards = []
    eval_rewards = []
    
    for epoch in range(num_epochs):
        episode_rewards = []
        episode_reward = 0
        state = env.reset()
        
        # Collect experience
        for step in range(steps_per_epoch):
            # Select action
            action = agent.select_action(state)
            
            # Take step in environment
            next_state, reward, done, info = env.step(action)
            
            # Store transition
            agent.store_transition(state, action, reward, next_state, done)
            
            episode_reward += reward
            
            if done:
                episode_rewards.append(episode_reward)
                state = env.reset()
                episode_reward = 0
            else:
                state = next_state
            
            # Update policy
            if (step + 1) % update_interval == 0:
                metrics = agent.update()
                if metrics:
                    print(f"Step {step + 1}, Policy Loss: {metrics['policy_loss']:.4f}, "
                          f"Q Loss: {metrics['q_loss']:.4f}, "
                          f"Mean Reward: {metrics['mean_reward']:.4f}, "
                          f"Mean Advantage: {metrics['mean_advantage']:.4f}")
        
        # Evaluate agent
        eval_reward = evaluate_agent(env, agent, eval_episodes)
        
        # Store metrics
        epoch_rewards.append(np.mean(episode_rewards) if episode_rewards else 0)
        eval_rewards.append(eval_reward)
        
        # Print progress
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print(f"Train Reward: {epoch_rewards[-1]:.2f}, Eval Reward: {eval_reward:.2f}")
        
        # Save model periodically
        if (epoch + 1) % 10 == 0:
            agent.save(f"grpo_model_epoch_{epoch+1}.pt")
    
    return epoch_rewards, eval_rewards

def evaluate_agent(env, agent, num_episodes):
    """Evaluate the agent's performance"""
    total_rewards = []
    
    for _ in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            state = next_state
        
        total_rewards.append(total_reward)
    
    return np.mean(total_rewards)

def plot_results(train_rewards, eval_rewards):
    """Plot training results"""
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_rewards, label='Train')
    plt.plot(eval_rewards, label='Eval')
    plt.title('Rewards over Training')
    plt.xlabel('Epoch')
    plt.ylabel('Average Reward')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(np.maximum.accumulate(eval_rewards), label='Best Eval')
    plt.title('Best Evaluation Reward')
    plt.xlabel('Epoch')
    plt.ylabel('Reward')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def main():
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Download and process data
    data_splits = process_data('SPY', start_date='2020-01-01')
    data = data_splits['train']
    
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
    
    # Create GRPO agent
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = GRPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=128,
        lr=3e-4,
        gamma=0.99,
        reward_scale=1.0,
        penalty_scale=0.5
    )
    
    # Train agent
    train_rewards, eval_rewards = train_grpo(
        env=env,
        agent=agent,
        num_epochs=50,
        steps_per_epoch=1000,
        eval_episodes=10,
        update_interval=1000
    )
    
    # Plot results
    plot_results(train_rewards, eval_rewards)
    
    # Save final model
    agent.save("grpo_model_final.pt")

if __name__ == '__main__':
    main()