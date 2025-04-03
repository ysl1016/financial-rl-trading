import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical

class GRPONetwork(nn.Module):
    """
    Pure policy network for GRPO without value function
    """
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(GRPONetwork, self).__init__()
        
        # Policy network (no value function)
        self.policy = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Direct state-action value estimator
        self.q_estimator = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state):
        """Returns action probabilities only"""
        return self.policy(state)
    
    def estimate_q_value(self, state, action_onehot):
        """Estimate Q-value for state-action pair"""
        sa_pair = torch.cat([state, action_onehot], dim=-1)
        return self.q_estimator(sa_pair)

class GRPOAgent:
    def __init__(self, state_dim, action_dim, hidden_dim=128, lr=3e-4, gamma=0.99,
                 reward_scale=1.0, penalty_scale=0.5, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.gamma = gamma
        self.action_dim = action_dim
        self.reward_scale = reward_scale
        self.penalty_scale = penalty_scale
        
        # Initialize network
        self.network = GRPONetwork(state_dim, action_dim, hidden_dim).to(device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        
        # Experience buffer
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
    
    def select_action(self, state):
        """Select action using the current policy"""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action_probs = self.network(state)
            dist = Categorical(action_probs)
            action = dist.sample()
        
        return action.item()
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in buffer"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
    
    def _to_onehot(self, actions):
        """Convert actions to one-hot vectors"""
        actions = torch.LongTensor(actions).to(self.device)
        onehot = torch.zeros(len(actions), self.action_dim).to(self.device)
        onehot.scatter_(1, actions.unsqueeze(1), 1)
        return onehot
    
    def update(self):
        """Update policy using GRPO algorithm"""
        if len(self.states) == 0:
            return {}
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        next_states = torch.FloatTensor(np.array(self.next_states)).to(self.device)
        actions = torch.LongTensor(self.actions).to(self.device)
        rewards = torch.FloatTensor(self.rewards).to(self.device)
        dones = torch.FloatTensor(self.dones).to(self.device)
        
        # Get action probabilities
        action_probs = self.network(states)
        dist = Categorical(action_probs)
        
        # Convert actions to one-hot for Q-value estimation
        actions_onehot = self._to_onehot(self.actions)
        
        # Estimate Q-values
        current_q = self.network.estimate_q_value(states, actions_onehot).squeeze()
        
        # Calculate next state maximum Q-values
        with torch.no_grad():
            next_action_probs = self.network(next_states)
            next_actions_onehot = torch.eye(self.action_dim).to(self.device)
            next_q_values = []
            
            for i in range(self.action_dim):
                next_action_onehot = next_actions_onehot[i].expand(len(next_states), -1)
                q = self.network.estimate_q_value(next_states, next_action_onehot).squeeze()
                next_q_values.append(q)
            
            next_q_values = torch.stack(next_q_values, dim=1)
            next_q = (next_q_values * next_action_probs).sum(dim=1)
            
            # Calculate target Q-values
            target_q = rewards + self.gamma * next_q * (1 - dones)
        
        # Calculate advantages (Q-value differences)
        advantages = target_q - current_q
        
        # GRPO loss calculation
        # Positive advantages lead to reward, negative to penalty
        positive_mask = advantages > 0
        negative_mask = ~positive_mask
        
        # Policy loss with reward-penalty separation
        log_probs = dist.log_prob(actions)
        reward_loss = -log_probs[positive_mask] * advantages[positive_mask] * self.reward_scale
        penalty_loss = log_probs[negative_mask] * advantages[negative_mask].abs() * self.penalty_scale
        
        policy_loss = (reward_loss.mean() + penalty_loss.mean()) if len(reward_loss) > 0 and len(penalty_loss) > 0 \
                     else (reward_loss.mean() if len(reward_loss) > 0 else penalty_loss.mean())
        
        # Q-value estimation loss
        q_loss = (current_q - target_q).pow(2).mean()
        
        # Total loss
        total_loss = policy_loss + q_loss
        
        # Update network
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        # Clear buffer
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        
        return {
            'policy_loss': policy_loss.item(),
            'q_loss': q_loss.item(),
            'mean_reward': rewards.mean().item(),
            'mean_advantage': advantages.mean().item()
        }
    
    def save(self, path):
        """Save model to file"""
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
    
    def load(self, path):
        """Load model from file"""
        checkpoint = torch.load(path)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])