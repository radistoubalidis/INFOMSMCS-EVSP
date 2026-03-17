import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from pricing_env import EVSPPricingEnv, random_policy


class PolicyNet(nn.Module):
    def __init__(self, state_dim, n_actions, hidden_dim=256):  # FIXED order
        super().__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.n_actions = n_actions
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )
        nn.init.xavier_uniform_(self.net[-1].weight)

    def forward(self, state_tensor):
        logits = self.net(state_tensor)
        logits = torch.clamp(logits, min=-10.0, max=10.0)  # CHANGED: added clamp to prevent NaN
        return torch.log_softmax(logits, dim=-1)


def state_to_vec(state, env: EVSPPricingEnv):
    current_idx, visited_mask, cum_pi, cum_time_cost = state
    # CHANGED: removed visited_mask from state vector
    # 193-dim bool mask made state space too large for REINFORCE to learn
    curr_onehot = np.zeros(env.n_trips + 1, dtype=np.float32)
    curr_onehot[current_idx + 1] = 1.0
    scalars = np.array([cum_pi, cum_time_cost], dtype=np.float32)
    return np.concatenate([curr_onehot, scalars])  # 196 dims instead of 389


def run_episode(env: EVSPPricingEnv, policy_net, device='cpu'):
    log_probs = []
    rewards = []
    state = env.reset()
    done = False

    while not done:
        state_vector = state_to_vec(state, env)
        state_tensor = torch.tensor(state_vector, dtype=torch.float32, device=device)
        logpi_all = policy_net(state_tensor)

        actions = [a for a in env.get_actions() if a != "STOP"]
        if not actions:
            _, reward, done, info = env.step("STOP")
            rewards.append(reward)
            break

        indices = [env.trip_to_idx[a[0]] for a in actions]
        logpi_available = logpi_all[indices]
        probs = torch.exp(logpi_available)
        probs = probs / (probs.sum() + 1e-8)  # CHANGED: added epsilon to prevent 0/0 = NaN

        # CHANGED: added NaN guard → fallback to uniform if probs still invalid
        if torch.isnan(probs).any() or torch.isinf(probs).any():
            probs = torch.ones(len(actions), dtype=torch.float32) / len(actions)

        m = torch.distributions.Categorical(probs)
        idx = m.sample()
        chosen_action = actions[idx.item()]
        chosen_log_prob = logpi_available[idx]

        state, reward, done, info = env.step(chosen_action)
        log_probs.append(chosen_log_prob)
        rewards.append(reward)

        if done:
            break

    # Compute returns (gamma=1.0 since reward is only at terminal)
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + G
        returns.insert(0, G)

    return log_probs, returns, info


def reinforce_update(policy_net, optimizer, log_probs, returns):
    # CHANGED: was float16 → gradients were vanishing, policy never updated
    returns_tensor = torch.tensor(returns, dtype=torch.float32)

    # CHANGED: added normalization to handle large dual values
    if len(returns_tensor) > 1:
        returns_tensor = (returns_tensor - returns_tensor.mean()) / (returns_tensor.std() + 1e-8)

    loss = sum(-logp * G for logp, G in zip(log_probs, returns_tensor))
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)  # CHANGED: added grad clipping
    optimizer.step()
    return loss.item()


def rl_pricing(env: EVSPPricingEnv, policy_net: PolicyNet, optimizer, train=True, epsilon=0.05):
    if np.random.random() < epsilon:
        return random_policy(env)

    log_probs, returns, info = run_episode(env, policy_net)
    block = info['block']
    reduced_cost = info['reduced_cost']
    if train and optimizer is not None:
        reinforce_update(policy_net, optimizer, log_probs, returns)
    return block, reduced_cost


def save_policy(policy_net, optimizer, stats, filepath="policy_checkpoint.pt"):
    checkpoint = {
        'state_dim': policy_net.state_dim,
        'n_actions': policy_net.n_actions,
        'hidden_dim': policy_net.hidden_dim,
        'policy_state_dict': policy_net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'total_episodes': stats.get('total_episodes', 0),
        'best_reduced_cost': stats.get('best_reduced_cost', float('inf')),
        'total_columns_generated': stats.get('total_columns_generated', 0)
    }
    torch.save(checkpoint, filepath)
    print(f"✅ Saved: {checkpoint['total_episodes']} episodes, best RC: {checkpoint['best_reduced_cost']:.3f}")


def load_policy(state_dim, n_actions, hidden_dim=256, filepath="policy_checkpoint.pt", device="cpu"):
    if os.path.exists(filepath):
        checkpoint = torch.load(filepath, map_location=device)
        policy_net = PolicyNet(state_dim, n_actions, hidden_dim)
        pretrained_dict = checkpoint['policy_state_dict']
        model_dict = policy_net.state_dict()
        loaded_count = 0
        for name, saved_param in pretrained_dict.items():
            if name in model_dict and saved_param.shape == model_dict[name].shape:
                model_dict[name].copy_(saved_param)
                loaded_count += 1
        policy_net.load_state_dict(model_dict)
        print(f"✅ Loaded {loaded_count}/{len(pretrained_dict)} layers")
        optimizer = optim.Adam(policy_net.parameters(), lr=3e-4)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        stats = {
            'total_episodes': checkpoint.get('total_episodes', 0),
            'best_reduced_cost': checkpoint.get('best_reduced_cost', float('inf')),
            'total_columns_generated': checkpoint.get('total_columns_generated', 0)
        }
        return policy_net, optimizer, stats
    else:
        print("🚀 New policy")
        policy_net = PolicyNet(state_dim, n_actions, hidden_dim)
        optimizer = optim.Adam(policy_net.parameters(), lr=3e-4)
        stats = {'total_episodes': 0, 'best_reduced_cost': float('inf'), 'total_columns_generated': 0}
        return policy_net, optimizer, stats


def get_env_dummy(trips, graph, arcs_df):
    pull_out_trips = arcs_df[arcs_df['arc_type'] == 'pull_out']['to_stop'].tolist()
    pull_out_trips = [
        (row.to_stop, row.travel_time, 0, row.distance_km)  # slack=0 for pull_out
        for row in arcs_df[arcs_df['arc_type'] == 'pull_out'].itertuples()
    ]
    env_dummy = EVSPPricingEnv(trips_df=trips, graph=graph, pull_out_trips=pull_out_trips, duals={t: 0 for t in trips.trip_number})
    state_dim = len(state_to_vec(env_dummy.reset(), env_dummy))
    n_actions = env_dummy.n_trips
    return env_dummy, state_dim, n_actions
