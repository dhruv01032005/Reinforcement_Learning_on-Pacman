import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque, namedtuple
import random
import math
from pathlib import Path
import json
from pacman_env import PacmanEnv

# -------------------- Setup --------------------
CHECKPOINT_DIR = Path("checkpoints_pacman")
METRICS_DIR = Path("metrics_pacman")
for d in [CHECKPOINT_DIR, METRICS_DIR]:
    d.mkdir(exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


# -------------------- Replay Memory --------------------
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# -------------------- DQN Model --------------------
class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(n_observations, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# -------------------- DQN Agent --------------------
class DQNAgent:
    def __init__(self, n_observations, n_actions, config):
        self.n_actions = n_actions
        self.config = config

        self.policy_net = DQN(n_observations, n_actions).to(device)
        self.target_net = DQN(n_observations, n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=config['lr'], amsgrad=True)
        self.memory = ReplayMemory(config['memory_size'])

        self.steps_done = 0
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_scores = []

    def select_action(self, state, training=True):
        if training:
            eps_threshold = self.config['eps_end'] + (self.config['eps_start'] - self.config['eps_end']) * \
                            math.exp(-1. * self.steps_done / self.config['eps_decay'])
            self.steps_done += 1
            if random.random() > eps_threshold:
                with torch.no_grad():
                    return self.policy_net(state).max(1).indices.view(1, 1)
            else:
                return torch.tensor([[random.randrange(self.n_actions)]],
                                    device=device, dtype=torch.long)
        else:
            with torch.no_grad():
                return self.policy_net(state).max(1).indices.view(1, 1)

    def optimize_model(self):
        if len(self.memory) < self.config['batch_size']:
            return

        transitions = self.memory.sample(self.config['batch_size'])
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)),
                                      device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(self.config['batch_size'], device=device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values

        expected_state_action_values = (next_state_values * self.config['gamma']) + reward_batch

        loss = nn.SmoothL1Loss()(state_action_values, expected_state_action_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def update_target_network(self):
        tau = self.config['tau']
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(tau * policy_param.data + (1.0 - tau) * target_param.data)


# -------------------- Helpers --------------------
def extract_state(env):
    """Return (x_pacman, y_pacman, x_ghost1, y_ghost1, x_ghost2, y_ghost2)"""
    y, x = env.pacman_pos
    ghosts = env.ghost_positions
    if len(ghosts) < 2:
        ghosts += [ghosts[0]] * (2 - len(ghosts))
    (g1y, g1x), (g2y, g2x) = ghosts[:2]
    return np.array([x, y, g1x, g1y, g2x, g2y], dtype=np.float32)

def save_checkpoint(agent, episode, config, path):
    checkpoint = {
        'episode': episode,
        'policy_net_state_dict': agent.policy_net.state_dict(),
        'target_net_state_dict': agent.target_net.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
        'steps_done': agent.steps_done,
        'config': config
    }
    torch.save(checkpoint, path)
    print(f"✅ Saved checkpoint: {path}")

def save_metrics(agent, episode, path):
    metrics = {
        'rewards': agent.episode_rewards,
        'scores': agent.episode_scores,
        'mean_reward': np.mean(agent.episode_rewards[-50:]),
        'mean_score': np.mean(agent.episode_scores[-50:])
    }
    with open(path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"📊 Saved metrics: {path}")

# -------------------- Training Loop --------------------
def train():
    config = {
        'lr': 1e-4,
        'batch_size': 128,
        'gamma': 0.99,
        'eps_start': 0.9,
        'eps_end': 0.05,
        'eps_decay': 5000,
        'tau': 0.005,
        'memory_size': 50000
    }

    num_episodes = 10000
    checkpoint_interval = 50

    env = PacmanEnv(render_mode=None)
    n_observations = 6  # (x_pacman, y_pacman, x_g1, y_g1, x_g2, y_g2)
    n_actions = 4  # up, down, left, right
    agent = DQNAgent(n_observations, n_actions, config)

    print(f"\n🎮 Starting Pac-Man DQN Training for {num_episodes} episodes\n")

    for episode in range(num_episodes):
        env.reset()
        state_np = extract_state(env)
        state = torch.tensor(state_np, dtype=torch.float32, device=device).unsqueeze(0)

        total_reward = 0
        total_steps = 0

        while True:
            action = agent.select_action(state, training=True)
            _, reward, terminated, truncated, info = env.step(action.item())

            next_state_np = extract_state(env)
            next_state = torch.tensor(next_state_np, dtype=torch.float32, device=device).unsqueeze(0)

            reward_tensor = torch.tensor([reward], device=device)
            done = terminated or truncated

            if done:
                next_state = None

            agent.memory.push(state, action, next_state, reward_tensor)
            state = next_state if next_state is not None else state
            total_reward += reward
            total_steps += 1

            agent.optimize_model()
            agent.update_target_network()

            if done:
                agent.episode_rewards.append(total_reward)
                agent.episode_lengths.append(total_steps)
                agent.episode_scores.append(info['score'])
                break

        # Logging
        if (episode + 1) % 10 == 0:
            avg_r = np.mean(agent.episode_rewards[-10:])
            avg_s = np.mean(agent.episode_scores[-10:])
            print(f"Ep {episode + 1}/{num_episodes} | AvgR: {avg_r:.2f} | AvgScore: {avg_s:.2f} | Steps: {agent.steps_done}")

        # Save checkpoints
        if (episode + 1) % checkpoint_interval == 0:
            ckpt_path = CHECKPOINT_DIR / f"checkpoint_ep{episode+1:05d}.pt"
            metric_path = METRICS_DIR / f"metrics_ep{episode+1:05d}.json"
            save_checkpoint(agent, episode + 1, config, ckpt_path)
            save_metrics(agent, episode + 1, metric_path)

    # Final save
    save_checkpoint(agent, num_episodes, config, CHECKPOINT_DIR / f"checkpoint_final.pt")
    save_metrics(agent, num_episodes, METRICS_DIR / f"metrics_final.json")

    env.close()
    print("\n🏁 Training complete!")
    print(f"Final average reward: {np.mean(agent.episode_rewards[-100:]):.2f}")
    print(f"Final average score: {np.mean(agent.episode_scores[-100:]):.2f}")


if __name__ == "__main__":
    train()