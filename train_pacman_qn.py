import numpy as np
import pickle
from pathlib import Path
import json
from collections import defaultdict
import random
from pacman_env import PacmanEnv

# -------------------- Setup --------------------
CHECKPOINT_DIR = Path("checkpoints_pacman_qn")
METRICS_DIR = Path("metrics_pacman_qn")
for d in [CHECKPOINT_DIR, METRICS_DIR]:
    d.mkdir(exist_ok=True)

print(f"Q-Learning Training - Tabular Method")


# -------------------- Q-Learning Agent --------------------
class QLearningAgent:
    """Tabular Q-learning agent with state discretization"""
    
    def __init__(self, n_actions, config):
        self.n_actions = n_actions
        self.config = config
        
        # Q-table: dictionary mapping (state_tuple) -> [Q-values for each action]
        self.q_table = defaultdict(lambda: np.zeros(n_actions))
        
        self.epsilon = config['epsilon_start']
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_scores = []
        self.steps_done = 0
    
    def discretize_state(self, state):
        """
        Discretize continuous state into bins for Q-table lookup.
        State: [x_pac, y_pac, x_g1, y_g1, x_g2, y_g2]
        We'll use grid positions (already discrete) and compute relative positions.
        """
        x_pac, y_pac, x_g1, y_g1, x_g2, y_g2 = state
        
        # Discretize relative ghost positions (distance bins)
        dx1 = int(np.clip((x_g1 - x_pac) / 5, -4, 4))  # Relative distance in bins
        dy1 = int(np.clip((y_g1 - y_pac) / 3, -3, 3))
        dx2 = int(np.clip((x_g2 - x_pac) / 5, -4, 4))
        dy2 = int(np.clip((y_g2 - y_pac) / 3, -3, 3))
        
        # Pacman position (coarse grid)
        px_bin = int(x_pac / 4)  # 5 bins for x
        py_bin = int(y_pac / 3)  # 4 bins for y
        
        return (px_bin, py_bin, dx1, dy1, dx2, dy2)
    
    def select_action(self, state, training=True):
        """Epsilon-greedy action selection"""
        if training:
            if random.random() < self.epsilon:
                return random.randrange(self.n_actions)
            else:
                state_key = self.discretize_state(state)
                q_values = self.q_table[state_key]
                return np.argmax(q_values)
        else:
            # Greedy during evaluation
            state_key = self.discretize_state(state)
            q_values = self.q_table[state_key]
            return np.argmax(q_values)
    
    def update(self, state, action, reward, next_state, done):
        """Q-learning update rule: Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]"""
        state_key = self.discretize_state(state)
        next_state_key = self.discretize_state(next_state)
        
        current_q = self.q_table[state_key][action]
        
        if done:
            target_q = reward
        else:
            max_next_q = np.max(self.q_table[next_state_key])
            target_q = reward + self.config['gamma'] * max_next_q
        
        # Q-learning update
        self.q_table[state_key][action] += self.config['alpha'] * (target_q - current_q)
        
        self.steps_done += 1
    
    def decay_epsilon(self):
        """Decay epsilon after each episode"""
        self.epsilon = max(
            self.config['epsilon_end'],
            self.epsilon * self.config['epsilon_decay']
        )


# -------------------- Helper Functions --------------------
def extract_state(env):
    """Return (x_pacman, y_pacman, x_ghost1, y_ghost1, x_ghost2, y_ghost2)"""
    y, x = env.pacman_pos
    ghosts = env.ghost_positions
    if len(ghosts) < 2:
        ghosts += [ghosts[0]] * (2 - len(ghosts))
    (g1y, g1x), (g2y, g2x) = ghosts[:2]
    return np.array([x, y, g1x, g1y, g2x, g2y], dtype=np.float32)


def save_checkpoint(agent, episode, config, path):
    """Save Q-table and agent state"""
    checkpoint = {
        'episode': episode,
        'q_table': dict(agent.q_table),  # Convert defaultdict to dict for pickling
        'epsilon': agent.epsilon,
        'steps_done': agent.steps_done,
        'config': config,
        'algorithm': 'Q-Learning'
    }
    with open(path, 'wb') as f:
        pickle.dump(checkpoint, f)
    print(f"✅ Saved checkpoint: {path}")


def save_metrics(agent, episode, path):
    """Save training metrics"""
    metrics = {
        'episode': episode,
        'rewards': agent.episode_rewards,
        'scores': agent.episode_scores,
        'mean_reward_last_50': float(np.mean(agent.episode_rewards[-50:])),
        'mean_score_last_50': float(np.mean(agent.episode_scores[-50:])),
        'q_table_size': len(agent.q_table)
    }
    with open(path, 'w') as f:
        json.dump(metrics, f, indent=2)


# -------------------- Training Loop --------------------
def train():
    """Main training function for Q-learning"""
    
    # Configuration
    config = {
        'n_episodes': 10000,
        'max_steps': 500,
        'alpha': 0.1,           # Learning rate
        'gamma': 0.99,          # Discount factor
        'epsilon_start': 1.0,   # Initial exploration
        'epsilon_end': 0.01,    # Final exploration
        'epsilon_decay': 0.995, # Decay per episode
        'checkpoint_freq': 50,
        'print_freq': 10
    }
    
    print("\n" + "="*60)
    print("Q-LEARNING TRAINING CONFIGURATION")
    print("="*60)
    for k, v in config.items():
        print(f"  {k}: {v}")
    print("="*60 + "\n")
    
    # Initialize environment and agent
    env = PacmanEnv(render_mode=None)
    n_actions = env.action_space.n
    
    agent = QLearningAgent(n_actions, config)
    
    print("🎮 Starting Q-Learning training...\n")
    
    for episode in range(1, config['n_episodes'] + 1):
        env.reset()
        state = extract_state(env)
        
        episode_reward = 0
        episode_score = 0
        steps = 0
        
        for step in range(config['max_steps']):
            # Select and perform action
            action = agent.select_action(state, training=True)
            _, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            next_state = extract_state(env)
            episode_reward += reward
            episode_score = info['score']
            steps += 1
            
            # Update Q-table
            agent.update(state, action, reward, next_state, done)
            
            state = next_state
            
            if done:
                break
        
        # Store episode statistics
        agent.episode_rewards.append(episode_reward)
        agent.episode_scores.append(episode_score)
        agent.episode_lengths.append(steps)
        
        # Decay epsilon
        agent.decay_epsilon()
        
        # Print progress
        if episode % config['print_freq'] == 0:
            num_episodes = config['n_episodes']
            avg_r = np.mean(agent.episode_rewards[-50:])
            avg_s = np.mean(agent.episode_scores[-50:])
            q_table_size = len(agent.q_table)
            print(f"Ep {episode + 1}/{num_episodes} | AvgR: {avg_r:.2f} | AvgScore: {avg_s:.2f} | Steps: {agent.steps_done} | Q-states: {q_table_size}")
        
        # Save checkpoint
        if episode % config['checkpoint_freq'] == 0:
            checkpoint_path = CHECKPOINT_DIR / f"checkpoint_ep{episode:05d}.pkl"
            save_checkpoint(agent, episode, config, checkpoint_path)
            
            metrics_path = METRICS_DIR / f"metrics_ep{episode:05d}.json"
            save_metrics(agent, episode, metrics_path)
    
    # Final save
    final_checkpoint = CHECKPOINT_DIR / f"checkpoint_ep{config['n_episodes']:05d}.pkl"
    save_checkpoint(agent, config['n_episodes'], config, final_checkpoint)
    
    final_metrics = METRICS_DIR / f"metrics_ep{config['n_episodes']:05d}.json"
    save_metrics(agent, config['n_episodes'], final_metrics)
    
    env.close()
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Total episodes: {config['n_episodes']}")
    print(f"Final mean reward (last 100): {np.mean(agent.episode_rewards[-100:]):.2f}")
    print(f"Final mean score (last 100): {np.mean(agent.episode_scores[-100:]):.2f}")
    print(f"Q-table size (unique states): {len(agent.q_table)}")
    print(f"Best score: {max(agent.episode_scores)}")
    print("="*60)


if __name__ == "__main__":
    train()
