import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys


def load_metrics(metrics_dir, algorithm_name):
    """Load all metrics files from a directory"""
    metrics_dir = Path(metrics_dir)
    if not metrics_dir.exists():
        print(f"❌ Metrics directory not found: {metrics_dir}")
        return None
    
    metric_files = sorted(metrics_dir.glob("metrics_ep*.json"))
    if not metric_files:
        print(f"❌ No metrics files found in {metrics_dir}")
        return None
    
    episodes = []
    mean_rewards = []
    mean_scores = []
    all_rewards = []
    all_scores = []
    
    for mf in metric_files:
        with open(mf, 'r') as f:
            data = json.load(f)
            
            # Extract episode number from filename if not in JSON
            if 'episode' in data:
                episode = data['episode']
            else:
                # Parse from filename like "metrics_ep00050.json"
                episode = int(mf.stem.replace('metrics_ep', ''))
            
            episodes.append(episode)
            
            mean_reward = data.get('mean_reward_last_50', 0)
            mean_score = data.get('mean_score_last_50', 0)
            
            mean_rewards.append(mean_reward)
            mean_scores.append(mean_score)
            
            # Store all raw data for final metrics
            if 'rewards' in data:
                all_rewards.extend(data['rewards'])
            if 'scores' in data:
                all_scores.extend(data['scores'])
    
    print(f"✅ Loaded {len(metric_files)} metric files for {algorithm_name}")
    
    return {
        'episodes': episodes,
        'mean_rewards': mean_rewards,
        'mean_scores': mean_scores,
        'all_rewards': all_rewards,
        'all_scores': all_scores
    }


def smooth_curve(data, window=5):
    """Apply moving average smoothing"""
    if len(data) < window:
        return data
    smoothed = np.convolve(data, np.ones(window)/window, mode='valid')
    return smoothed


def plot_comparison(dqn_metrics, ddqn_metrics, qn_metrics, output_file='training_comparison.png'):
    """Create comparison plots (Mean Reward and Mean Score)."""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('DQN vs Double DQN vs Q-Learning Training Comparison', fontsize=16, fontweight='bold')

    # Plot 1: Mean Rewards over Episodes
    ax1 = axes[0]
    if dqn_metrics:
        ax1.plot(dqn_metrics['episodes'], dqn_metrics['mean_rewards'],
                 label='DQN', color='blue', alpha=0.7, linewidth=2)
    if ddqn_metrics:
        ax1.plot(ddqn_metrics['episodes'], ddqn_metrics['mean_rewards'],
                 label='Double DQN', color='red', alpha=0.7, linewidth=2)
    if qn_metrics:
        ax1.plot(qn_metrics['episodes'], qn_metrics['mean_rewards'],
                 label='Q-Learning', color='green', alpha=0.7, linewidth=2)
    ax1.set_xlabel('Episode', fontsize=12)
    ax1.set_ylabel('Mean Reward (last 50 eps)', fontsize=12)
    ax1.set_title('Average Reward Progression', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Mean Scores over Episodes
    ax2 = axes[1]
    if dqn_metrics:
        ax2.plot(dqn_metrics['episodes'], dqn_metrics['mean_scores'],
                 label='DQN', color='blue', alpha=0.7, linewidth=2)
    if ddqn_metrics:
        ax2.plot(ddqn_metrics['episodes'], ddqn_metrics['mean_scores'],
                 label='Double DQN', color='red', alpha=0.7, linewidth=2)
    if qn_metrics:
        ax2.plot(qn_metrics['episodes'], qn_metrics['mean_scores'],
                 label='Q-Learning', color='green', alpha=0.7, linewidth=2)
    ax2.set_xlabel('Episode', fontsize=12)
    ax2.set_ylabel('Mean Score (last 50 eps)', fontsize=12)
    ax2.set_title('Average Score Progression', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n📊 Comparison plot saved to: {output_file}")
    plt.show()


def main():
    print("\n📊 Training Visualization Tool")
    print("="*60)
    
    # Load metrics
    script_dir = Path(__file__).parent
    dqn_dir = script_dir / "metrics_pacman_dqn"
    ddqn_dir = script_dir / "metrics_pacman_ddqn"
    qn_dir = script_dir / "metrics_pacman_qn"
    
    dqn_metrics = load_metrics(dqn_dir, "DQN")
    ddqn_metrics = load_metrics(ddqn_dir, "Double DQN")
    qn_metrics = load_metrics(qn_dir, "Q-Learning")
    
    if dqn_metrics is None and ddqn_metrics is None and qn_metrics is None:
        print("\n❌ No metrics found for any algorithm!")
        print("Train the models first:")
        print("  - DQN: python train_pacman_dqn.py")
        print("  - Double DQN: python train_pacman_ddqn.py")
        print("  - Q-Learning: python train_pacman_qn.py")
        return
    
    # Create plots
    print("\n📈 Generating comparison plots...")
    plot_comparison(dqn_metrics, ddqn_metrics, qn_metrics)
    
    print("\n✅ Visualization complete!")


if __name__ == "__main__":
    main()
