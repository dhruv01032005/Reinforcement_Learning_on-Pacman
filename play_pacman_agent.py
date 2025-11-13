"""
Play Pac-Man with a trained DQN agent
Loads checkpoint and watches AI play in real-time
"""

import pygame
import torch
import numpy as np
from pathlib import Path
import sys
from pacman_env import PacmanEnv
from train_pacman import DQNAgent

# ---------------- Display and Colors ----------------
TILE_SIZE = 20
GRID_WIDTH = 20
GRID_HEIGHT = 11
SCREEN_WIDTH = GRID_WIDTH * TILE_SIZE
SCREEN_HEIGHT = GRID_HEIGHT * TILE_SIZE
FPS = 10

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
GREEN = (0, 255, 0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------- Helper Functions ----------------
def select_checkpoint():
    """Interactive checkpoint selection"""
    checkpoint_dir = Path("checkpoints_pacman")
    if not checkpoint_dir.exists():
        print("❌ No checkpoints found! Run training first.")
        sys.exit(1)

    checkpoints = sorted(checkpoint_dir.glob("checkpoint_ep*.pt"))
    if not checkpoints:
        print("❌ No checkpoint files found in checkpoints_pacman/")
        sys.exit(1)

    print("\nAvailable checkpoints:")
    for i, cp in enumerate(checkpoints):
        episode = int(cp.stem.split("ep")[1])
        print(f"{i + 1}. Episode {episode:4d}")

    while True:
        choice = input("\nSelect checkpoint number (or 'q' to quit): ").strip()
        if choice.lower() == 'q':
            sys.exit(0)
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(checkpoints):
                return checkpoints[idx]
            else:
                print(f"Enter a number between 1 and {len(checkpoints)}")
        except ValueError:
            print("Invalid input. Enter a number.")


def load_agent(checkpoint_path):
    """Load DQN agent from checkpoint"""
    print(f"\n📂 Loading agent from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']

    n_observations = 6  # (x_pacman, y_pacman, x_g1, y_g1, x_g2, y_g2)
    n_actions = 4  # up, down, left, right

    agent = DQNAgent(n_observations, n_actions, config)
    agent.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
    episode = checkpoint['episode']
    print(f"✅ Loaded checkpoint from Episode {episode}\n")
    return agent, episode


def extract_state(env):
    """Return (x_pacman, y_pacman, x_g1, y_g1, x_g2, y_g2)"""
    y, x = env.pacman_pos
    ghosts = env.ghost_positions
    if len(ghosts) < 2:
        ghosts += [ghosts[0]] * (2 - len(ghosts))
    (g1y, g1x), (g2y, g2x) = ghosts[:2]
    return np.array([x, y, g1x, g1y, g2x, g2y], dtype=np.float32)


# ---------------- Pygame Draw ----------------
def draw_game(screen, env, score, episode_num, font, small_font, paused=False, game_over=False):
    """Draw the Pac-Man game grid and stats"""
    screen.fill(BLACK)

    for y in range(GRID_HEIGHT):
        for x in range(GRID_WIDTH):
            rect = pygame.Rect(x * TILE_SIZE, y * TILE_SIZE, TILE_SIZE, TILE_SIZE)
            val = env.grid[y, x]
            if val == 1:
                pygame.draw.rect(screen, BLUE, rect)
            elif val == 2:
                pygame.draw.circle(screen, WHITE, rect.center, 3)
            elif val == 3:
                pygame.draw.circle(screen, RED, rect.center, TILE_SIZE // 2 - 2)
            elif val == 4:
                pygame.draw.circle(screen, YELLOW, rect.center, TILE_SIZE // 2)

    # Draw Score Box
    score_bg = pygame.Rect(10, SCREEN_HEIGHT - 35, 180, 30)
    pygame.draw.rect(screen, (50, 50, 50), score_bg, border_radius=8)
    pygame.draw.rect(screen, GREEN, score_bg, 2, border_radius=8)
    score_text = font.render(f"SCORE: {score}", True, YELLOW)
    screen.blit(score_text, (20, SCREEN_HEIGHT - 33))

    # Draw Episode Info
    ep_text = small_font.render(f"Checkpoint: Episode {episode_num}", True, WHITE)
    screen.blit(ep_text, (SCREEN_WIDTH - 230, SCREEN_HEIGHT - 30))

    # Pause Overlay
    if paused:
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        overlay.set_alpha(140)
        overlay.fill(BLACK)
        screen.blit(overlay, (0, 0))
        pause_text = font.render("PAUSED", True, WHITE)
        screen.blit(pause_text, (SCREEN_WIDTH // 2 - 60, SCREEN_HEIGHT // 2 - 20))

    # Game Over Overlay
    if game_over:
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        overlay.set_alpha(150)
        overlay.fill(BLACK)
        screen.blit(overlay, (0, 0))

        go_text = font.render("GAME OVER", True, RED)
        score_text = font.render(f"Final Score: {score}", True, WHITE)
        restart_text = small_font.render("Press R to restart or Q to quit", True, WHITE)

        screen.blit(go_text, (SCREEN_WIDTH // 2 - 80, SCREEN_HEIGHT // 2 - 60))
        screen.blit(score_text, (SCREEN_WIDTH // 2 - 90, SCREEN_HEIGHT // 2))
        screen.blit(restart_text, (SCREEN_WIDTH // 2 - 130, SCREEN_HEIGHT // 2 + 50))


# ---------------- Main Function ----------------
def main():
    pygame.init()
    pygame.display.set_caption("Pac-Man DQN Agent")

    # Select checkpoint
    checkpoint_path = select_checkpoint()
    agent, episode_num = load_agent(checkpoint_path)

    # Setup screen
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 32)
    small_font = pygame.font.Font(None, 24)

    # Create environment
    env = PacmanEnv(render_mode=None)
    env.reset()
    state_np = extract_state(env)
    state = torch.tensor(state_np, dtype=torch.float32, device=device).unsqueeze(0)
    score = 0
    running = True
    paused = False
    game_over = False

    print("\n🎮 Controls:")
    print("  Q - Quit")
    print("  R - Restart")
    print("  P - Pause/Resume\n")

    while running:
        clock.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                elif event.key == pygame.K_p:
                    paused = not paused
                elif event.key == pygame.K_r and game_over:
                    env.reset()
                    state_np = extract_state(env)
                    state = torch.tensor(state_np, dtype=torch.float32, device=device).unsqueeze(0)
                    game_over = False
                    score = 0

        if not paused and not game_over:
            with torch.no_grad():
                action = agent.select_action(state, training=False)
            _, reward, terminated, truncated, info = env.step(action.item())
            score = info["score"]

            if terminated or truncated:
                game_over = True
            else:
                next_state_np = extract_state(env)
                state = torch.tensor(next_state_np, dtype=torch.float32, device=device).unsqueeze(0)

        draw_game(screen, env, score, episode_num, font, small_font, paused, game_over)
        pygame.display.flip()

    env.close()
    pygame.quit()
    print("\n👋 Exiting Pac-Man AI Viewer.")


if __name__ == "__main__":
    main()