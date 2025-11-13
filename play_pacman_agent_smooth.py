"""
Play Pac-Man with a trained DQN agent (Smooth Animated Version)
---------------------------------------------------------------
Enhanced with animated Pac-Man and ghosts, double-buffer rendering, and flicker-free UI.
"""

import pygame
import torch
import numpy as np
from pathlib import Path
import sys
import math
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
GHOST_COLOR = (220, 40, 40)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Animation parameters
PACMAN_ANIM_INTERVAL = 6
GHOST_BOB_AMPLITUDE = 2
GHOST_BOB_PERIOD = 20


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

    n_observations = 6
    n_actions = 4

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


# ---------------- Draw Functions ----------------
def draw_maze_and_entities(screen, env, pacman_mouth_open, ghost_bob_offset):
    """Draw maze, pellets, ghosts, and Pac-Man with animations."""
    screen.fill(BLACK)

    for y in range(GRID_HEIGHT):
        for x in range(GRID_WIDTH):
            val = env.grid[y, x]
            rect = pygame.Rect(x * TILE_SIZE, y * TILE_SIZE, TILE_SIZE, TILE_SIZE)
            if val == 1:
                pygame.draw.rect(screen, BLUE, rect)
                pygame.draw.rect(screen, (20, 20, 80), rect, 2)
            elif val == 2:
                pygame.draw.circle(screen, WHITE, rect.center, 3)

    # Draw ghosts
    for i, gpos in enumerate(env.ghost_positions):
        gy, gx = gpos
        cx = gx * TILE_SIZE + TILE_SIZE // 2
        cy = gy * TILE_SIZE + TILE_SIZE // 2 + int(ghost_bob_offset * (1 if i % 2 == 0 else -1))
        pygame.draw.circle(screen, GHOST_COLOR, (cx, cy), TILE_SIZE // 2 - 2)
        eye_x_offset = TILE_SIZE // 6
        pygame.draw.circle(screen, WHITE, (cx - eye_x_offset, cy - 4), 4)
        pygame.draw.circle(screen, WHITE, (cx + eye_x_offset, cy - 4), 4)
        pygame.draw.circle(screen, BLACK, (cx - eye_x_offset + 1, cy - 4), 2)
        pygame.draw.circle(screen, BLACK, (cx + eye_x_offset + 1, cy - 4), 2)

    # Draw Pac-Man (animated mouth)
    py, px = env.pacman_pos
    pcx = px * TILE_SIZE + TILE_SIZE // 2
    pcy = py * TILE_SIZE + TILE_SIZE // 2
    radius = TILE_SIZE // 2 - 1
    angle = 0
    if hasattr(env, "last_action"):
        if env.last_action == 0:
            angle = 3 * math.pi / 2
        elif env.last_action == 1:
            angle = math.pi / 2
        elif env.last_action == 2:
            angle = math.pi
        else:
            angle = 0

    if pacman_mouth_open:
        start_angle = angle + 0.25
        end_angle = angle - 0.25
        points = [(pcx, pcy)]
        steps = 10
        for s in range(steps + 1):
            t = start_angle + (end_angle - start_angle) * (s / steps)
            pxp = pcx + int(radius * math.cos(t))
            pyp = pcy + int(radius * math.sin(t))
            points.append((pxp, pyp))
        pygame.draw.polygon(screen, YELLOW, points)
        pygame.draw.circle(screen, (200, 180, 0), (pcx, pcy), radius, 2)
    else:
        pygame.draw.circle(screen, YELLOW, (pcx, pcy), radius)
        pygame.draw.circle(screen, (200, 180, 0), (pcx, pcy), radius, 2)

    # Pac-Man eye
    eye_offset_x = int(radius * 0.35)
    eye_offset_y = -int(radius * 0.3)
    if hasattr(env, "last_action"):
        if env.last_action == 2:
            eye_pos = (pcx - eye_offset_x, pcy + eye_offset_y)
        elif env.last_action == 0:
            eye_pos = (pcx + eye_offset_x, pcy - int(radius * 0.6))
        elif env.last_action == 1:
            eye_pos = (pcx + eye_offset_x, pcy + int(radius * 0.1))
        else:
            eye_pos = (pcx + eye_offset_x, pcy + eye_offset_y)
    else:
        eye_pos = (pcx + eye_offset_x, pcy + eye_offset_y)
    pygame.draw.circle(screen, BLACK, eye_pos, max(1, radius // 6))


def draw_overlay(screen, score, episode_num, font, small_font, paused=False, game_over=False):
    """Draw score, checkpoint, pause/game over overlays."""
    # Score box
    score_bg = pygame.Rect(10, SCREEN_HEIGHT - 35, 180, 30)
    pygame.draw.rect(screen, (30, 60, 30), score_bg, border_radius=8)
    pygame.draw.rect(screen, GREEN, score_bg, 2, border_radius=8)
    score_text = font.render(f"SCORE: {score}", True, YELLOW)
    screen.blit(score_text, (20, SCREEN_HEIGHT - 33))

    # Episode info
    ep_text = small_font.render(f"Checkpoint: Episode {episode_num}", True, WHITE)
    screen.blit(ep_text, (SCREEN_WIDTH - 230, SCREEN_HEIGHT - 30))

    # Pause overlay
    if paused or game_over:
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 150))
        screen.blit(overlay, (0, 0))

        if paused:
            pause_text = font.render("PAUSED", True, YELLOW)
            screen.blit(pause_text, pause_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)))
        elif game_over:
            go_text = font.render("GAME OVER", True, RED)
            fs_text = small_font.render(f"Final Score: {score}", True, WHITE)
            screen.blit(go_text, go_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 30)))
            screen.blit(fs_text, fs_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 20)))


# ---------------- Main ----------------
def main():
    checkpoint_path = select_checkpoint()
    agent, episode_num = load_agent(checkpoint_path)
    
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.DOUBLEBUF)
    pygame.display.set_caption("Pac-Man DQN Agent (Smooth Animated)")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 32)
    small_font = pygame.font.Font(None, 24)

    env = PacmanEnv(render_mode=None)
    obs, info = env.reset()
    state_np = extract_state(env)
    state = torch.tensor(state_np, dtype=torch.float32, device=device).unsqueeze(0)

    # --- Stats tracking ---
    match_count = 0
    total_score = 0
    score = 0

    running = True
    paused = False
    game_over = False
    frame_counter = 0
    pacman_mouth_open = True

    print("\n🎮 Controls:\n  Q - Quit\n  R - Restart\n  P - Pause/Resume\n")

    while running:
        clock.tick(FPS)
        frame_counter += 1

        # Animation toggles
        if frame_counter % PACMAN_ANIM_INTERVAL == 0:
            pacman_mouth_open = not pacman_mouth_open
        ghost_bob_offset = int(
            GHOST_BOB_AMPLITUDE * math.sin(2 * math.pi * (frame_counter % GHOST_BOB_PERIOD) / GHOST_BOB_PERIOD)
        )

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                elif event.key == pygame.K_p:
                    paused = not paused
                elif event.key == pygame.K_r and game_over:
                    # Restart game
                    env.reset()
                    state_np = extract_state(env)
                    state = torch.tensor(state_np, dtype=torch.float32, device=device).unsqueeze(0)
                    game_over = False
                    score = 0
                    frame_counter = 0
                    pacman_mouth_open = True

        # ---- Agent play ----
        if not paused and not game_over:
            with torch.no_grad():
                action = agent.select_action(state, training=False)
            _, _, terminated, truncated, info = env.step(action.item())
            score = info["score"]

            if terminated or truncated:
                game_over = True
                match_count += 1
                total_score += score
                avg_score = total_score / match_count
                print(f"💀 Game Over | Match {match_count} | Score: {score} | Avg Score: {avg_score:.2f}")
            else:
                next_state_np = extract_state(env)
                state = torch.tensor(next_state_np, dtype=torch.float32, device=device).unsqueeze(0)

        # --- Drawing ---
        draw_maze_and_entities(screen, env, pacman_mouth_open, ghost_bob_offset)
        draw_overlay(screen, score, episode_num, font, small_font, paused, game_over)
        pygame.display.flip()

    # ---- Final stats ----
    if match_count > 0:
        avg_score = total_score / match_count
        print("\n📊 Session Summary:")
        print(f"  Total Matches: {match_count}")
        print(f"  Total Score:   {total_score}")
        print(f"  Average Score: {avg_score:.2f}")
    else:
        print("\n📊 No matches played.")

    env.close()
    pygame.quit()
    print("\n👋 Exiting Pac-Man AI Viewer.")

if __name__ == "__main__":
    main()