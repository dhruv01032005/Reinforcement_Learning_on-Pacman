import pygame
import numpy as np
from pathlib import Path
import sys
import math
import pickle
from pacman_env import PacmanEnv

# ---------------- Display and Colors ----------------
TILE_SIZE = 20
GRID_WIDTH = 20
GRID_HEIGHT = 11
GAME_HEIGHT = GRID_HEIGHT * TILE_SIZE
SCREEN_WIDTH = GRID_WIDTH * TILE_SIZE
SCREEN_HEIGHT = GAME_HEIGHT + 60  # Extra space for UI
FPS = 10

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
GREEN = (0, 255, 0)
GHOST_COLOR = (220, 40, 40)

# Animation parameters
PACMAN_ANIM_INTERVAL = 6
GHOST_BOB_AMPLITUDE = 2
GHOST_BOB_PERIOD = 20


# ---------------- Q-Learning Agent (for loading) ----------------
class QLearningAgent:
    """Q-learning agent for playback"""
    
    def __init__(self, n_actions, q_table):
        self.n_actions = n_actions
        self.q_table = q_table
    
    def discretize_state(self, state):
        """Same discretization as training"""
        x_pac, y_pac, x_g1, y_g1, x_g2, y_g2 = state
        
        dx1 = int(np.clip((x_g1 - x_pac) / 5, -4, 4))
        dy1 = int(np.clip((y_g1 - y_pac) / 3, -3, 3))
        dx2 = int(np.clip((x_g2 - x_pac) / 5, -4, 4))
        dy2 = int(np.clip((y_g2 - y_pac) / 3, -3, 3))
        
        px_bin = int(x_pac / 4)
        py_bin = int(y_pac / 3)
        
        return (px_bin, py_bin, dx1, dy1, dx2, dy2)
    
    def select_action(self, state):
        """Greedy action selection (no exploration)"""
        state_key = self.discretize_state(state)
        if state_key in self.q_table:
            q_values = self.q_table[state_key]
            return np.argmax(q_values)
        else:
            # Random action if state never seen
            return np.random.randint(self.n_actions)


# ---------------- Helper Functions ----------------
def select_checkpoint():
    """Interactive checkpoint selection"""
    script_dir = Path(__file__).parent
    checkpoint_dir = script_dir / "checkpoints_pacman_qn"
    if not checkpoint_dir.exists():
        print("❌ No Q-Learning checkpoints found! Run training first:")
        print("   python train_pacman_qn.py")
        sys.exit(1)

    checkpoints = sorted(checkpoint_dir.glob("checkpoint_ep*.pkl"))
    if not checkpoints:
        print("❌ No checkpoint files found in checkpoints_pacman_qn/")
        sys.exit(1)

    print("\nAvailable Q-Learning checkpoints:")
    for i, cp in enumerate(checkpoints):
        episode = int(cp.stem.split("ep")[1])
        print(f"{i + 1}. Episode {episode:5d}")

    while True:
        choice = input("\nSelect checkpoint number (or 'latest' or 'q' to quit): ").strip()
        if choice.lower() == 'q':
            sys.exit(0)
        if choice.lower() == 'latest':
            return checkpoints[-1]
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(checkpoints):
                return checkpoints[idx]
            else:
                print(f"Enter a number between 1 and {len(checkpoints)}")
        except ValueError:
            print("Invalid input. Enter a number or 'latest'.")


def load_agent(checkpoint_path):
    """Load Q-Learning agent from checkpoint"""
    print(f"\n📂 Loading Q-Learning agent from: {checkpoint_path}")
    with open(checkpoint_path, 'rb') as f:
        checkpoint = pickle.load(f)
    
    q_table = checkpoint['q_table']
    episode = checkpoint['episode']
    algorithm = checkpoint.get('algorithm', 'Q-Learning')
    
    n_actions = 4
    agent = QLearningAgent(n_actions, q_table)
    
    print(f"✅ Loaded checkpoint from Episode {episode} ({algorithm})")
    print(f"   Q-table size: {len(q_table)} states\n")
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

    # Eye
    eye_offset_x = int(radius * 0.35)
    eye_offset_y = -int(radius * 0.3)
    if env.last_action == 2:
        eye_pos = (pcx - eye_offset_x, pcy + eye_offset_y)
    elif env.last_action == 0:
        eye_pos = (pcx + eye_offset_x, pcy - int(radius * 0.6))
    elif env.last_action == 1:
        eye_pos = (pcx + eye_offset_x, pcy)
    else:
        eye_pos = (pcx + eye_offset_x, pcy + eye_offset_y)
    pygame.draw.circle(screen, BLACK, eye_pos, 3)


def draw_ui_overlay(screen, score, episode_num, match_count, avg_score, font, small_font, paused=False, game_over=False):
    """Draw UI overlays"""
    # Score box positioned below game area
    y_pos = GAME_HEIGHT + 10
    score_bg = pygame.Rect(10, y_pos, 200, 45)
    pygame.draw.rect(screen, (30, 30, 30), score_bg, border_radius=8)
    pygame.draw.rect(screen, GREEN, score_bg, 2, border_radius=8)
    
    score_text = font.render(f"SCORE: {score}", True, YELLOW)
    screen.blit(score_text, (20, y_pos + 5))

    # Episode info
    qn_text = small_font.render(f"Q-Learning", True, WHITE)
    ep_text = small_font.render(f"Episode {episode_num}", True, WHITE)
    screen.blit(qn_text, (SCREEN_WIDTH - 180, y_pos + 0))
    screen.blit(ep_text, (SCREEN_WIDTH - 180, y_pos + 15))

    # Overlays
    if paused or game_over:
        overlay = pygame.Surface((SCREEN_WIDTH, GAME_HEIGHT))
        overlay.set_alpha(140)
        overlay.fill(BLACK)
        screen.blit(overlay, (0, 0))

        if paused:
            pause_text = font.render("PAUSED", True, WHITE)
            screen.blit(pause_text, (SCREEN_WIDTH // 2 - 60, GAME_HEIGHT // 2 - 20))
        elif game_over:
            go_text = font.render("GAME OVER", True, RED)
            score_text = font.render(f"Score: {score}", True, WHITE)
            restart_text = small_font.render("Press R to restart or Q to quit", True, WHITE)
            
            screen.blit(go_text, (SCREEN_WIDTH // 2 - 80, GAME_HEIGHT // 2 - 60))
            screen.blit(score_text, (SCREEN_WIDTH // 2 - 60, GAME_HEIGHT // 2))
            screen.blit(restart_text, (SCREEN_WIDTH // 2 - 130, GAME_HEIGHT // 2 + 50))


# ---------------- Main Function ----------------
def main():
    pygame.init()
    pygame.display.set_caption("Pac-Man Q-Learning Agent")

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
    
    score = 0
    running = True
    paused = False
    game_over = False
    frame_count = 0
    match_count = 1
    scores_history = []

    print("\n🎮 Controls:")
    print("  Q - Quit")
    print("  R - Restart")
    print("  P - Pause/Resume\n")
    print("🤖 Watching Q-Learning agent play...\n")

    while running:
        clock.tick(FPS)
        frame_count += 1

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
                    game_over = False
                    score = 0
                    match_count += 1

        if not paused and not game_over:
            action = agent.select_action(state_np)
            _, reward, terminated, truncated, info = env.step(action)
            score = info["score"]

            if terminated or truncated:
                game_over = True
                scores_history.append(score)
                avg_score = sum(scores_history) / len(scores_history)
                print(f"💀 Game Over | Match {match_count} | Score: {score} | Avg Score: {avg_score:.2f}")
            else:
                next_state_np = extract_state(env)
                state_np = next_state_np

        # Animations
        pacman_mouth_open = (frame_count // PACMAN_ANIM_INTERVAL) % 2 == 0
        ghost_bob_offset = GHOST_BOB_AMPLITUDE * math.sin(2 * math.pi * frame_count / GHOST_BOB_PERIOD)

        # Draw
        draw_maze_and_entities(screen, env, pacman_mouth_open, ghost_bob_offset)
        avg_score = sum(scores_history) / len(scores_history) if scores_history else 0
        draw_ui_overlay(screen, score, episode_num, match_count, avg_score, font, small_font, paused, game_over)
        pygame.display.flip()

    env.close()
    pygame.quit()
    
    if scores_history:
        print(f"\n📊 Final Statistics:")
        print(f"   Total matches: {match_count}")
        print(f"   Average score: {sum(scores_history) / len(scores_history):.2f}")
        print(f"   Best score: {max(scores_history)}")
    
    print("\n👋 Exiting Q-Learning Pac-Man Viewer.")


if __name__ == "__main__":
    main()
