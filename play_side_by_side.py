import pygame
import torch
import numpy as np
from pathlib import Path
import sys
import math
import pickle
from pacman_env import PacmanEnv
from train_pacman_dqn import DQNAgent
from train_pacman_ddqn import DoubleDQNAgent

# Display settings
TILE_SIZE = 20
GRID_WIDTH = 20
GRID_HEIGHT = 11
GAME_WIDTH = GRID_WIDTH * TILE_SIZE
GAME_HEIGHT = GRID_HEIGHT * TILE_SIZE
GAP = 15  # Gap between game windows
SCREEN_WIDTH = GAME_WIDTH * 3 + GAP * 2  # Three games
SCREEN_HEIGHT = GAME_HEIGHT + 60  # Extra space for stats
FPS = 10

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
GREEN = (0, 255, 0)
CYAN = (0, 255, 255)
GHOST_COLOR = (220, 40, 40)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PACMAN_ANIM_INTERVAL = 6
GHOST_BOB_AMPLITUDE = 2
GHOST_BOB_PERIOD = 20


# Q-Learning Agent class for loading
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
        """Greedy action selection"""
        state_key = self.discretize_state(state)
        if state_key in self.q_table:
            q_values = self.q_table[state_key]
            return np.argmax(q_values)
        else:
            return np.random.randint(self.n_actions)


def extract_state(env):
    y, x = env.pacman_pos
    ghosts = env.ghost_positions
    if len(ghosts) < 2:
        ghosts += [ghosts[0]] * (2 - len(ghosts))
    (g1y, g1x), (g2y, g2x) = ghosts[:2]
    return np.array([x, y, g1x, g1y, g2x, g2y], dtype=np.float32)


def load_agent(checkpoint_path, algorithm='DQN'):
    print(f"📂 Loading {algorithm} from: {checkpoint_path.name}")
    
    if algorithm == 'Q-Learning':
        with open(checkpoint_path, 'rb') as f:
            checkpoint = pickle.load(f)
        q_table = checkpoint['q_table']
        agent = QLearningAgent(4, q_table)
        episode = checkpoint['episode']
    else:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        config = checkpoint['config']
        
        if algorithm == 'DQN':
            agent = DQNAgent(6, 4, config)
        else:  # Double DQN
            agent = DoubleDQNAgent(6, 4, config)
        
        agent.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        episode = checkpoint['episode']
    
    print(f"✅ Loaded Episode {episode}")
    return agent, episode


def draw_game(surface, env, frame_count, x_offset):
    """Draw a single game instance"""
    for y in range(GRID_HEIGHT):
        for x in range(GRID_WIDTH):
            val = env.grid[y, x]
            rect = pygame.Rect(x * TILE_SIZE + x_offset, y * TILE_SIZE, TILE_SIZE, TILE_SIZE)
            if val == 1:
                pygame.draw.rect(surface, BLUE, rect)
                pygame.draw.rect(surface, (20, 20, 80), rect, 2)
            elif val == 2:
                pygame.draw.circle(surface, WHITE, rect.center, 3)
    
    # Draw ghosts
    ghost_bob_offset = GHOST_BOB_AMPLITUDE * math.sin(2 * math.pi * frame_count / GHOST_BOB_PERIOD)
    for i, gpos in enumerate(env.ghost_positions):
        gy, gx = gpos
        cx = gx * TILE_SIZE + TILE_SIZE // 2 + x_offset
        cy = gy * TILE_SIZE + TILE_SIZE // 2 + int(ghost_bob_offset * (1 if i % 2 == 0 else -1))
        pygame.draw.circle(surface, GHOST_COLOR, (cx, cy), TILE_SIZE // 2 - 2)
        eye_x_offset = TILE_SIZE // 6
        pygame.draw.circle(surface, WHITE, (cx - eye_x_offset, cy - 4), 4)
        pygame.draw.circle(surface, WHITE, (cx + eye_x_offset, cy - 4), 4)
        pygame.draw.circle(surface, BLACK, (cx - eye_x_offset + 1, cy - 4), 2)
        pygame.draw.circle(surface, BLACK, (cx + eye_x_offset + 1, cy - 4), 2)
    
    # Draw Pac-Man
    py, px = env.pacman_pos
    pcx = px * TILE_SIZE + TILE_SIZE // 2 + x_offset
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
    
    pacman_mouth_open = (frame_count // PACMAN_ANIM_INTERVAL) % 2 == 0
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
        pygame.draw.polygon(surface, YELLOW, points)
        pygame.draw.circle(surface, (200, 180, 0), (pcx, pcy), radius, 2)
    else:
        pygame.draw.circle(surface, YELLOW, (pcx, pcy), radius)
        pygame.draw.circle(surface, (200, 180, 0), (pcx, pcy), radius, 2)
    
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
    pygame.draw.circle(surface, BLACK, eye_pos, 3)


def draw_stats(surface, stats_dqn, stats_ddqn, stats_qn, font, small_font):
    """Draw comparison statistics at the bottom for three agents"""
    y_pos = GAME_HEIGHT + 10
    
    # DQN stats (left)
    dqn_text = font.render("DQN", True, CYAN)
    surface.blit(dqn_text, (60, y_pos))
    dqn_score = small_font.render(f"Score: {stats_dqn['score']} | Avg: {stats_dqn['avg']:.1f}", True, WHITE)
    surface.blit(dqn_score, (10, y_pos + 30))
    
    # DDQN stats (center)
    ddqn_text = font.render("DDQN", True, GREEN)
    surface.blit(ddqn_text, (GAME_WIDTH + GAP + 60, y_pos))
    ddqn_score = small_font.render(f"Score: {stats_ddqn['score']} | Avg: {stats_ddqn['avg']:.1f}", True, WHITE)
    surface.blit(ddqn_score, (GAME_WIDTH + GAP + 10, y_pos + 30))
    
    # Q-Learning stats (right)
    qn_text = font.render("Q-LEARN", True, YELLOW)
    surface.blit(qn_text, (GAME_WIDTH * 2 + GAP * 2 + 40, y_pos))
    qn_score = small_font.render(f"Score: {stats_qn['score']} | Avg: {stats_qn['avg']:.1f}", True, WHITE)
    surface.blit(qn_score, (GAME_WIDTH * 2 + GAP * 2 + 10, y_pos + 30))
    
    # Dividers
    pygame.draw.line(surface, WHITE, (GAME_WIDTH + GAP // 2, 0), (GAME_WIDTH + GAP // 2, GAME_HEIGHT), 2)
    pygame.draw.line(surface, WHITE, (GAME_WIDTH * 2 + GAP + GAP // 2, 0), (GAME_WIDTH * 2 + GAP + GAP // 2, GAME_HEIGHT), 2)
    
    # Match count
    match_text = small_font.render(f"Match: {stats_dqn['matches']}", True, RED)
    surface.blit(match_text, (SCREEN_WIDTH // 2 - 40, 5))


def main():
    pygame.init()
    pygame.display.set_caption("Side-by-Side: DQN vs Double DQN vs Q-Learning")
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 32)
    small_font = pygame.font.Font(None, 20)
    
    # Load checkpoints
    script_dir = Path(__file__).parent
    dqn_dir = script_dir / "checkpoints_pacman_dqn"
    ddqn_dir = script_dir / "checkpoints_pacman_ddqn"
    qn_dir = script_dir / "checkpoints_pacman_qn"
    
    dqn_checkpoints = sorted(dqn_dir.glob("checkpoint_ep*.pt"))
    ddqn_checkpoints = sorted(ddqn_dir.glob("checkpoint_ep*.pt"))
    qn_checkpoints = sorted(qn_dir.glob("checkpoint_ep*.pkl"))
    
    if not dqn_checkpoints or not ddqn_checkpoints or not qn_checkpoints:
        print(f"❌ Missing checkpoints!")
        print(f"DQN checkpoints found: {len(dqn_checkpoints)}")
        print(f"DDQN checkpoints found: {len(ddqn_checkpoints)}")
        print(f"Q-Learning checkpoints found: {len(qn_checkpoints)}")
        print(f"\nTrain missing models:")
        if not dqn_checkpoints:
            print("  - DQN: python train_pacman_dqn.py")
        if not ddqn_checkpoints:
            print("  - DDQN: python train_pacman_ddqn.py")
        if not qn_checkpoints:
            print("  - Q-Learning: python train_pacman_qn.py")
        return
    
    print("\n🎮 Side-by-Side Agent Comparison (3 Agents)")
    print("="*60)
    
    dqn_agent, dqn_ep = load_agent(dqn_checkpoints[-1], 'DQN')
    ddqn_agent, ddqn_ep = load_agent(ddqn_checkpoints[-1], 'Double DQN')
    qn_agent, qn_ep = load_agent(qn_checkpoints[-1], 'Q-Learning')
    
    # Create environments
    env_dqn = PacmanEnv(render_mode=None)
    env_ddqn = PacmanEnv(render_mode=None)
    env_qn = PacmanEnv(render_mode=None)
    
    # Initialize
    env_dqn.reset()
    env_ddqn.reset()
    env_qn.reset()
    
    state_dqn = torch.tensor(extract_state(env_dqn), dtype=torch.float32, device=device).unsqueeze(0)
    state_ddqn = torch.tensor(extract_state(env_ddqn), dtype=torch.float32, device=device).unsqueeze(0)
    state_qn = extract_state(env_qn)
    
    # Stats
    stats_dqn = {'score': 0, 'avg': 0, 'matches': 1, 'scores': []}
    stats_ddqn = {'score': 0, 'avg': 0, 'matches': 1, 'scores': []}
    stats_qn = {'score': 0, 'avg': 0, 'matches': 1, 'scores': []}
    
    frame_count = 0
    running = True
    game_over_dqn = False
    game_over_ddqn = False
    game_over_qn = False
    
    print("\n🎮 Controls:")
    print("  R - Restart all games")
    print("  Q - Quit")
    print("  SPACE - Restart when all games over\n")
    print("▶️  Watching agents play...\n")
    
    while running:
        clock.tick(FPS)
        frame_count += 1
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                elif event.key == pygame.K_r or (event.key == pygame.K_SPACE and game_over_dqn and game_over_ddqn and game_over_qn):
                    # Restart all
                    env_dqn.reset()
                    env_ddqn.reset()
                    env_qn.reset()
                    state_dqn = torch.tensor(extract_state(env_dqn), dtype=torch.float32, device=device).unsqueeze(0)
                    state_ddqn = torch.tensor(extract_state(env_ddqn), dtype=torch.float32, device=device).unsqueeze(0)
                    state_qn = extract_state(env_qn)
                    stats_dqn['matches'] += 1
                    stats_ddqn['matches'] += 1
                    stats_qn['matches'] += 1
                    game_over_dqn = False
                    game_over_ddqn = False
                    game_over_qn = False
                    print(f"🔄 Match {stats_dqn['matches']} started")
        
        # Update DQN
        if not game_over_dqn:
            with torch.no_grad():
                action_dqn = dqn_agent.select_action(state_dqn, training=False)
            _, reward_dqn, terminated_dqn, truncated_dqn, info_dqn = env_dqn.step(action_dqn.item())
            stats_dqn['score'] = info_dqn['score']
            
            if terminated_dqn or truncated_dqn:
                game_over_dqn = True
                stats_dqn['scores'].append(info_dqn['score'])
                stats_dqn['avg'] = sum(stats_dqn['scores']) / len(stats_dqn['scores'])
                print(f"💀 DQN died - Score: {info_dqn['score']}")
            else:
                state_dqn = torch.tensor(extract_state(env_dqn), dtype=torch.float32, device=device).unsqueeze(0)
        
        # Update DDQN
        if not game_over_ddqn:
            with torch.no_grad():
                action_ddqn = ddqn_agent.select_action(state_ddqn, training=False)
            _, reward_ddqn, terminated_ddqn, truncated_ddqn, info_ddqn = env_ddqn.step(action_ddqn.item())
            stats_ddqn['score'] = info_ddqn['score']
            
            if terminated_ddqn or truncated_ddqn:
                game_over_ddqn = True
                stats_ddqn['scores'].append(info_ddqn['score'])
                stats_ddqn['avg'] = sum(stats_ddqn['scores']) / len(stats_ddqn['scores'])
                print(f"💀 DDQN died - Score: {info_ddqn['score']}")
            else:
                state_ddqn = torch.tensor(extract_state(env_ddqn), dtype=torch.float32, device=device).unsqueeze(0)
        
        # Update Q-Learning
        if not game_over_qn:
            action_qn = qn_agent.select_action(state_qn)
            _, reward_qn, terminated_qn, truncated_qn, info_qn = env_qn.step(action_qn)
            stats_qn['score'] = info_qn['score']
            
            if terminated_qn or truncated_qn:
                game_over_qn = True
                stats_qn['scores'].append(info_qn['score'])
                stats_qn['avg'] = sum(stats_qn['scores']) / len(stats_qn['scores'])
                print(f"💀 Q-Learning died - Score: {info_qn['score']}")
            else:
                state_qn = extract_state(env_qn)
        
        # Draw
        screen.fill(BLACK)
        draw_game(screen, env_dqn, frame_count, 0)
        draw_game(screen, env_ddqn, frame_count, GAME_WIDTH + GAP)
        draw_game(screen, env_qn, frame_count, GAME_WIDTH * 2 + GAP * 2)
        draw_stats(screen, stats_dqn, stats_ddqn, stats_qn, font, small_font)
        
        # Game over overlays
        if game_over_dqn:
            overlay = pygame.Surface((GAME_WIDTH, GAME_HEIGHT))
            overlay.set_alpha(100)
            overlay.fill(BLACK)
            screen.blit(overlay, (0, 0))
            go_text = small_font.render("GAME OVER", True, RED)
            screen.blit(go_text, (GAME_WIDTH // 2 - 60, GAME_HEIGHT // 2))
        
        if game_over_ddqn:
            overlay = pygame.Surface((GAME_WIDTH, GAME_HEIGHT))
            overlay.set_alpha(100)
            overlay.fill(BLACK)
            screen.blit(overlay, (GAME_WIDTH + GAP, 0))
            go_text = small_font.render("GAME OVER", True, RED)
            screen.blit(go_text, (GAME_WIDTH + GAP + GAME_WIDTH // 2 - 60, GAME_HEIGHT // 2))
        
        if game_over_qn:
            overlay = pygame.Surface((GAME_WIDTH, GAME_HEIGHT))
            overlay.set_alpha(100)
            overlay.fill(BLACK)
            screen.blit(overlay, (GAME_WIDTH * 2 + GAP * 2, 0))
            go_text = small_font.render("GAME OVER", True, RED)
            screen.blit(go_text, (GAME_WIDTH * 2 + GAP * 2 + GAME_WIDTH // 2 - 60, GAME_HEIGHT // 2))
        
        if game_over_dqn and game_over_ddqn and game_over_qn:
            restart_text = small_font.render("Press SPACE to restart", True, YELLOW)
            screen.blit(restart_text, (SCREEN_WIDTH // 2 - 100, GAME_HEIGHT // 2 + 40))
        
        pygame.display.flip()
    
    env_dqn.close()
    env_ddqn.close()
    env_qn.close()
    pygame.quit()
    
    # Final stats
    if stats_dqn['scores'] and stats_ddqn['scores'] and stats_qn['scores']:
        print("\n" + "="*60)
        print("FINAL STATISTICS")
        print("="*60)
        print(f"DQN         - Avg Score: {stats_dqn['avg']:.2f} | Best: {max(stats_dqn['scores'])}")
        print(f"DDQN        - Avg Score: {stats_ddqn['avg']:.2f} | Best: {max(stats_ddqn['scores'])}")
        print(f"Q-Learning  - Avg Score: {stats_qn['avg']:.2f} | Best: {max(stats_qn['scores'])}")
        print("="*60)
    
    print("\n👋 Exiting comparison viewer")


if __name__ == "__main__":
    main()
