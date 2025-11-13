# play_pacman_manual_smooth.py
import pygame
from pacman_env import PacmanEnv
import math

# Directions
UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
FPS = 10

TILE_SIZE = 20
GRID_WIDTH = 20
GRID_HEIGHT = 11
SCREEN_WIDTH = GRID_WIDTH * TILE_SIZE
SCREEN_HEIGHT = GRID_HEIGHT * TILE_SIZE

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
GREEN = (0, 255, 0)
GHOST_COLOR = (220, 40, 40)

# Animation params
PACMAN_ANIM_INTERVAL = 6  # frames between mouth toggles
GHOST_BOB_AMPLITUDE = 2   # pixels
GHOST_BOB_PERIOD = 20     # frames


def draw_maze_and_entities(screen, env, pacman_mouth_open, ghost_bob_offset):
    """Draw maze, pellets, ghosts, and pacman using env.grid."""
    screen.fill(BLACK)

    # Draw walls and pellets
    for y in range(GRID_HEIGHT):
        for x in range(GRID_WIDTH):
            val = env.grid[y, x]
            rect = pygame.Rect(x * TILE_SIZE, y * TILE_SIZE, TILE_SIZE, TILE_SIZE)
            if val == 1:
                pygame.draw.rect(screen, BLUE, rect)
                pygame.draw.rect(screen, (20, 20, 80), rect, 2)
            elif val == 2:
                pygame.draw.circle(screen, WHITE, rect.center, 3)

    # Draw ghosts (with bobbing)
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

    # Draw Pac-Man (with mouth animation)
    py, px = env.pacman_pos
    pcx = px * TILE_SIZE + TILE_SIZE // 2
    pcy = py * TILE_SIZE + TILE_SIZE // 2
    radius = TILE_SIZE // 2 - 1

    # mouth angle (based on direction)
    angle = 0
    if hasattr(env, "last_action"):
        if env.last_action == UP:
            angle = 3 * math.pi / 2
        elif env.last_action == DOWN:
            angle = math.pi / 2
        elif env.last_action == LEFT:
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
    if env.last_action == LEFT:
        eye_pos = (pcx - eye_offset_x, pcy + eye_offset_y)
    elif env.last_action == UP:
        eye_pos = (pcx + eye_offset_x, pcy - int(radius * 0.6))
    elif env.last_action == DOWN:
        eye_pos = (pcx + eye_offset_x, pcy)
    else:  # RIGHT
        eye_pos = (pcx + eye_offset_x, pcy + eye_offset_y)
    pygame.draw.circle(screen, BLACK, eye_pos, max(1, radius // 6))


def draw_overlay(screen, score, font, small_font, paused=False, game_over=False,
                 ep_text_surface=None, paused_surf=None, gameover_surf=None, final_score_surf=None):
    """UI overlay — draw on top of rendered maze"""
    score_bg = pygame.Rect(10, SCREEN_HEIGHT - 35, 180, 30)
    pygame.draw.rect(screen, (30, 60, 30), score_bg, border_radius=8)
    pygame.draw.rect(screen, GREEN, score_bg, 2, border_radius=8)
    score_text = font.render(f"SCORE: {score}", True, YELLOW)
    screen.blit(score_text, (20, SCREEN_HEIGHT - 33))

    if ep_text_surface:
        screen.blit(ep_text_surface, (SCREEN_WIDTH - 230, SCREEN_HEIGHT - 30))

    if paused or game_over:
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 150))
        screen.blit(overlay, (0, 0))
        if paused:
            if paused_surf:
                screen.blit(paused_surf, paused_surf.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)))
        elif game_over:
            if gameover_surf:
                screen.blit(gameover_surf, gameover_surf.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 30)))
            if final_score_surf:
                screen.blit(final_score_surf, final_score_surf.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 20)))


def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.DOUBLEBUF)
    pygame.display.set_caption("Classic Pac-Man (Smooth Manual Play)")
    clock = pygame.time.Clock()

    font = pygame.font.Font(None, 36)
    small_font = pygame.font.Font(None, 22)

    ep_text_surface = small_font.render("Mode: Manual Play", True, WHITE)
    paused_surf = font.render("PAUSED", True, YELLOW)
    gameover_surf = font.render("GAME OVER", True, RED)

    env = PacmanEnv(render_mode=None)
    obs, info = env.reset()

    running = True
    paused = False
    game_over = False
    direction = RIGHT
    score = 0

    frame_counter = 0
    pacman_mouth_open = True

    # --- Stats ---
    match_count = 0
    total_score = 0

    print("\n" + "=" * 50)
    print("CLASSIC PAC-MAN 🎮")
    print("=" * 50)
    print("Controls:")
    print("  ↑ ↓ ← →  - Change direction")
    print("  R        - Restart after Game Over")
    print("  P        - Pause/Resume")
    print("  Q        - Quit")
    print("=" * 50)

    while running:
        clock.tick(FPS)
        frame_counter += 1

        if frame_counter % PACMAN_ANIM_INTERVAL == 0:
            pacman_mouth_open = not pacman_mouth_open

        ghost_bob_offset = int(GHOST_BOB_AMPLITUDE * math.sin(2 * math.pi * (frame_counter % GHOST_BOB_PERIOD) / GHOST_BOB_PERIOD))

        new_dir = None
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                elif event.key == pygame.K_p:
                    paused = not paused
                    print("⏸️ Paused" if paused else "▶️ Resumed")
                elif event.key == pygame.K_r and game_over:
                    obs, info = env.reset()
                    direction = RIGHT
                    game_over = False
                    score = 0
                    frame_counter = 0
                    pacman_mouth_open = True
                    print("🔁 Game Restarted!")
                elif not paused and not game_over:
                    if event.key == pygame.K_UP:
                        new_dir = UP
                    elif event.key == pygame.K_DOWN:
                        new_dir = DOWN
                    elif event.key == pygame.K_LEFT:
                        new_dir = LEFT
                    elif event.key == pygame.K_RIGHT:
                        new_dir = RIGHT

        if new_dir is not None:
            direction = new_dir

        if not paused and not game_over:
            obs, reward, terminated, truncated, info = env.step(direction)
            score = info["score"]
            if terminated or truncated:
                game_over = True
                match_count += 1
                total_score += score
                avg_score = total_score / match_count
                print(f"💀 Game Over | Match {match_count} | Score: {score} | Avg Score: {avg_score:.2f}")

        draw_maze_and_entities(screen, env, pacman_mouth_open, ghost_bob_offset)
        final_score_surf = small_font.render(f"Final Score: {score}", True, WHITE) if game_over else None
        draw_overlay(screen, score, font, small_font, paused, game_over,
                     ep_text_surface=ep_text_surface, paused_surf=paused_surf,
                     gameover_surf=gameover_surf, final_score_surf=final_score_surf)
        pygame.display.flip()

    # --- Final session summary ---
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

if __name__ == "__main__":
    main()