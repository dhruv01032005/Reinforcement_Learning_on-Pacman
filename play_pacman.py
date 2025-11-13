import pygame
from pacman_env import PacmanEnv

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
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
GREEN = (0, 255, 0)


def draw_overlay(screen, score, font, small_font, paused=False, game_over=False):
    """Draw score and overlay text on top of environment render (no flicker)."""

    # --- Score Bar (bottom-left)
    score_bg = pygame.Rect(10, SCREEN_HEIGHT - 35, 180, 30)
    pygame.draw.rect(screen, (30, 60, 30), score_bg, border_radius=8)
    pygame.draw.rect(screen, GREEN, score_bg, 2, border_radius=8)
    score_text = font.render(f"SCORE: {score}", True, YELLOW)
    screen.blit(score_text, (20, SCREEN_HEIGHT - 33))

    # --- Mode Info (bottom-right)
    ep_text = small_font.render("Checkpoint: Manual Mode", True, WHITE)
    screen.blit(ep_text, (SCREEN_WIDTH - 230, SCREEN_HEIGHT - 30))

    # --- Pause / Game Over overlays
    if paused or game_over:
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        overlay.set_alpha(150)
        overlay.fill(BLACK)
        screen.blit(overlay, (0, 0))

        if paused:
            pause_text = font.render("PAUSED", True, YELLOW)
            screen.blit(pause_text, (SCREEN_WIDTH // 2 - 60, SCREEN_HEIGHT // 2 - 20))
        elif game_over:
            go_text = font.render("GAME OVER", True, RED)
            final_text = font.render(f"Final Score: {score}", True, WHITE)
            restart_text = small_font.render("Press SPACE to Restart or Q to Quit", True, WHITE)
            screen.blit(go_text, (SCREEN_WIDTH // 2 - 80, SCREEN_HEIGHT // 2 - 60))
            screen.blit(final_text, (SCREEN_WIDTH // 2 - 90, SCREEN_HEIGHT // 2))
            screen.blit(restart_text, (SCREEN_WIDTH // 2 - 130, SCREEN_HEIGHT // 2 + 50))


def main():
    pygame.init()
    pygame.display.set_caption("Classic Pac-Man (Manual Play)")
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    clock = pygame.time.Clock()

    font = pygame.font.Font(None, 36)
    small_font = pygame.font.Font(None, 24)

    env = PacmanEnv(render_mode="human")
    obs, info = env.reset()

    running = True
    paused = False
    game_over = False
    direction = RIGHT
    score = 0

    print("\n" + "=" * 50)
    print("CLASSIC PAC-MAN 🎮")
    print("=" * 50)
    print("Controls:")
    print("  ↑ ↓ ← →  - Change direction")
    print("  SPACE    - Restart after Game Over")
    print("  P        - Pause/Resume")
    print("  Q        - Quit")
    print("=" * 50)

    while running:
        clock.tick(FPS)
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
                elif event.key == pygame.K_SPACE and game_over:
                    obs, info = env.reset()
                    direction = RIGHT
                    game_over = False
                    score = 0
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

        # Game logic
        if not paused and not game_over:
            obs, reward, terminated, truncated, info = env.step(direction)
            score = info["score"]
            if terminated or truncated:
                game_over = True
                print(f"💀 Game Over! Final Score: {score}")

        # --- Render the environment first ---
        env.render()

        # --- Then draw overlays on top (no flicker) ---
        draw_overlay(screen, score, font, small_font, paused, game_over)

        pygame.display.flip()

    env.close()
    pygame.quit()


if __name__ == "__main__":
    main()