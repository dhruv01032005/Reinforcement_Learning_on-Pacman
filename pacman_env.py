import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
import random
from typing import Optional, Tuple

# Game constants
TILE_SIZE = 20
GRID_WIDTH = 20
GRID_HEIGHT = 11
SCREEN_WIDTH = GRID_WIDTH * TILE_SIZE
SCREEN_HEIGHT = GRID_HEIGHT * TILE_SIZE
FPS = 10  # Pac-Man speed

# Colors
BLACK = (0, 0, 0)
YELLOW = (255, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
WHITE = (255, 255, 255)


class PacmanEnv(gym.Env):
    """
    Classic Pac-Man environment (manual play).

    Grid legend:
        0: empty
        1: wall
        2: pellet
        3: ghost
        4: Pac-Man
    """

    metadata = {"render_modes": ["human"], "render_fps": FPS}

    def __init__(self, render_mode: Optional[str] = None):
        super().__init__()
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=4, shape=(GRID_HEIGHT, GRID_WIDTH), dtype=np.int32)

        self.render_mode = render_mode
        self.screen = None
        self.clock = None

        self.grid = None
        self.pacman_pos = None
        self.ghost_positions = []
        self.ghost_under = {}
        self.score = 0
        self.steps = 0
        self.ghost_move_counter = 0  # for controlling ghost speed

        if render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption("Classic Pac-Man")
            self.clock = pygame.time.Clock()

    def _load_maze(self):
        """Create a classic maze-like layout (approximation of original Pac-Man)."""
        layout = np.ones((GRID_HEIGHT, GRID_WIDTH), dtype=np.int32)

        layout[1][1] = 0
        layout[1][2] = 2
        layout[1][3] = 2
        layout[1][4] = 2
        layout[1][6] = 2
        layout[1][7] = 2
        layout[1][8] = 2
        layout[1][9] = 2
        layout[1][10] = 2
        layout[1][11] = 2
        layout[1][12] = 2
        layout[1][13] = 2
        layout[1][15] = 2
        layout[1][16] = 2
        layout[1][17] = 2
        layout[1][18] = 0
        layout[2][1] = 2
        layout[2][4] = 2
        layout[2][6] = 2
        layout[2][13] = 2
        layout[2][15] = 2
        layout[2][18] = 2
        layout[3][1] = 2
        layout[3][3] = 2
        layout[3][4] = 2
        layout[3][5] = 2
        layout[3][6] = 2
        layout[3][7] = 2
        layout[3][8] = 2
        layout[3][9] = 2
        layout[3][10] = 2
        layout[3][11] = 2
        layout[3][12] = 2
        layout[3][13] = 2
        layout[3][14] = 2
        layout[3][15] = 2
        layout[3][16] = 2
        layout[3][18] = 2
        layout[4][1] = 2
        layout[4][3] = 2
        layout[4][6] = 2
        layout[4][13] = 2
        layout[4][16] = 2
        layout[4][18] = 2
        layout[5][1] = 2
        layout[5][2] = 2
        layout[5][3] = 2
        layout[5][4] = 2
        layout[5][5] = 2
        layout[5][6] = 2
        layout[5][7] = 2
        layout[5][8] = 2
        layout[5][9] = 2
        layout[5][10] = 2
        layout[5][11] = 2
        layout[5][12] = 2
        layout[5][13] = 2
        layout[5][14] = 2
        layout[5][15] = 2
        layout[5][16] = 2
        layout[5][17] = 2
        layout[5][18] = 2
        layout[6][1] = 2
        layout[6][3] = 2
        layout[6][6] = 2
        layout[6][13] = 2
        layout[6][16] = 2
        layout[6][18] = 2
        layout[7][1] = 2
        layout[7][3] = 2
        layout[7][4] = 2
        layout[7][5] = 2
        layout[7][6] = 2
        layout[7][7] = 2
        layout[7][8] = 2
        layout[7][9] = 2
        layout[7][10] = 2
        layout[7][11] = 2
        layout[7][12] = 2
        layout[7][13] = 2
        layout[7][14] = 2
        layout[7][15] = 2
        layout[7][16] = 2
        layout[7][18] = 2
        layout[8][1] = 2
        layout[8][4] = 2
        layout[8][6] = 2
        layout[8][13] = 2
        layout[8][15] = 2
        layout[8][18] = 2
        layout[9][1] = 2
        layout[9][2] = 2
        layout[9][3] = 2
        layout[9][4] = 2
        layout[9][6] = 2
        layout[9][7] = 2
        layout[9][8] = 2
        layout[9][9] = 0
        layout[9][10] = 2
        layout[9][11] = 2
        layout[9][12] = 2
        layout[9][13] = 2
        layout[9][15] = 2
        layout[9][16] = 2
        layout[9][17] = 2
        layout[9][18] = 2
        
        return layout

    def _place_pacman(self):
        x,y = 9,9
        self.grid[y, x] = 4
        return (y, x)

    def _place_ghosts(self):
        ghost_spots = [(1, 1), (1, 18)]
        for pos in ghost_spots:
            self.grid[pos] = 3
        return ghost_spots

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.grid = self._load_maze()
        self.pacman_pos = self._place_pacman()
        self.ghost_under = {}
        self.ghost_positions = self._place_ghosts()
        self.score = 0
        self.steps = 0
        self.last_action = 3  # start moving right
        self.ghost_move_counter = 0
        return self.grid.copy(), {"score": self.score}

    def _move_entity(self, pos, action):
        y, x = pos
        if action == 0:
            new = (y - 1, x)
        elif action == 1:
            new = (y + 1, x)
        elif action == 2:
            new = (y, x - 1)
        elif action == 3:
            new = (y, x + 1)
        else:
            return pos

        if not (0 <= new[0] < GRID_HEIGHT and 0 <= new[1] < GRID_WIDTH):
            return pos
        if self.grid[new] == 1:
            return pos
        return new

    def _get_valid_moves(self, pos):
        """Return all possible moves (excluding walls)."""
        y, x = pos
        moves = []
        for action, (dy, dx) in enumerate([(-1, 0), (1, 0), (0, -1), (0, 1)]):
            ny, nx = y + dy, x + dx
            if 0 <= ny < GRID_HEIGHT and 0 <= nx < GRID_WIDTH and self.grid[ny, nx] != 1:
                moves.append(((ny, nx), action))
        return moves

    def _move_ghosts(self):
        """Ghosts chase Pac-Man with 80% chance, random move with 20% chance."""
        new_positions = []
        new_under = {}
        py, px = self.pacman_pos

        for gpos in self.ghost_positions:
            # Restore what was under the ghost before moving
            self.grid[gpos] = self.ghost_under.get(gpos, 0)
            gy, gx = gpos

            # Get all valid moves
            valid_moves = self._get_valid_moves(gpos)
            if not valid_moves:
                new_positions.append(gpos)
                continue

            # --- 80% chase, 20% random ---
            if random.random() < 0.8:
                # Choose move that minimizes distance to Pac-Man
                best_move = gpos
                min_dist = float("inf")
                for (ny, nx), _ in valid_moves:
                    dist = np.hypot(py - ny, px - nx)
                    if dist < min_dist:
                        min_dist = dist
                        best_move = (ny, nx)
            else:
                # Choose a random valid move
                best_move, _ = random.choice(valid_moves)

            # Update ghost position
            new_under[best_move] = self.grid[best_move] if self.grid[best_move] == 2 else 0
            self.grid[best_move] = 3
            new_positions.append(best_move)

        self.ghost_positions = new_positions
        self.ghost_under = new_under

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        self.steps += 1
        reward = -0.01
        terminated = False
        truncated = False

        # Pac-Man movement (immediate direction change if valid)
        new_pos = self._move_entity(self.pacman_pos, action)
        if new_pos == self.pacman_pos:
            new_pos = self._move_entity(self.pacman_pos, self.last_action)
        else:
            self.last_action = action

        # Update grid for Pac-Man
        y, x = self.pacman_pos
        self.grid[y, x] = 0
        y, x = new_pos
        if self.grid[y, x] == 2:
            reward += 1
            self.score += 1
        self.grid[y, x] = 4
        self.pacman_pos = new_pos

        # 🕹️ Move ghosts only every 2 frames (half speed)
        self.ghost_move_counter += 1
        if self.ghost_move_counter >= 2:
            self._move_ghosts()
            self.ghost_move_counter = 0

        # Collision check
        if any(g == self.pacman_pos for g in self.ghost_positions):
            reward = -20
            terminated = True

        # Win condition
        if not np.any(self.grid == 2):
            reward += 100
            terminated = True

        obs = self.grid.copy()
        info = {"score": self.score, "steps": self.steps}
        if self.render_mode == "human":
            self.render()
        return obs, reward, terminated, truncated, info

    def render(self):
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            self.clock = pygame.time.Clock()

        self.screen.fill(BLACK)
        for y in range(GRID_HEIGHT):
            for x in range(GRID_WIDTH):
                rect = pygame.Rect(x * TILE_SIZE, y * TILE_SIZE, TILE_SIZE, TILE_SIZE)
                val = self.grid[y, x]
                if val == 1:
                    pygame.draw.rect(self.screen, BLUE, rect)
                elif val == 2:
                    pygame.draw.circle(self.screen, WHITE, rect.center, 3)
                elif val == 3:
                    pygame.draw.circle(self.screen, RED, rect.center, TILE_SIZE // 2 - 2)
                elif val == 4:
                    pygame.draw.circle(self.screen, YELLOW, rect.center, TILE_SIZE // 2)

        font = pygame.font.Font(None, 32)
        score_text = font.render(f"SCORE: {self.score}", True, YELLOW)
        self.screen.blit(score_text, (20, SCREEN_HEIGHT - 30))
        pygame.display.flip()
        self.clock.tick(FPS)

    def close(self):
        if self.screen:
            pygame.quit()
            self.screen = None