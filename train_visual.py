"""
Visual Training Mode for Connect Four Q-Learning AI
Watch the AI train and learn in real-time!
"""

import pygame
import numpy as np
import random
from connectfour import ConnectFour, ConnectFourAI

# Game constants
ROWS = 6
COLUMNS = 7
SQUARE_SIZE = 80
RADIUS = int(SQUARE_SIZE / 2 - 5)
WIDTH = COLUMNS * SQUARE_SIZE
HEIGHT = (ROWS + 1) * SQUARE_SIZE

# Stats panel
STATS_WIDTH = 350
TOTAL_WIDTH = WIDTH + STATS_WIDTH

# Colors
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
GREEN = (0, 255, 0)
GRAY = (128, 128, 128)
DARK_GRAY = (64, 64, 64)


class VisualTrainer:
    """Handles visual training of the AI."""
    
    def __init__(self, games_to_train, alpha=0.5, epsilon=0.1):
        """
        Initialize visual trainer.
        
        Args:
            games_to_train: Total number of games to train
            alpha: Learning rate
            epsilon: Exploration rate
        """
        pygame.init()
        self.screen = pygame.display.set_mode((TOTAL_WIDTH, HEIGHT))
        pygame.display.set_caption("Connect Four - AI Visual Training")
        
        self.ai = ConnectFourAI(alpha=alpha, epsilon=epsilon)
        self.games_to_train = games_to_train
        self.games_played = 0
        
        # Training stats
        self.player1_wins = 0
        self.player2_wins = 0
        self.draws = 0
        self.last_100_results = []  # Track last 100 games
        
        # Speed control
        self.speed = 1  # 1=slow, 2=medium, 3=fast, 4=very fast, 5=instant
        self.paused = False
        self.show_q_values = False
        
        # Fonts
        self.title_font = pygame.font.SysFont("monospace", 24, bold=True)
        self.stat_font = pygame.font.SysFont("monospace", 18)
        self.small_font = pygame.font.SysFont("monospace", 14)
        self.tiny_font = pygame.font.SysFont("monospace", 12)
        
    def draw_board(self, game):
        """Draw the Connect Four board."""
        for c in range(COLUMNS):
            for r in range(ROWS):
                # Draw blue board
                pygame.draw.rect(self.screen, BLUE,
                               (c * SQUARE_SIZE, r * SQUARE_SIZE + SQUARE_SIZE,
                                SQUARE_SIZE, SQUARE_SIZE))
                # Draw empty circles
                pygame.draw.circle(self.screen, BLACK,
                                 (int(c * SQUARE_SIZE + SQUARE_SIZE / 2),
                                  int(r * SQUARE_SIZE + SQUARE_SIZE + SQUARE_SIZE / 2)),
                                 RADIUS)
        
        # Draw pieces
        for c in range(COLUMNS):
            for r in range(ROWS):
                if game.board[r][c] == 1:
                    pygame.draw.circle(self.screen, RED,
                                     (int(c * SQUARE_SIZE + SQUARE_SIZE / 2),
                                      HEIGHT - int(r * SQUARE_SIZE + SQUARE_SIZE / 2)),
                                     RADIUS)
                elif game.board[r][c] == 2:
                    pygame.draw.circle(self.screen, YELLOW,
                                     (int(c * SQUARE_SIZE + SQUARE_SIZE / 2),
                                      HEIGHT - int(r * SQUARE_SIZE + SQUARE_SIZE / 2)),
                                     RADIUS)
    
    def draw_stats(self, game, last_action=None):
        """Draw statistics panel."""
        # Background
        pygame.draw.rect(self.screen, DARK_GRAY, (WIDTH, 0, STATS_WIDTH, HEIGHT))
        
        y = 10
        
        # Title
        title = self.title_font.render("Training Stats", True, WHITE)
        self.screen.blit(title, (WIDTH + 10, y))
        y += 40
        
        # Progress
        progress_text = f"Game: {self.games_played}/{self.games_to_train}"
        progress = self.stat_font.render(progress_text, True, WHITE)
        self.screen.blit(progress, (WIDTH + 10, y))
        y += 25
        
        # Progress bar
        progress_pct = self.games_played / self.games_to_train if self.games_to_train > 0 else 0
        bar_width = STATS_WIDTH - 40
        pygame.draw.rect(self.screen, GRAY, (WIDTH + 20, y, bar_width, 20))
        pygame.draw.rect(self.screen, GREEN, (WIDTH + 20, y, int(bar_width * progress_pct), 20))
        y += 35
        
        # Wins
        p1_text = f"Player 1 (Red): {self.player1_wins}"
        p2_text = f"Player 2 (Yellow): {self.player2_wins}"
        draws_text = f"Draws: {self.draws}"
        
        p1_label = self.stat_font.render(p1_text, True, RED)
        p2_label = self.stat_font.render(p2_text, True, YELLOW)
        draws_label = self.stat_font.render(draws_text, True, WHITE)
        
        self.screen.blit(p1_label, (WIDTH + 10, y))
        y += 25
        self.screen.blit(p2_label, (WIDTH + 10, y))
        y += 25
        self.screen.blit(draws_label, (WIDTH + 10, y))
        y += 35
        
        # Win rate (last 100 games)
        if len(self.last_100_results) > 0:
            p1_recent = self.last_100_results.count(1)
            p2_recent = self.last_100_results.count(2)
            draws_recent = self.last_100_results.count(0)
            total_recent = len(self.last_100_results)
            
            recent_title = self.stat_font.render("Last 100 Games:", True, WHITE)
            self.screen.blit(recent_title, (WIDTH + 10, y))
            y += 25
            
            p1_pct = f"P1: {p1_recent}/{total_recent} ({p1_recent/total_recent*100:.1f}%)"
            p2_pct = f"P2: {p2_recent}/{total_recent} ({p2_recent/total_recent*100:.1f}%)"
            
            p1_pct_label = self.small_font.render(p1_pct, True, RED)
            p2_pct_label = self.small_font.render(p2_pct, True, YELLOW)
            
            self.screen.blit(p1_pct_label, (WIDTH + 20, y))
            y += 20
            self.screen.blit(p2_pct_label, (WIDTH + 20, y))
            y += 30
        
        # Q-values learned
        q_text = f"Q-values: {len(self.ai.q):,}"
        q_label = self.stat_font.render(q_text, True, WHITE)
        self.screen.blit(q_label, (WIDTH + 10, y))
        y += 35
        
        # Current player
        current_player = f"Turn: Player {game.player}"
        color = RED if game.player == 1 else YELLOW
        current_label = self.stat_font.render(current_player, True, color)
        self.screen.blit(current_label, (WIDTH + 10, y))
        y += 30
        
        # Last action
        if last_action is not None:
            action_text = f"Last move: Col {last_action}"
            action_label = self.small_font.render(action_text, True, WHITE)
            self.screen.blit(action_label, (WIDTH + 10, y))
            y += 30
        
        # Speed control
        pygame.draw.line(self.screen, GRAY, (WIDTH + 10, y), (WIDTH + STATS_WIDTH - 10, y), 2)
        y += 15
        
        speed_title = self.stat_font.render("Controls:", True, WHITE)
        self.screen.blit(speed_title, (WIDTH + 10, y))
        y += 25
        
        speed_text = f"Speed: {'█' * self.speed}{'░' * (5 - self.speed)}"
        speed_label = self.small_font.render(speed_text, True, GREEN)
        self.screen.blit(speed_label, (WIDTH + 10, y))
        y += 20
        
        pause_text = "PAUSED" if self.paused else "Running"
        pause_color = YELLOW if self.paused else GREEN
        pause_label = self.small_font.render(pause_text, True, pause_color)
        self.screen.blit(pause_label, (WIDTH + 10, y))
        y += 25
        
        # Instructions
        instructions = [
            "↑/↓: Speed",
            "SPACE: Pause",
            "Q: Show Q-values",
            "ESC: Stop"
        ]
        
        for instruction in instructions:
            inst_label = self.tiny_font.render(instruction, True, GRAY)
            self.screen.blit(inst_label, (WIDTH + 10, y))
            y += 18
    
    def draw_q_values(self, game, current_player):
        """Draw Q-values for current state."""
        if not self.show_q_values:
            return
        
        # Normalize board for AI
        if current_player == 1:
            ai_board = game.board.copy()
        else:
            ai_board = np.where(game.board == 1, 2, np.where(game.board == 2, 1, 0))
        
        available = ConnectFour.available_actions(game.board)
        
        # Draw Q-values above each column
        for col in range(COLUMNS):
            if col in available:
                q_val = self.ai.get_q_value(ai_board, col)
                text = f"{q_val:.2f}"
                color = GREEN if q_val > 0 else (RED if q_val < 0 else WHITE)
                label = self.tiny_font.render(text, True, color)
                self.screen.blit(label, (col * SQUARE_SIZE + 10, 5))
    
    def handle_events(self):
        """Handle pygame events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_UP:
                    self.speed = min(5, self.speed + 1)
                elif event.key == pygame.K_DOWN:
                    self.speed = max(1, self.speed - 1)
                elif event.key == pygame.K_q:
                    self.show_q_values = not self.show_q_values
        
        return True
    
    def get_delay(self):
        """Get delay based on speed setting."""
        delays = {
            1: 1000,  # 1 second per move
            2: 500,   # 0.5 seconds
            3: 200,   # 0.2 seconds
            4: 50,    # 0.05 seconds
            5: 0      # Instant
        }
        return delays[self.speed]
    
    def train(self):
        """Main training loop with visualization."""
        running = True
        
        while running and self.games_played < self.games_to_train:
            # Handle events
            running = self.handle_events()
            if not running:
                break
            
            # Skip if paused
            if self.paused:
                pygame.time.wait(100)
                continue
            
            # Play one game
            game = ConnectFour()
            last_action = None
            
            while not game.game_over and running:
                # Handle events during game
                running = self.handle_events()
                if not running:
                    break
                
                if self.paused:
                    pygame.time.wait(100)
                    continue
                
                current_player = game.player
                
                # Normalize board
                if current_player == 1:
                    normalized_board = game.board.copy()
                else:
                    normalized_board = np.where(game.board == 1, 2,
                                               np.where(game.board == 2, 1, 0))
                
                old_state = tuple(map(tuple, normalized_board))
                action = self.ai.choose_action(normalized_board)
                
                if action is None:
                    break
                
                last_action = action
                
                # Make move
                game.make_move(action)
                
                # Compute new_board_current (normalized for current after move)
                if current_player == 1:
                    new_board_current = game.board.copy()
                else:
                    new_board_current = np.where(game.board == 1, 2,
                                                 np.where(game.board == 2, 1, 0))
                
                # Opponent's normalized state (swap labels for opponent to see self as 1)
                opponent_board = np.where(new_board_current == 1, 2,
                                          np.where(new_board_current == 2, 1, 0))
                new_state = tuple(map(tuple, opponent_board))
                
                # Determine reward and terminal
                reward = 0
                terminal = game.game_over
                if terminal:
                    if game.winner == current_player:
                        reward = 1
                    else:
                        reward = -1  # Loss for current player
                
                # Update Q-value
                self.ai.update(old_state, action, new_state, reward, terminal)
                
                # Update visualization
                self.screen.fill(BLACK)
                self.draw_board(game)
                self.draw_q_values(game, game.player)  # Show for next player
                self.draw_stats(game, last_action)
                pygame.display.update()
                
                # Delay based on speed
                delay = self.get_delay()
                if delay > 0:
                    pygame.time.wait(delay)
            
            # Update stats
            self.games_played += 1
            
            if game.winner == 1:
                self.player1_wins += 1
                self.last_100_results.append(1)
            elif game.winner == 2:
                self.player2_wins += 1
                self.last_100_results.append(2)
            else:
                self.draws += 1
                self.last_100_results.append(0)
            
            # Keep only last 100 results
            if len(self.last_100_results) > 100:
                self.last_100_results.pop(0)
            
            # Show final board briefly
            if game.game_over and self.speed < 5:
                self.screen.fill(BLACK)
                self.draw_board(game)
                self.draw_stats(game, last_action)
                
                # Show winner
                if game.winner is not None:
                    winner_text = f"Player {game.winner} wins!"
                    color = RED if game.winner == 1 else YELLOW
                    label = self.title_font.render(winner_text, True, color)
                    self.screen.blit(label, (WIDTH // 2 - 100, HEIGHT // 2))
                else:
                    draw_text = self.title_font.render("Draw!", True, WHITE)
                    self.screen.blit(draw_text, (WIDTH // 2 - 100, HEIGHT // 2))
                
                pygame.display.update()
                pygame.time.wait(max(100, self.get_delay() // 2))
        
        # Training complete
        pygame.quit()
        return self.ai


def main():
    """Main function."""
    print("="*80)
    print("Connect Four - Visual Training Mode")
    print("="*80)
    print("\nWatch the AI learn to play Connect Four!")
    print("\nControls during training:")
    print("  ↑/↓ arrows: Adjust speed")
    print("  SPACE: Pause/Resume")
    print("  Q: Toggle Q-value display")
    print("  ESC: Stop training and save")
    print("\nSpeed levels:")
    print("  1 = Slow (1 move/second) - Best for watching strategy")
    print("  2 = Medium (2 moves/second)")
    print("  3 = Fast (5 moves/second)")
    print("  4 = Very Fast (20 moves/second)")
    print("  5 = Instant (no delay) - For quick training")
    
    games = int(input("\nHow many games to train? (e.g., 10000): "))
    alpha = float(input("Learning rate (0.5 recommended, press Enter): ") or 0.5)
    epsilon = float(input("Exploration rate (0.1 recommended, press Enter): ") or 0.1)
    
    print(f"\nStarting visual training for {games} games...")
    print("Press UP arrow to speed up, DOWN to slow down")
    print("Press Q to see Q-values (numbers above columns)")
    
    trainer = VisualTrainer(games, alpha, epsilon)
    trained_ai = trainer.train()
    
    print("\n" + "="*80)
    print("Training Complete!")
    print("="*80)
    print(f"Games played: {trainer.games_played}")
    if trainer.games_played > 0:
        print(f"Player 1 wins: {trainer.player1_wins} ({trainer.player1_wins/trainer.games_played*100:.1f}%)")
        print(f"Player 2 wins: {trainer.player2_wins} ({trainer.player2_wins/trainer.games_played*100:.1f}%)")
        print(f"Draws: {trainer.draws} ({trainer.draws/trainer.games_played*100:.1f}%)")
    print(f"Q-values learned: {len(trained_ai.q):,}")
    
    save = input("\nSave this model? (y/n): ").strip().lower()
    if save == 'y':
        from model_manager import ModelManager
        manager = ModelManager()
        manager.save_model(trained_ai, trainer.games_played, alpha, epsilon)
        print("Model saved!")
    else:
        print("Model not saved.")


if __name__ == "__main__":
    main()