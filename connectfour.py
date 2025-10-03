import random
import copy
import pygame
import numpy as np

# GLOBAL VARIABLES
ROWS = 6
COLUMNS = 7
CONNECT_X = 4
SQUARE_SIZE = 100
RADIUS = int(SQUARE_SIZE / 2 - 5)
WIDTH = COLUMNS * SQUARE_SIZE
HEIGHT = (ROWS + 1) * SQUARE_SIZE

# COLORS
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)


class ConnectFour:
    def __init__(self):
        """
        Initialize game board.
        Each game board has:
            - `board`: a 2D numpy array representing the game state
            - `player`: 1 or 2 to indicate which player's turn
            - `winner`: None, 1, or 2 to indicate who the winner is
        """
        self.board = np.zeros((ROWS, COLUMNS))
        self.player = 1
        self.winner = None
        self.game_over = False

    @classmethod
    def available_actions(cls, board):
        """
        Returns all available actions (valid columns) for the current board state.
        An action is valid if the top row of that column is empty.
        """
        actions = []
        for col in range(COLUMNS):
            if board[ROWS - 1][col] == 0:  # Check if top row is empty
                actions.append(col)
        return actions

    @classmethod
    def other_player(cls, player):
        """Returns the other player (1 -> 2, 2 -> 1)"""
        return 1 if player == 2 else 2

    def switch_player(self):
        """Switch the current player to the other player."""
        self.player = ConnectFour.other_player(self.player)

    def get_next_open_row(self, col):
        """Find the lowest empty row in a column."""
        for row in range(ROWS):
            if self.board[row][col] == 0:
                return row
        return None

    def make_move(self, col):
        """
        Make a move in the specified column.
        Returns True if move was successful, False otherwise.
        """
        if col not in self.available_actions(self.board):
            return False

        row = self.get_next_open_row(col)
        if row is not None:
            self.board[row][col] = self.player
            
            # Check for winner
            if self.check_win(self.player):
                self.winner = self.player
                self.game_over = True
            elif self.is_board_full():
                self.game_over = True  # Draw
            else:
                self.switch_player()
            
            return True
        return False

    def check_win(self, player):
        """Check if the specified player has won."""
        # Check horizontal
        for c in range(COLUMNS - 3):
            for r in range(ROWS):
                if (self.board[r][c] == player and 
                    self.board[r][c + 1] == player and 
                    self.board[r][c + 2] == player and 
                    self.board[r][c + 3] == player):
                    return True

        # Check vertical
        for c in range(COLUMNS):
            for r in range(ROWS - 3):
                if (self.board[r][c] == player and 
                    self.board[r + 1][c] == player and 
                    self.board[r + 2][c] == player and 
                    self.board[r + 3][c] == player):
                    return True

        # Check positive diagonal
        for c in range(COLUMNS - 3):
            for r in range(ROWS - 3):
                if (self.board[r][c] == player and 
                    self.board[r + 1][c + 1] == player and 
                    self.board[r + 2][c + 2] == player and 
                    self.board[r + 3][c + 3] == player):
                    return True

        # Check negative diagonal
        for c in range(COLUMNS - 3):
            for r in range(3, ROWS):
                if (self.board[r][c] == player and 
                    self.board[r - 1][c + 1] == player and 
                    self.board[r - 2][c + 2] == player and 
                    self.board[r - 3][c + 3] == player):
                    return True

        return False

    def is_board_full(self):
        """Check if the board is completely full."""
        return len(self.available_actions(self.board)) == 0

    def get_state(self):
        """Return the current board state as a tuple (for use as dictionary key)."""
        return tuple(map(tuple, self.board))


class ConnectFourAI:
    def __init__(self, alpha=0.5, epsilon=0.1, gamma=1.0):
        """
        Initialize AI with Q-learning parameters.
        
        Args:
            alpha: Learning rate (0-1) - how much to update Q-values
            epsilon: Exploration rate (0-1) - probability of random action
            gamma: Discount factor (0-1) - future reward importance
        """
        self.q = dict()  # Q-learning dictionary: (state, action) -> Q-value
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma

    def _canonicalize(self, state, action=None):
        """Return a canonical (state, action) pair folded over horizontal symmetry.

        This reduces the state space by treating mirrored boards as the same scenario.
        Action is mirrored as well when the reflected board is the chosen canonical view.
        """
        board = np.asarray(state, dtype=int)
        if board.shape != (ROWS, COLUMNS):
            board = board.reshape((ROWS, COLUMNS))

        mirrored = np.fliplr(board)
        state_tuple = tuple(map(tuple, board))
        mirrored_tuple = tuple(map(tuple, mirrored))

        use_mirrored = mirrored_tuple < state_tuple

        if use_mirrored:
            state_tuple = mirrored_tuple
            if action is not None:
                action = COLUMNS - 1 - action
        elif mirrored_tuple == state_tuple and action is not None:
            mirrored_action = COLUMNS - 1 - action
            if mirrored_action < action:
                action = mirrored_action

        return state_tuple, action

    def get_q_value(self, state, action):
        """
        Return the Q-value for a state-action pair.
        Returns 0 if the pair hasn't been seen before.
        """
        state_key, action_key = self._canonicalize(state, action)
        return self.q.get((state_key, action_key), 0)

    def update_q_value(self, state, action, old_q, td_target):
        """
        Update Q-value using the formula:
        Q(s,a) = old_q + alpha * (td_target - old_q)
        """
        state_key, action_key = self._canonicalize(state, action)
        self.q[(state_key, action_key)] = old_q + self.alpha * (td_target - old_q)

    def best_future_reward(self, state):
        """
        Return the maximum Q-value for all available actions in the given state.
        Returns 0 if no actions are available.
        """
        board = np.asarray(state, dtype=int)
        if board.shape != (ROWS, COLUMNS):
            board = board.reshape((ROWS, COLUMNS))
        actions = ConnectFour.available_actions(board)

        if not actions:
            return 0

        max_reward = max(
            self.q.get(self._canonicalize(board, action), 0)
            for action in actions
        )
        return max_reward

    def update(self, old_state, action, new_state, reward, terminal=False):
        """
        Update Q-learning model based on an action and its result.
        new_state is the normalized state from the opponent's perspective (for self-play learning).
        """
        old_q = self.get_q_value(old_state, action)
        max_future = 0 if terminal else self.best_future_reward(new_state)
        td_target = reward + self.gamma * max_future
        self.update_q_value(old_state, action, old_q, td_target)

    def choose_action(self, state, epsilon=True):
        """
        Choose an action using epsilon-greedy strategy.

        Args:
            state: Current board state (numpy array or tuple)
            epsilon: If True, use epsilon-greedy; if False, always choose best action

        Returns:
            Column number to play (0-6)
        """
        board = np.asarray(state, dtype=int)
        if board.shape != (ROWS, COLUMNS):
            board = board.reshape((ROWS, COLUMNS))
        actions = ConnectFour.available_actions(board)
        if not actions:
            return None

        # Exploration: choose random action
        if epsilon and random.random() < self.epsilon:
            return random.choice(actions)

        # Exploitation: choose best action based on canonicalized Q-values
        q_lookup = []
        for action in actions:
            key = self._canonicalize(board, action)
            q_lookup.append((action, self.q.get(key, 0)))

        max_value = max(value for _, value in q_lookup)
        best_actions = [action for action, value in q_lookup if abs(value - max_value) < 1e-9]
        return random.choice(best_actions)


def train(n):
    """
    Train an AI by playing n games against itself.
    AI learns to play as player 1. When playing as player 2, 
    board is normalized so AI always sees itself as 1.
    """
    ai = ConnectFourAI()

    for i in range(n):
        if (i + 1) % 1000 == 0:
            print(f"Playing training game {i + 1}")
        game = ConnectFour()

        while not game.game_over:
            current_player = game.player
            
            # Normalize board so current player always sees itself as 1
            if current_player == 1:
                normalized_board = game.board.copy()
            else:
                # Swap 1s and 2s so player 2 sees itself as 1
                normalized_board = np.where(game.board == 1, 2, 
                                           np.where(game.board == 2, 1, 0))
            
            # Get state and choose action on normalized board
            old_state = tuple(map(tuple, normalized_board))
            action = ai.choose_action(normalized_board)
            
            if action is None:
                break

            # Make move on actual game board
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
            ai.update(old_state, action, new_state, reward, terminal)

    print("Done training")
    print(f"Total Q-values learned: {len(ai.q)}")
    return ai


def play_graphical(ai, human_player=None):
    """
    Play a graphical game against the AI using Pygame.
    
    Args:
        ai: Trained ConnectFourAI instance
        human_player: 1 or 2 (None for random)
    """
    pygame.init()
    
    # If no player order set, choose randomly
    if human_player is None:
        human_player = random.randint(1, 2)
    
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Connect Four - Q-Learning AI")
    myfont = pygame.font.SysFont("monospace", 75)
    smallfont = pygame.font.SysFont("monospace", 40)
    
    game = ConnectFour()
    
    def draw_board():
        """Draw the game board."""
        for c in range(COLUMNS):
            for r in range(ROWS):
                # Draw blue board
                pygame.draw.rect(screen, BLUE, 
                               (c * SQUARE_SIZE, r * SQUARE_SIZE + SQUARE_SIZE, 
                                SQUARE_SIZE, SQUARE_SIZE))
                # Draw black circles for empty spaces
                pygame.draw.circle(screen, BLACK, 
                                 (int(c * SQUARE_SIZE + SQUARE_SIZE / 2), 
                                  int(r * SQUARE_SIZE + SQUARE_SIZE + SQUARE_SIZE / 2)), 
                                 RADIUS)
        
        # Draw pieces
        for c in range(COLUMNS):
            for r in range(ROWS):
                if game.board[r][c] == 1:
                    pygame.draw.circle(screen, RED, 
                                     (int(c * SQUARE_SIZE + SQUARE_SIZE / 2), 
                                      HEIGHT - int(r * SQUARE_SIZE + SQUARE_SIZE / 2)), 
                                     RADIUS)
                elif game.board[r][c] == 2:
                    pygame.draw.circle(screen, YELLOW, 
                                     (int(c * SQUARE_SIZE + SQUARE_SIZE / 2), 
                                      HEIGHT - int(r * SQUARE_SIZE + SQUARE_SIZE / 2)), 
                                     RADIUS)
        
        pygame.display.update()
    
    draw_board()
    
    # If AI goes first, make its move
    if game.player != human_player:
        pygame.time.wait(500)
        # Normalize board for AI
        if game.player == 1:
            ai_board = game.board.copy()
        else:
            ai_board = np.where(game.board == 1, 2, np.where(game.board == 2, 1, 0))
        
        ai_col = ai.choose_action(ai_board, epsilon=False)
        if ai_col is not None:
            game.make_move(ai_col)
            draw_board()
    
    running = True
    while running and not game.game_over:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            if event.type == pygame.MOUSEMOTION and game.player == human_player:
                pygame.draw.rect(screen, BLACK, (0, 0, WIDTH, SQUARE_SIZE))
                posx = event.pos[0]
                color = RED if human_player == 1 else YELLOW
                pygame.draw.circle(screen, color, (posx, int(SQUARE_SIZE / 2)), RADIUS)
                pygame.display.update()
            
            if event.type == pygame.MOUSEBUTTONDOWN and game.player == human_player:
                pygame.draw.rect(screen, BLACK, (0, 0, WIDTH, SQUARE_SIZE))
                posx = event.pos[0]
                col = int(posx // SQUARE_SIZE)
                
                # Human move
                if game.make_move(col):
                    draw_board()
                    
                    if not game.game_over:
                        # AI move
                        pygame.time.wait(500)
                        
                        # Normalize board for AI
                        if game.player == 1:
                            ai_board = game.board.copy()
                        else:
                            ai_board = np.where(game.board == 1, 2, 
                                              np.where(game.board == 2, 1, 0))
                        
                        ai_col = ai.choose_action(ai_board, epsilon=False)
                        if ai_col is not None:
                            game.make_move(ai_col)
                            draw_board()
    
    # Display winner
    if game.winner is not None:
        winner_text = "You win!" if game.winner == human_player else "AI wins!"
        color = RED if game.winner == 1 else YELLOW
        label = myfont.render(winner_text, 1, color)
        screen.blit(label, (40, 10))
        pygame.display.update()
    else:
        label = smallfont.render("Draw!", 1, BLUE)
        screen.blit(label, (WIDTH // 2 - 60, 10))
        pygame.display.update()
    
    pygame.time.wait(3000)
    pygame.quit()


def play_console(ai, human_player=None):
    """
    Play a console-based game against the AI.
    """
    if human_player is None:
        human_player = random.randint(1, 2)
    
    game = ConnectFour()
    print(f"\nYou are Player {human_player}")
    print("AI is Player", ConnectFour.other_player(human_player))
    
    while not game.game_over:
        # Display board
        print("\n" + "=" * 30)
        print(np.flip(game.board, 0))
        print("Columns: 0  1  2  3  4  5  6")
        print(f"\nPlayer {game.player}'s turn")
        
        if game.player == human_player:
            # Human move
            while True:
                try:
                    col = int(input("Choose column (0-6): "))
                    if col in game.available_actions(game.board):
                        game.make_move(col)
                        break
                    else:
                        print("Invalid column. Try again.")
                except ValueError:
                    print("Please enter a number between 0 and 6.")
        else:
            # AI move
            print("AI is thinking...")
            
            # Normalize board for AI
            if game.player == 1:
                ai_board = game.board.copy()
            else:
                ai_board = np.where(game.board == 1, 2, np.where(game.board == 2, 1, 0))
            
            col = ai.choose_action(ai_board, epsilon=False)
            if col is not None:
                game.make_move(col)
                print(f"AI chose column {col}")
    
    # Display final board
    print("\n" + "=" * 30)
    print(np.flip(game.board, 0))
    print("=" * 30)
    
    if game.winner is not None:
        if game.winner == human_player:
            print("\n🎉 You win! Congratulations!")
        else:
            print("\n🤖 AI wins! Better luck next time!")
    else:
        print("\n🤝 It's a draw!")

"""
Enhanced ConnectFourAI class with additional features for better Q-Learning
This can replace the ConnectFourAI class in connectfour.py
"""

import numpy as np
import random
from typing import Optional, Tuple, List, Dict


class ConnectFourAI:
    def __init__(self, alpha=0.5, epsilon=0.1, gamma=1.0, 
                 epsilon_decay=0.995, epsilon_min=0.01):
        """
        Initialize AI with Q-learning parameters.
        
        Args:
            alpha: Learning rate (0-1) - how much to update Q-values
            epsilon: Exploration rate (0-1) - probability of random action
            gamma: Discount factor (0-1) - future reward importance
            epsilon_decay: Rate at which epsilon decreases each episode
            epsilon_min: Minimum epsilon value
        """
        self.q: Dict[Tuple, float] = dict()  # Q-learning dictionary
        self.alpha = alpha
        self.epsilon = epsilon
        self.initial_epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.gamma = gamma
        self.training_episodes = 0
        
        # Track statistics for analysis
        self.stats = {
            'updates': 0,
            'wins': 0,
            'losses': 0,
            'draws': 0
        }
    
    def decay_epsilon(self):
        """Decay epsilon after each training episode."""
        self.epsilon = max(self.epsilon_min, 
                          self.epsilon * self.epsilon_decay)
        self.training_episodes += 1
    
    def reset_epsilon(self):
        """Reset epsilon to initial value (useful for retraining)."""
        self.epsilon = self.initial_epsilon
        self.training_episodes = 0
    
    def _canonicalize(self, state, action=None):
        """Return a canonical (state, action) pair folded over horizontal symmetry.
        
        This reduces the state space by treating mirrored boards as the same scenario.
        Action is mirrored as well when the reflected board is the chosen canonical view.
        """
        ROWS, COLUMNS = 6, 7
        board = np.asarray(state, dtype=int)
        if board.shape != (ROWS, COLUMNS):
            board = board.reshape((ROWS, COLUMNS))

        mirrored = np.fliplr(board)
        state_tuple = tuple(map(tuple, board))
        mirrored_tuple = tuple(map(tuple, mirrored))

        use_mirrored = mirrored_tuple < state_tuple

        if use_mirrored:
            state_tuple = mirrored_tuple
            if action is not None:
                action = COLUMNS - 1 - action
        elif mirrored_tuple == state_tuple and action is not None:
            mirrored_action = COLUMNS - 1 - action
            if mirrored_action < action:
                action = mirrored_action

        return state_tuple, action
    
    def get_q_value(self, state, action):
        """
        Return the Q-value for a state-action pair.
        Returns 0 if the pair hasn't been seen before.
        """
        state_key, action_key = self._canonicalize(state, action)
        return self.q.get((state_key, action_key), 0)
    
    def get_all_q_values(self, state, available_actions):
        """
        Get Q-values for all available actions in a state.
        
        Args:
            state: Current board state
            available_actions: List of valid actions
            
        Returns:
            Dict mapping actions to Q-values
        """
        return {action: self.get_q_value(state, action) 
                for action in available_actions}
    
    def update_q_value(self, state, action, old_q, td_target):
        """
        Update Q-value using the formula:
        Q(s,a) = old_q + alpha * (td_target - old_q)
        """
        state_key, action_key = self._canonicalize(state, action)
        self.q[(state_key, action_key)] = old_q + self.alpha * (td_target - old_q)
        self.stats['updates'] += 1
    
    def best_future_reward(self, state):
        """
        Return the maximum Q-value for all available actions in the given state.
        Returns 0 if no actions are available.
        """
        from connectfour import ConnectFour
        
        board = np.asarray(state, dtype=int)
        if board.shape != (6, 7):
            board = board.reshape((6, 7))
        actions = ConnectFour.available_actions(board)

        if not actions:
            return 0

        max_reward = max(
            self.q.get(self._canonicalize(board, action), 0)
            for action in actions
        )
        return max_reward
    
    def update(self, old_state, action, new_state, reward, terminal=False):
        """
        Update Q-learning model based on an action and its result.
        
        Args:
            old_state: State before action
            action: Action taken
            new_state: State after action (from opponent's perspective)
            reward: Immediate reward received
            terminal: Whether this is a terminal state
        """
        old_q = self.get_q_value(old_state, action)
        max_future = 0 if terminal else self.best_future_reward(new_state)
        td_target = reward + self.gamma * max_future
        self.update_q_value(old_state, action, old_q, td_target)
        
        # Track statistics
        if terminal:
            if reward > 0:
                self.stats['wins'] += 1
            elif reward < 0:
                self.stats['losses'] += 1
            else:
                self.stats['draws'] += 1
    
    def choose_action(self, state, epsilon=True, temperature=None):
        """
        Choose an action using epsilon-greedy or softmax strategy.

        Args:
            state: Current board state
            epsilon: If True, use epsilon-greedy; if False, always choose best
            temperature: If provided, use softmax selection with this temperature
                        (higher = more random, lower = more deterministic)

        Returns:
            Column number to play (0-6) or None if no moves available
        """
        from connectfour import ConnectFour
        
        board = np.asarray(state, dtype=int)
        if board.shape != (6, 7):
            board = board.reshape((6, 7))
        actions = ConnectFour.available_actions(board)
        
        if not actions:
            return None

        # Softmax selection (alternative to epsilon-greedy)
        if temperature is not None and temperature > 0:
            q_values = [self.get_q_value(board, a) for a in actions]
            # Convert Q-values to probabilities using softmax
            exp_values = np.exp(np.array(q_values) / temperature)
            probabilities = exp_values / exp_values.sum()
            return np.random.choice(actions, p=probabilities)

        # Epsilon-greedy selection
        if epsilon and random.random() < self.epsilon:
            # Exploration: random action
            return random.choice(actions)

        # Exploitation: choose best action
        q_lookup = []
        for action in actions:
            key = self._canonicalize(board, action)
            q_lookup.append((action, self.q.get(key, 0)))

        max_value = max(value for _, value in q_lookup)
        best_actions = [action for action, value in q_lookup 
                       if abs(value - max_value) < 1e-9]
        return random.choice(best_actions)
    
    def get_action_probabilities(self, state, temperature=1.0):
        """
        Get probability distribution over actions using softmax.
        
        Args:
            state: Current board state
            temperature: Softmax temperature
            
        Returns:
            Dict mapping actions to probabilities
        """
        from connectfour import ConnectFour
        
        board = np.asarray(state, dtype=int)
        if board.shape != (6, 7):
            board = board.reshape((6, 7))
        actions = ConnectFour.available_actions(board)
        
        if not actions:
            return {}
        
        q_values = {a: self.get_q_value(board, a) for a in actions}
        
        if temperature <= 0:
            # Deterministic: choose best action(s)
            max_q = max(q_values.values())
            best_actions = [a for a, q in q_values.items() 
                          if abs(q - max_q) < 1e-9]
            return {a: 1.0/len(best_actions) if a in best_actions else 0.0 
                   for a in actions}
        
        # Softmax probabilities
        exp_values = {a: np.exp(q / temperature) for a, q in q_values.items()}
        total = sum(exp_values.values())
        return {a: exp_values[a] / total for a in actions}
    
    def get_statistics(self):
        """Return training statistics."""
        return {
            'q_table_size': len(self.q),
            'training_episodes': self.training_episodes,
            'current_epsilon': self.epsilon,
            'total_updates': self.stats['updates'],
            'wins': self.stats['wins'],
            'losses': self.stats['losses'],  
            'draws': self.stats['draws']
        }
    
    def save_q_table_analysis(self, filepath='q_analysis.txt'):
        """
        Save analysis of Q-table for debugging and insight.
        
        Args:
            filepath: Where to save the analysis
        """
        with open(filepath, 'w') as f:
            f.write("Q-Learning Analysis\n")
            f.write("=" * 50 + "\n\n")
            
            stats = self.get_statistics()
            for key, value in stats.items():
                f.write(f"{key}: {value}\n")
            
            f.write("\n" + "=" * 50 + "\n")
            f.write("Top 20 Q-values:\n")
            f.write("-" * 30 + "\n")
            
            # Get top Q-values
            sorted_q = sorted(self.q.items(), key=lambda x: x[1], reverse=True)
            for i, ((state, action), value) in enumerate(sorted_q[:20]):
                f.write(f"{i+1}. Action {action}: Q={value:.4f}\n")
            
            f.write("\n" + "=" * 50 + "\n")
            f.write("Bottom 20 Q-values:\n")
            f.write("-" * 30 + "\n")
            
            for i, ((state, action), value) in enumerate(sorted_q[-20:]):
                f.write(f"{i+1}. Action {action}: Q={value:.4f}\n")
    
    def clone(self):
        """Create a deep copy of this AI."""
        import copy
        return copy.deepcopy(self)


def enhanced_train(n, alpha=0.5, epsilon=0.1, gamma=1.0, 
                   epsilon_decay=0.995, epsilon_min=0.01,
                   verbose=True, save_interval=1000):
    """
    Enhanced training function with epsilon decay and progress tracking.
    
    Args:
        n: Number of games to train
        alpha: Learning rate
        epsilon: Initial exploration rate
        gamma: Discount factor
        epsilon_decay: Rate of epsilon decay per episode
        epsilon_min: Minimum epsilon value
        verbose: Whether to print progress
        save_interval: Save statistics every N games
        
    Returns:
        Trained ConnectFourAI instance
    """
    from connectfour import ConnectFour
    import numpy as np
    
    ai = ConnectFourAI(alpha=alpha, epsilon=epsilon, gamma=gamma, 
                      epsilon_decay=epsilon_decay, epsilon_min=epsilon_min)
    
    win_history = []  # Track recent wins for performance monitoring
    
    for i in range(n):
        if verbose and (i + 1) % 100 == 0:
            recent_wins = sum(win_history[-100:]) if win_history else 0
            print(f"Game {i + 1}/{n} | Epsilon: {ai.epsilon:.4f} | "
                  f"Q-table size: {len(ai.q)} | Recent P1 wins: {recent_wins}%")
        
        if save_interval and (i + 1) % save_interval == 0:
            stats = ai.get_statistics()
            if verbose:
                print(f"  Stats - Wins: {stats['wins']}, Losses: {stats['losses']}, "
                      f"Draws: {stats['draws']}")
        
        game = ConnectFour()
        game_winner = None
        
        while not game.game_over:
            current_player = game.player
            
            # Normalize board so current player always sees itself as 1
            if current_player == 1:
                normalized_board = game.board.copy()
            else:
                normalized_board = np.where(game.board == 1, 2, 
                                           np.where(game.board == 2, 1, 0))
            
            old_state = tuple(map(tuple, normalized_board))
            action = ai.choose_action(normalized_board)
            
            if action is None:
                break
            
            game.make_move(action)
            
            # Compute normalized board after move
            if current_player == 1:
                new_board_current = game.board.copy()
            else:
                new_board_current = np.where(game.board == 1, 2, 
                                             np.where(game.board == 2, 1, 0))
            
            # Opponent's perspective
            opponent_board = np.where(new_board_current == 1, 2, 
                                      np.where(new_board_current == 2, 1, 0))
            new_state = tuple(map(tuple, opponent_board))
            
            # Determine reward
            reward = 0
            terminal = game.game_over
            if terminal:
                if game.winner == current_player:
                    reward = 1
                    game_winner = current_player
                else:
                    reward = -1
            
            # Update Q-value
            ai.update(old_state, action, new_state, reward, terminal)
        
        # Track performance
        win_history.append(1 if game_winner == 1 else 0)
        if len(win_history) > 100:
            win_history.pop(0)
        
        # Decay epsilon after each episode
        ai.decay_epsilon()
    
    if verbose:
        print("\nTraining complete!")
        stats = ai.get_statistics()
        print(f"Final statistics:")
        print(f"  Q-table size: {stats['q_table_size']}")
        print(f"  Total updates: {stats['total_updates']}")
        print(f"  Final epsilon: {stats['current_epsilon']:.4f}")
        print(f"  Win rate: {stats['wins']/(stats['wins']+stats['losses']+stats['draws'])*100:.1f}%")
    
    return ai