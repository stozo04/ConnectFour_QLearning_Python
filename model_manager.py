"""
Model Manager for Connect Four Q-Learning AI
Handles saving, loading, and caching of trained models
"""

import os
import pickle
import json
from datetime import datetime
import numpy as np
from connectfour import ConnectFourAI, ConnectFour
import time


class ModelManager:
    """Manages saving, loading, and caching of trained AI models."""
    
    def __init__(self, models_dir='trained_models'):
        """
        Initialize the ModelManager.
        
        Args:
            models_dir: Directory to store trained models
        """
        self.models_dir = models_dir
        self.metadata_file = os.path.join(models_dir, 'models_metadata.json')
        
        # Create models directory if it doesn't exist
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
            print(f"Created models directory: {models_dir}")
        
        # Load or create metadata
        self.metadata = self._load_metadata()
    
    def _load_metadata(self):
        """Load metadata about all trained models."""
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_metadata(self):
        """Save metadata to file."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def _get_model_filename(self, games, alpha, epsilon):
        """Generate a filename for a model based on its parameters."""
        return f"connectfour_ai_{games}games_a{alpha}_e{epsilon}.pkl"
    
    def _get_model_key(self, games, alpha, epsilon):
        """Generate a unique key for a model configuration."""
        return f"{games}_{alpha}_{epsilon}"
    
    def model_exists(self, games, alpha=0.5, epsilon=0.1):
        """
        Check if a trained model exists for the given parameters.
        
        Args:
            games: Number of training games
            alpha: Learning rate
            epsilon: Exploration rate
        
        Returns:
            bool: True if model exists, False otherwise
        """
        model_key = self._get_model_key(games, alpha, epsilon)
        
        if model_key in self.metadata:
            filename = self.metadata[model_key]['filename']
            filepath = os.path.join(self.models_dir, filename)
            return os.path.exists(filepath)
        
        return False
    
    def save_model(self, ai, games, alpha, epsilon, training_time=None):
        """
        Save a trained AI model with metadata.
        
        Args:
            ai: Trained ConnectFourAI instance
            games: Number of training games
            alpha: Learning rate used
            epsilon: Exploration rate used
            training_time: Time taken to train (seconds)
        """
        filename = self._get_model_filename(games, alpha, epsilon)
        filepath = os.path.join(self.models_dir, filename)
        model_key = self._get_model_key(games, alpha, epsilon)
        
        # Save the AI model
        with open(filepath, 'wb') as f:
            pickle.dump(ai, f)
        
        # Save metadata
        self.metadata[model_key] = {
            'filename': filename,
            'games': games,
            'alpha': alpha,
            'epsilon': epsilon,
            'q_values_count': len(ai.q),
            'training_time': training_time,
            'created_at': datetime.now().isoformat(),
            'file_size_mb': os.path.getsize(filepath) / (1024 * 1024)
        }
        
        self._save_metadata()
        print(f"\n✓ Model saved: {filename}")
        print(f"  Q-values learned: {len(ai.q)}")
        print(f"  File size: {self.metadata[model_key]['file_size_mb']:.2f} MB")
    
    def load_model(self, games, alpha=0.5, epsilon=0.1):
        """
        Load a trained AI model.
        
        Args:
            games: Number of training games
            alpha: Learning rate
            epsilon: Exploration rate
        
        Returns:
            ConnectFourAI instance or None if not found
        """
        model_key = self._get_model_key(games, alpha, epsilon)
        
        if model_key not in self.metadata:
            return None
        
        filename = self.metadata[model_key]['filename']
        filepath = os.path.join(self.models_dir, filename)
        
        if not os.path.exists(filepath):
            print(f"Warning: Model file not found: {filename}")
            return None
        
        with open(filepath, 'rb') as f:
            ai = pickle.load(f)
        
        print(f"\n✓ Model loaded: {filename}")
        print(f"  Q-values learned: {len(ai.q)}")
        print(f"  Training games: {games}")
        
        return ai
    
    def get_or_train(self, games, alpha=0.5, epsilon=0.1, force_retrain=False):
        """
        Get a trained model if it exists, otherwise train a new one.
        
        Args:
            games: Number of training games
            alpha: Learning rate
            epsilon: Exploration rate
            force_retrain: If True, retrain even if model exists
        
        Returns:
            ConnectFourAI instance
        """
        if not force_retrain and self.model_exists(games, alpha, epsilon):
            print(f"\n🎯 Found existing model trained on {games} games!")
            return self.load_model(games, alpha, epsilon)
        
        print(f"\n🏋️ Training new model with {games} games...")
        print(f"Parameters: alpha={alpha}, epsilon={epsilon}")
        
        start_time = time.time()
        
        # Train the model
        ai = self._train_model(games, alpha, epsilon)
        
        training_time = time.time() - start_time
        
        # Save the trained model
        self.save_model(ai, games, alpha, epsilon, training_time)
        
        print(f"\n✓ Training complete in {training_time:.1f} seconds")
        
        return ai
    
    def _train_model(self, games, alpha, epsilon):
        """
        Internal method to train a model.
        Uses board normalization so AI always sees itself as player 1.
        """
        ai = ConnectFourAI(alpha=alpha, epsilon=epsilon)
        
        for i in range(games):
            if (i + 1) % 1000 == 0:
                print(f"  Training game {i + 1}/{games} ({(i+1)/games*100:.1f}%)")
            
            game = ConnectFour()
            
            while not game.game_over:
                current_player = game.player
                
                # Normalize board so current player always sees itself as 1
                if current_player == 1:
                    normalized_board = game.board.copy()
                else:
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
        
        return ai
    
    def list_models(self):
        """List all available trained models."""
        if not self.metadata:
            print("\nNo trained models found.")
            return
        
        print("\n" + "="*80)
        print("Available Trained Models")
        print("="*80)
        print(f"{'Games':<10} {'Alpha':<8} {'Epsilon':<10} {'Q-Values':<12} {'Size (MB)':<12} {'Date'}")
        print("-"*80)
        
        for key, info in sorted(self.metadata.items(), key=lambda x: x[1]['games']):
            date = datetime.fromisoformat(info['created_at']).strftime('%Y-%m-%d %H:%M')
            print(f"{info['games']:<10} {info['alpha']:<8} {info['epsilon']:<10} "
                  f"{info['q_values_count']:<12} {info['file_size_mb']:<12.2f} {date}")
        
        print("="*80)
        
        total_size = sum(info['file_size_mb'] for info in self.metadata.values())
        print(f"\nTotal models: {len(self.metadata)}")
        print(f"Total disk space: {total_size:.2f} MB")
    
    def delete_model(self, games, alpha=0.5, epsilon=0.1):
        """
        Delete a trained model.
        
        Args:
            games: Number of training games
            alpha: Learning rate
            epsilon: Exploration rate
        
        Returns:
            bool: True if deleted, False if not found
        """
        model_key = self._get_model_key(games, alpha, epsilon)
        
        if model_key not in self.metadata:
            print(f"Model not found: {games} games, alpha={alpha}, epsilon={epsilon}")
            return False
        
        filename = self.metadata[model_key]['filename']
        filepath = os.path.join(self.models_dir, filename)
        
        # Delete file
        if os.path.exists(filepath):
            os.remove(filepath)
        
        # Remove from metadata
        del self.metadata[model_key]
        self._save_metadata()
        
        print(f"✓ Deleted model: {filename}")
        return True
    
    def clear_all_models(self, confirm=True):
        """
        Delete all trained models.
        
        Args:
            confirm: If True, ask for confirmation before deleting
        """
        if not self.metadata:
            print("No models to delete.")
            return
        
        if confirm:
            response = input(f"\nDelete all {len(self.metadata)} models? (yes/no): ")
            if response.lower() != 'yes':
                print("Cancelled.")
                return
        
        # Delete all files
        for info in self.metadata.values():
            filepath = os.path.join(self.models_dir, info['filename'])
            if os.path.exists(filepath):
                os.remove(filepath)
        
        # Clear metadata
        self.metadata = {}
        self._save_metadata()
        
        print("✓ All models deleted.")
    
    def get_model_info(self, games, alpha=0.5, epsilon=0.1):
        """
        Get detailed information about a specific model.
        
        Args:
            games: Number of training games
            alpha: Learning rate
            epsilon: Exploration rate
        
        Returns:
            dict: Model information or None if not found
        """
        model_key = self._get_model_key(games, alpha, epsilon)
        return self.metadata.get(model_key)


def smart_train(games, alpha=0.5, epsilon=0.1, force_retrain=False):
    """
    Smart training function that uses caching.
    
    Args:
        games: Number of training games
        alpha: Learning rate
        epsilon: Exploration rate
        force_retrain: If True, retrain even if model exists
    
    Returns:
        ConnectFourAI instance
    """
    manager = ModelManager()
    return manager.get_or_train(games, alpha, epsilon, force_retrain)


if __name__ == "__main__":
    # Example usage
    manager = ModelManager()
    
    print("Model Manager - Connect Four Q-Learning")
    print("="*80)
    
    # List existing models
    manager.list_models()
    
    print("\nExample: Getting or training a model...")
    ai = manager.get_or_train(games=5000, alpha=0.5, epsilon=0.1)
    print(f"\nReady to play with AI trained on 5000 games!")