"""
Connect Four Q-Learning - Smart Training with Model Caching
Automatically saves and loads trained models
"""

from connectfour import play_graphical, play_console
from model_manager import ModelManager, smart_train


def main():
    """Main function with smart model management."""
    
    print("="*80)
    print("Connect Four - Q-Learning AI with Smart Caching")
    print("="*80)
    
    # Initialize model manager
    manager = ModelManager()
    
    # Show available models
    print("\nðŸ“¦ Checking for existing trained models...")
    manager.list_models()
    
    # Ask user for training preference
    print("\n" + "="*80)
    print("Training Options:")
    print("="*80)
    print("1. Quick (1,000 games) - Beginner level")
    print("2. Standard (10,000 games) - Good opponent â­ Recommended")
    print("3. Strong (50,000 games) - Challenging")
    print("4. Expert (100,000 games) - Very strong")
    print("5. Master (500,000 games) - Elite level")
    print("6. Ultra (1,000,000 games) - Maximum strength")
    print("7. Custom - Specify your own number of games")
    print("8. Use existing model (if available)")
    
    choice = input("\nEnter your choice (1-8): ").strip()
    
    # Map choices to game counts
    game_counts = {
        '1': 1000,
        '2': 10000,
        '3': 50000,
        '4': 100000,
        '5': 500000,
        '6': 1000000
    }
    
    if choice in game_counts:
        games = game_counts[choice]
        print(f"\nSelected: {games:,} training games")
        
        # Check if we should retrain
        if manager.model_exists(games):
            retrain = input("Model already exists. Retrain? (y/n): ").strip().lower()
            force_retrain = retrain == 'y'
        else:
            force_retrain = False
        
        # Get or train the model (smart caching!)
        ai = smart_train(games, alpha=0.5, epsilon=0.1, force_retrain=force_retrain)
    
    elif choice == '7':
        # Custom training
        games = int(input("Enter number of training games: "))
        alpha = float(input("Learning rate (default 0.5): ") or 0.5)
        epsilon = float(input("Exploration rate (default 0.1): ") or 0.1)
        
        if manager.model_exists(games, alpha, epsilon):
            retrain = input("Model already exists. Retrain? (y/n): ").strip().lower()
            force_retrain = retrain == 'y'
        else:
            force_retrain = False
        
        ai = smart_train(games, alpha, epsilon, force_retrain)
    
    elif choice == '8':
        # Use existing model
        if not manager.metadata:
            print("\nNo trained models available. Please train a new model first.")
            return
        
        print("\nAvailable models:")
        models_list = list(manager.metadata.items())
        for idx, (key, info) in enumerate(models_list, 1):
            print(f"{idx}. {info['games']:,} games (alpha={info['alpha']}, epsilon={info['epsilon']})")
        
        model_choice = int(input("\nSelect model number: ")) - 1
        if 0 <= model_choice < len(models_list):
            key = models_list[model_choice][0]
            info = manager.metadata[key]
            ai = manager.load_model(info['games'], info['alpha'], info['epsilon'])
            if ai is None:
                print("Failed to load model.")
                return
        else:
            print("Invalid choice.")
            return
    
    else:
        print("Invalid choice. Defaulting to Standard (10,000 games).")
        ai = smart_train(10000)
    
    # Choose game mode
    print("\n" + "="*80)
    print("Game Mode Selection")
    print("="*80)
    print("1. Graphical (Pygame) - Visual interface")
    print("2. Console - Text-based")
    
    mode_choice = input("\nEnter your choice (1-2): ").strip()
    
    print("\nðŸŽ® Starting game...\n")
    
    if mode_choice == "2":
        play_console(ai)
    else:
        play_graphical(ai)
    
    print("\n" + "="*80)
    print("Thanks for playing!")
    print("="*80)
    print("\nðŸ’¡ Tip: Your trained models are saved and can be reused!")
    print("   Run this script again to instantly load your trained AI.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nGame interrupted. Goodbye!")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        import traceback
        traceback.print_exc()