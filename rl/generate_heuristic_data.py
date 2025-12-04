"""
Generate training data by having heuristic AI play against itself.

This script creates expert demonstrations that can be used to train
the AlphaZero model through imitation learning.
"""

import sys
from pathlib import Path
import pickle
import numpy as np
from tqdm import tqdm

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from env.gomoku_env import GomokuEnv9x9
from rl.alphazero_train import simple_heuristic_move


def get_state_from_board(board, current_player):
    """
    Convert board state to AlphaZero format (3, 9, 9).
    
    Args:
        board: 9x9 numpy array (0=empty, 1=player1, 2=player2)
        current_player: Current player (1 or 2)
    
    Returns:
        3-channel observation array (3, 9, 9)
    """
    obs = np.zeros((3, 9, 9), dtype=np.float32)
    
    # Channel 0: Current player's stones
    obs[0] = (board == current_player).astype(np.float32)
    
    # Channel 1: Opponent's stones
    opponent = 3 - current_player
    obs[1] = (board == opponent).astype(np.float32)
    
    # Channel 2: Current player indicator (1 if player 1, 0 if player 2)
    if current_player == 1:
        obs[2] = np.ones((9, 9), dtype=np.float32)
    
    return obs


def create_policy_from_action(action, board_size=9):
    """
    Create a policy vector (one-hot) for the given action.
    
    Args:
        action: Action index (0-80)
        board_size: Size of the board (default 9)
    
    Returns:
        Policy vector of length 81 (one-hot)
    """
    policy = np.zeros(board_size * board_size, dtype=np.float32)
    policy[action] = 1.0
    return policy


def play_heuristic_game():
    """
    Play one game between two heuristic AIs.
    
    Returns:
        List of (state, policy, value) tuples for training
    """
    env = GomokuEnv9x9()
    obs, _ = env.reset()
    
    game_history = []  # Store (state, policy, player) tuples
    board = env.board.copy()
    current_player = env.current_player
    
    move_count = 0
    
    while not env.done:
        # Get heuristic move
        action = simple_heuristic_move(board.copy(), current_player)
        
        # Convert to state format
        state = get_state_from_board(board, current_player)
        
        # Create policy (one-hot for the chosen action)
        policy = create_policy_from_action(action)
        
        # Store state, policy, and which player made this move
        game_history.append((state, policy, current_player))
        
        # Make the move
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Check if game ended
        if terminated or truncated:
            winner = info.get('winner', 0)
            
            # Now assign values to all states based on the final outcome
            # Value is from the perspective of the player who made each move
            for i, (state, policy, player) in enumerate(game_history):
                if winner == 0:
                    value = 0.0  # Draw
                elif winner == player:
                    value = 1.0  # This player won
                else:
                    value = -1.0  # This player lost
                
                game_history[i] = (state, policy, value)
            
            break
        
        # Update board and player
        board = env.board.copy()
        current_player = env.current_player
        move_count += 1
        
        # Safety check
        if move_count > 81:
            print("Warning: Game exceeded 81 moves, breaking")
            # Assign draw values to remaining states
            for i in range(len(game_history)):
                state, policy, _ = game_history[i]
                game_history[i] = (state, policy, 0.0)
            break
    
    return game_history


def generate_heuristic_data(num_games=1000, output_file="heuristic_data.pkl"):
    """
    Generate training data from heuristic self-play.
    
    Args:
        num_games: Number of games to play
        output_file: Output file path for the data
    """
    print(f"Generating {num_games} games of heuristic self-play...")
    
    all_data = []
    
    for game_idx in tqdm(range(num_games), desc="Playing games"):
        game_data = play_heuristic_game()
        all_data.extend(game_data)
        
        if (game_idx + 1) % 100 == 0:
            print(f"  Generated {len(all_data)} training samples from {game_idx + 1} games")
    
    # Save the data
    output_path = project_root / "models" / output_file
    output_path.parent.mkdir(exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(all_data, f)
    
    print(f"\nSaved {len(all_data)} training samples to {output_path}")
    print(f"Average game length: {len(all_data) / num_games:.1f} moves")
    
    # Print some statistics
    values = [d[2] for d in all_data]
    print(f"Value distribution:")
    print(f"  Wins (1.0): {sum(1 for v in values if v == 1.0)}")
    print(f"  Losses (-1.0): {sum(1 for v in values if v == -1.0)}")
    print(f"  Draws (0.0): {sum(1 for v in values if v == 0.0)}")
    
    return all_data


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate heuristic self-play data")
    parser.add_argument("--games", type=int, default=1000, help="Number of games to play")
    parser.add_argument("--output", type=str, default="heuristic_data.pkl", help="Output file name")
    
    args = parser.parse_args()
    
    generate_heuristic_data(num_games=args.games, output_file=args.output)

