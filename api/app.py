from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
from pathlib import Path
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from rl.mcts_ai import mcts_move

app = Flask(__name__)
CORS(app)

BOARD_SIZE = 9
MCTS_SIMULATIONS = 500
MCTS_MAX_TIME = 3.0

print("Using pure MCTS AI")
print(f"  Simulations: {MCTS_SIMULATIONS}")
print(f"  Max time: {MCTS_MAX_TIME}s")

def check_winner(board, row, col, player):
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
    for dr, dc in directions:
        count = 1
        r, c = row + dr, col + dc
        while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and board[r, c] == player:
            count += 1
            r += dr
            c += dc
        r, c = row - dr, col - dc
        while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and board[r, c] == player:
            count += 1
            r -= dr
            c -= dc
        if count >= 5:
            return True
    return False

@app.route('/api/move', methods=['POST'])
def get_ai_move():
    try:
        data = request.json
        board = np.array(data['board'], dtype=np.int8)
        player = int(data['player'])
        
        # Use pure MCTS
        action = mcts_move(
            board.copy(), 
            player, 
            simulations=MCTS_SIMULATIONS, 
            max_time=MCTS_MAX_TIME
        )
        
        row = int(action // 9)
        col = int(action % 9)
        
        if board[row, col] != 0:
            return jsonify({'error': f'Invalid move: position ({row}, {col}) is already occupied'}), 400
        
        board[row, col] = player
        
        winner = None
        game_over = False
        
        if check_winner(board, row, col, player):
            winner = player
            game_over = True
        elif np.all(board != 0):
            winner = 0
            game_over = True
        
        board_list = board.tolist()
        
        return jsonify({
            'row': row,
            'col': col,
            'board': board_list,
            'winner': winner,
            'gameOver': game_over,
            'aiType': 'MCTS'
        })
    except Exception as e:
        import traceback
        error_msg = str(e)
        traceback.print_exc()
        return jsonify({'error': error_msg}), 500

@app.route('/api/check', methods=['POST'])
def check_game_state():
    try:
        data = request.json
        board = np.array(data['board'], dtype=np.int8)
        row = int(data['row'])
        col = int(data['col'])
        player = int(data['player'])
        
        winner = None
        game_over = False
        
        if check_winner(board, row, col, player):
            winner = player
            game_over = True
        elif np.all(board != 0):
            winner = 0
            game_over = True
        
        return jsonify({
            'winner': winner,
            'gameOver': game_over
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'aiType': 'MCTS',
        'simulations': MCTS_SIMULATIONS,
        'maxTime': MCTS_MAX_TIME
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
