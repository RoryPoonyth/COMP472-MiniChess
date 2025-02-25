# search.py
import math
import copy

# ----------------------------------------------------------------------
#                          HEURISTICS
# ----------------------------------------------------------------------

def evaluate_state_e0(game_state):
    """
    A simple material-only heuristic for 5x5 mini-chess.
    Piece values:
      Pawn (p) = 1
      Knight (N), Bishop (B) = 3
      Queen (Q) = 9
      King (K)  = 999  (very high to avoid losing the king)
    """
    board = game_state["board"]
    score = 0
    piece_values = {
        'p': 1,
        'N': 3,
        'B': 3,
        'Q': 9,
        'K': 999
    }
    for row in board:
        for piece in row:
            if piece == '.':
                continue
            color = piece[0]  # 'w' or 'b'
            ptype = piece[1]  # 'p', 'N', 'B', 'Q', 'K'
            value = piece_values.get(ptype, 0)
            if color == 'w':
                score += value
            else:
                score -= value
    return score


def evaluate_state_e1(game_state):
    """
    e1: Material + Pawn Advancement + Center Control

    1) Material (e0) as base.
    2) Small bonus for how far pawns have advanced:
       - White pawns => +0.2 * (4 - row)
       - Black pawns => -0.2 * row
    3) Small bonus for occupying center squares:
       let's define (1,1), (1,2), (2,1), (2,2) as "center".
       Each piece = +0.5 if white, -0.5 if black.
    """
    board = game_state["board"]
    base_score = evaluate_state_e0(game_state)

    pawn_adv_bonus = 0.0
    center_control_bonus = 0.0
    center_squares = {(1,1), (1,2), (2,1), (2,2)}

    for r, row in enumerate(board):
        for c, piece in enumerate(row):
            if piece == '.':
                continue

            color = piece[0]
            ptype = piece[1]

            # Pawn Advancement
            if ptype == 'p':
                if color == 'w':
                    pawn_adv_bonus += 0.2 * (4 - r)
                else:
                    pawn_adv_bonus -= 0.2 * r

            # Center Control
            if (r, c) in center_squares:
                if color == 'w':
                    center_control_bonus += 0.5
                else:
                    center_control_bonus -= 0.5

    return base_score + pawn_adv_bonus + center_control_bonus


def evaluate_state_e2(game_state):
    """
    e2: Builds on e1 by adding:
        - Mobility: (# White moves - # Black moves) * 0.1
        - King Safety: penalize if a king is on the board edge.

    For mobility, we do an approximate count of possible moves 
    (like a partial version of valid_moves). This is not perfect 
    but good enough for a small board.

    For king safety, if a king is on an edge => 
    White king on edge => -1 to the score
    Black king on edge => +1 to the score (also effectively -1 for black).
    """
    score = evaluate_state_e1(game_state)
    board = game_state["board"]

    # Mobility
    w_moves = approximate_moves(board, 'w')
    b_moves = approximate_moves(board, 'b')
    mobility_term = 0.1 * (w_moves - b_moves)
    score += mobility_term

    # King safety
    for r, row in enumerate(board):
        for c, piece in enumerate(row):
            if piece.endswith('K'):  # wK or bK
                is_edge = (r == 0 or r == 4 or c == 0 or c == 4)
                if is_edge:
                    if piece.startswith('w'):
                        score -= 1.0  # White king on edge => bad
                    else:
                        score += 1.0  # Black king on edge => good for White

    return score

def approximate_moves(board, color_char):
    """
    Roughly approximate how many moves a given side (w/b) might have.
    We'll reuse some logic from e0 to gather pieces 
    and do a partial "valid moves" check.
    """
    pieceColor = color_char  # 'w' or 'b'
    opponentColor = 'b' if pieceColor == 'w' else 'w'
    moves_count = 0

    for r, row in enumerate(board):
        for c, piece in enumerate(row):
            if piece.startswith(pieceColor):
                ptype = piece[1]
                if ptype == 'p':
                    # Forward move
                    front = r - 1 if pieceColor == 'w' else r + 1
                    if 0 <= front < 5 and board[front][c] == '.':
                        moves_count += 1
                    # Captures left/right
                    leftCol = c - 1
                    rightCol = c + 1
                    if 0 <= front < 5 and 0 <= leftCol < 5 and board[front][leftCol].startswith(opponentColor):
                        moves_count += 1
                    if 0 <= front < 5 and 0 <= rightCol < 5 and board[front][rightCol].startswith(opponentColor):
                        moves_count += 1

                elif ptype == 'N':
                    knight_steps = [(-2, -1), (-2, +1), (+2, -1), (+2, +1),
                                    (-1, -2), (-1, +2), (+1, -2), (+1, +2)]
                    for (dr, dc) in knight_steps:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < 5 and 0 <= nc < 5:
                            if board[nr][nc] == '.' or board[nr][nc].startswith(opponentColor):
                                moves_count += 1

                elif ptype == 'B':
                    directions = [(-1, -1), (-1, +1), (+1, -1), (+1, +1)]
                    for dr, dc in directions:
                        nr, nc = r, c
                        while True:
                            nr += dr
                            nc += dc
                            if not (0 <= nr < 5 and 0 <= nc < 5):
                                break
                            if board[nr][nc] == '.':
                                moves_count += 1
                            elif board[nr][nc].startswith(opponentColor):
                                moves_count += 1
                                break
                            else:
                                break

                elif ptype == 'Q':
                    directions = [
                        (-1, 0), (1, 0), (0, -1), (0, 1),
                        (-1, -1), (-1, +1), (1, -1), (1, +1)
                    ]
                    for dr, dc in directions:
                        nr, nc = r, c
                        while True:
                            nr += dr
                            nc += dc
                            if not (0 <= nr < 5 and 0 <= nc < 5):
                                break
                            if board[nr][nc] == '.':
                                moves_count += 1
                            elif board[nr][nc].startswith(opponentColor):
                                moves_count += 1
                                break
                            else:
                                break

                elif ptype == 'K':
                    directions = [
                        (-1, 0), (1, 0),
                        (0, -1), (0, 1),
                        (-1, -1), (-1, +1),
                        (1, -1), (1, +1)
                    ]
                    for dr, dc in directions:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < 5 and 0 <= nc < 5:
                            if board[nr][nc] == '.' or board[nr][nc].startswith(opponentColor):
                                moves_count += 1

    return moves_count


HEURISTICS = {
    'e0': evaluate_state_e0,
    'e1': evaluate_state_e1,
    'e2': evaluate_state_e2
}

# ----------------------------------------------------------------------
#                 MINIMAX / ALPHA-BETA SEARCH
# ----------------------------------------------------------------------

def minimax(game, game_state, depth, maximizing_player, heuristic_func):
    """
    Basic Minimax without alpha-beta. 
    Returns (best_score, best_move).
    """
    if depth == 0 or game.game_over(game_state):
        return heuristic_func(game_state), None

    valid_moves = game.valid_moves(game_state)
    if not valid_moves:
        return heuristic_func(game_state), None

    if maximizing_player:
        best_score = -math.inf
        best_move = None
        for move in valid_moves:
            next_state = copy_game_state(game_state)
            game.make_move(next_state, move)
            new_score, _ = minimax(game, next_state, depth - 1, False, heuristic_func)
            if new_score > best_score:
                best_score = new_score
                best_move = move
        return best_score, best_move
    else:
        best_score = math.inf
        best_move = None
        for move in valid_moves:
            next_state = copy_game_state(game_state)
            game.make_move(next_state, move)
            new_score, _ = minimax(game, next_state, depth - 1, True, heuristic_func)
            if new_score < best_score:
                best_score = new_score
                best_move = move
        return best_score, best_move


def alpha_beta(game, game_state, depth, alpha, beta, maximizing_player, heuristic_func):
    """
    Minimax with Alpha-Beta pruning.
    Returns (best_score, best_move).
    """
    if depth == 0 or game.game_over(game_state):
        return heuristic_func(game_state), None

    valid_moves = game.valid_moves(game_state)
    if not valid_moves:
        return heuristic_func(game_state), None

    if maximizing_player:
        best_score = -math.inf
        best_move = None
        for move in valid_moves:
            next_state = copy_game_state(game_state)
            game.make_move(next_state, move)
            new_score, _ = alpha_beta(game, next_state, depth - 1, alpha, beta, False, heuristic_func)
            if new_score > best_score:
                best_score = new_score
                best_move = move
            alpha = max(alpha, best_score)
            if alpha >= beta:
                break  # beta cutoff
        return best_score, best_move
    else:
        best_score = math.inf
        best_move = None
        for move in valid_moves:
            next_state = copy_game_state(game_state)
            game.make_move(next_state, move)
            new_score, _ = alpha_beta(game, next_state, depth - 1, alpha, beta, True, heuristic_func)
            if new_score < best_score:
                best_score = new_score
                best_move = move
            beta = min(beta, best_score)
            if beta <= alpha:
                break  # alpha cutoff
        return best_score, best_move


def choose_best_move(game, game_state, max_depth, use_alpha_beta, heuristic_name='e0'):
    """
    Wrapper that picks the best move using either minimax or alpha-beta.
    Returns (best_move, best_score).
    """
    heuristic_func = HEURISTICS.get(heuristic_name, evaluate_state_e0)
    maximizing_player = (game_state["turn"] == "white")

    state_copy = copy_game_state(game_state)

    if use_alpha_beta:
        best_score, best_move = alpha_beta(
            game, state_copy, max_depth,
            alpha=-math.inf, beta=math.inf,
            maximizing_player=maximizing_player,
            heuristic_func=heuristic_func
        )
    else:
        best_score, best_move = minimax(
            game, state_copy, max_depth,
            maximizing_player=maximizing_player,
            heuristic_func=heuristic_func
        )
    return best_move, best_score


def copy_game_state(game_state):
    """Safely copy the board and turn."""
    new_state = {
        "board": [row[:] for row in game_state["board"]],
        "turn": game_state["turn"]
    }
    return new_state
