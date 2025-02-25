import math
import copy

# ----------------------------------------------------------------------
#                          HEURISTICS
# ----------------------------------------------------------------------

def evaluate_state_e0(game_state):
    """
    e0: Simple material + potential captures.
    Piece values:
      Pawn (p)   = 1
      Knight (N) = 3
      Bishop (B) = 3
      Queen (Q)  = 9
      King (K)   = 999
    We add a small bonus for pieces that can be captured in one move,
    weighting the captured piece's value by 0.2, for instance.
    """
    board = game_state["board"]
    score = 0

    # Basic material
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

    # Potential captures
    white_capture_value = approximate_captures(board, 'w', piece_values)
    black_capture_value = approximate_captures(board, 'b', piece_values)
    # Add a fraction of the capturable value to the score
    # e.g. 0.2 factor
    capture_factor = 0.2
    capture_term = capture_factor * (white_capture_value - black_capture_value)
    score += capture_term

    return score

def evaluate_state_e1(game_state):
    """
    e1: Builds on e0 by also adding:
       - Pawn advancement
       - Center control
       - + potential captures
    """
    board = game_state["board"]

    # Start with the base e0 (material + capture potential)
    base_score = evaluate_state_e0(game_state)

    # Pawn Advancement + Center Control
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
        - Mobility: (#White moves - #Black moves) * 0.1
        - King safety: penalize king on board edge
        - + potential captures (inherited from e0)

    So this has everything:
      Material + captures + pawn advancement + center + mobility + king safety
    """
    board = game_state["board"]

    # Start with e1 logic (which already includes e0 inside it)
    score = evaluate_state_e1(game_state)

    # 1) Mobility
    w_moves = approximate_moves(board, 'w')
    b_moves = approximate_moves(board, 'b')
    mobility_term = 0.1 * (w_moves - b_moves)
    score += mobility_term

    # 2) King safety
    for r, row in enumerate(board):
        for c, piece in enumerate(row):
            if piece.endswith('K'):  # wK or bK
                is_edge = (r == 0 or r == 4 or c == 0 or c == 4)
                if is_edge:
                    if piece.startswith('w'):
                        score -= 1.0
                    else:
                        score += 1.0

    return score

def approximate_captures(board, color_char, piece_values):
    """
    Returns the total *value* of opponent pieces that 'color_char' can capture
    in one move. The higher this is, the better for that color, since they 
    can potentially win material. We'll reuse logic similar to approximate_moves,
    but specifically to see if the square targeted is an opponent piece.

    piece_values: a dict mapping {'p':1, 'N':3,...}
    """
    opponent = 'w' if color_char == 'b' else 'b'
    total_capture_value = 0

    for r, row in enumerate(board):
        for c, piece in enumerate(row):
            if piece.startswith(color_char):
                ptype = piece[1]
                # We'll check each potential move for a capture
                possible_targets = get_possible_moves_for_piece(board, r, c, piece, opponent)
                # If the target is an opponent piece, add its piece_values
                for (tr, tc) in possible_targets:
                    target_piece = board[tr][tc]
                    if target_piece.startswith(opponent):
                        ttype = target_piece[1]
                        val = piece_values.get(ttype, 0)
                        total_capture_value += val

    return total_capture_value

def get_possible_moves_for_piece(board, r, c, piece, opponent):
    """
    Return a list of board coordinates (row,col) 
    that this piece can move to in one step, ignoring checks, etc. 
    Only used for the purpose of evaluating potential captures.
    """
    ptype = piece[1]
    color = piece[0]
    moves = []

    if ptype == 'p':
        front = r - 1 if color == 'w' else r + 1
        leftCol = c - 1
        rightCol = c + 1
        # Pawn captures are diagonals
        if 0 <= front < 5 and 0 <= leftCol < 5:
            moves.append((front, leftCol))
        if 0 <= front < 5 and 0 <= rightCol < 5:
            moves.append((front, rightCol))

    elif ptype == 'N':
        knight_steps = [(-2, -1), (-2, +1), (+2, -1), (+2, +1),
                        (-1, -2), (-1, +2), (+1, -2), (+1, +2)]
        for (dr, dc) in knight_steps:
            nr, nc = r + dr, c + dc
            if 0 <= nr < 5 and 0 <= nc < 5:
                moves.append((nr, nc))

    elif ptype == 'B':
        directions = [(-1, -1), (-1, +1), (+1, -1), (+1, +1)]
        for dr, dc in directions:
            nr, nc = r, c
            while True:
                nr += dr
                nc += dc
                if not (0 <= nr < 5 and 0 <= nc < 5):
                    break
                moves.append((nr, nc))
                # If we hit any piece (opponent or ours), we stop
                if board[nr][nc] != '.':
                    break

    elif ptype == 'Q':
        directions = [
            (-1,  0), (1,  0),
            (0, -1),  (0,  1),
            (-1, -1), (-1,  1),
            (1, -1),  (1,  1)
        ]
        for dr, dc in directions:
            nr, nc = r, c
            while True:
                nr += dr
                nc += dc
                if not (0 <= nr < 5 and 0 <= nc < 5):
                    break
                moves.append((nr, nc))
                # Stop if we hit any piece
                if board[nr][nc] != '.':
                    break

    elif ptype == 'K':
        directions = [
            (-1, 0), (1, 0),
            (0, -1), (0, 1),
            (-1, -1), (-1, +1),
            (1, -1), (1, +1)
        ]
        for dr, dc in directions:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < 5 and 0 <= nc < 5:
                moves.append((nr, nc))

    return moves

def approximate_moves(board, color_char):
    """
    Same as before: approximate mobility by counting 
    how many squares 'color_char' pieces can move to 
    (including captures on opponent squares).
    """
    pieceColor = color_char  
    opponentColor = 'b' if pieceColor == 'w' else 'w'
    moves_count = 0

    for r, row in enumerate(board):
        for c, piece in enumerate(row):
            if piece.startswith(pieceColor):
                ptype = piece[1]
                # We'll do a partial simulation of valid moves 
                # (including potential captures).
                if ptype == 'p':
                    front = r - 1 if pieceColor == 'w' else r + 1
                    if 0 <= front < 5 and board[front][c] == '.':
                        moves_count += 1
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
                        (-1,  0), (1,  0),
                        (0, -1),  (0,  1),
                        (-1, -1), (-1,  1),
                        (1, -1),  (1,  1)
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
                        (1, -1),  (1, +1)
                    ]
                    for dr, dc in directions:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < 5 and 0 <= nc < 5:
                            if board[nr][nc] == '.' or board[nr][nc].startswith(opponentColor):
                                moves_count += 1

    return moves_count


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
    # Select the correct heuristic function by name
    from collections import defaultdict
    from functools import partial

    from sys import exit
    # For safety, fallback to e0 if invalid
    from math import inf

    # Our dictionary is defined at the bottom, but we've already declared it above:
    # It's possible we do it like this for clarity:
    #   heuristic_func = HEURISTICS.get(heuristic_name, evaluate_state_e0)
    heuristic_func = HEURISTICS.get(heuristic_name, evaluate_state_e0)

    maximizing_player = (game_state["turn"] == "white")

    # Copy the state so we don't mutate the actual game_state
    copied = copy_game_state(game_state)

    if use_alpha_beta:
        best_score, best_move = alpha_beta(
            game,
            copied,
            depth=max_depth,
            alpha=-math.inf,
            beta= math.inf,
            maximizing_player=maximizing_player,
            heuristic_func=heuristic_func
        )
    else:
        best_score, best_move = minimax(
            game,
            copied,
            depth=max_depth,
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


# We keep a reference here, so choose_best_move can get it easily
HEURISTICS = {
    'e0': evaluate_state_e0,
    'e1': evaluate_state_e1,
    'e2': evaluate_state_e2
}
