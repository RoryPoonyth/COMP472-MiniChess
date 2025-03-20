import math
import copy
import time
import sys
from datetime import datetime

# NEW: import our search logic (with e0, e1, e2 heuristics)
from search import choose_best_move

class MiniChess:
    def __init__(self, max_time_white, max_time_black, max_turns, use_alpha_beta_white, use_alpha_beta_black, 
                 heuristic_name_white, heuristic_name_black, mode, human_color ):
        """
        If mode=1 => Human vs Human
        If mode=2 => Human vs AI (human_color='white' or 'black')
        If mode=3 => AI vs AI
        """
        self.max_time_white = max_time_white
        self.max_time_black = max_time_black
        self.max_turns = max_turns
        self.use_alpha_beta_white = use_alpha_beta_white
        self.use_alpha_beta_black = use_alpha_beta_black
        self.heuristic_name_white = heuristic_name_white
        self.heuristic_name_black = heuristic_name_black
        self.mode = mode
        self.human_color = human_color
        self.turn_number = 1
        self.current_game_state = self.init_board()
        self.white_player = 'human' if (mode == 1 or (mode == 2 and human_color == 'white')) else 'ai'
        self.black_player = 'human' if (mode == 1 or (mode == 2 and human_color == 'black')) else 'ai'
        
        # One "turn" = White moves, then Black moves
        self.turn_number = 1

         # Build a more descriptive log filename
        now_str = datetime.now().strftime("%Y%m%d-%H%M%S")  # e.g. "20250307-143025"
        mode_str = f"mode{mode}"
        white_ab = f"wAB{str(self.use_alpha_beta_white).lower()}"
        black_ab = f"bAB{str(self.use_alpha_beta_black).lower()}"

        # Heuristics might be None if it's a human for that side, so guard with a default
        w_heur = self.heuristic_name_white if self.heuristic_name_white else "none"
        b_heur = self.heuristic_name_black if self.heuristic_name_black else "none"

        # Putting it all together
        self.log_filename = (
            f"gameTrace_{now_str}_"
            f"{mode_str}_"
            f"wTime{self.max_time_white}_"
            f"bTime{self.max_time_black}_"
            f"{white_ab}_"
            f"{black_ab}_"
            f"wH-{w_heur}_"
            f"bH-{b_heur}_"
            f"maxT{self.max_turns}.txt"
        )

        self.log_file = open(self.log_filename, "w", encoding="utf-8")

        # Placeholder stats for AI
        self.cumulative_states_explored = 0
        self.states_by_depth = {}  # e.g. {0:count, 1:count,...}

        # Write initial log info
        self.log_initial_parameters()
        self.log_initial_board()

        # --------------------------------------
        #  Track "no-capture" / "no change" rule
        # --------------------------------------
        # We'll track how many consecutive full turns had the same piece count.
        self.stall_counter = 0  
        self.last_piece_count = self.count_pieces(self.current_game_state)

    def init_board(self):
        """
        5x5 mini-chess board
        White pawns typically on row=3, black pawns on row=1, etc.
        """
        state = {
            "board": [
                ['bK', 'bQ', 'bB', 'bN', '.'],
                ['.', '.', 'bp', 'bp', '.'],
                ['.', '.', '.', '.', '.'],
                ['.', 'wp', 'wp', '.', '.'],
                ['.', 'wN', 'wB', 'wQ', 'wK']
            ],
            "turn": 'white',  # White starts
        }
        return state

    # ----------------------------------------------------------------------
    #                           LOGGING
    # ----------------------------------------------------------------------

    def log(self, text=""):
        """Helper for writing a line to the log file."""
        self.log_file.write(text + "\n")

    def log_initial_parameters(self):
        """
        1) Game parameters:
           a) time limit
           b) max number of turns
           c) play mode (who is AI/human)
           d) alpha-beta on/off
           e) heuristic name
        """
        def player_desc(player):
            return "AI" if player == 'ai' else "H"

        mode_str = f"Player1={player_desc(self.white_player)} & Player2={player_desc(self.black_player)}"
        self.log("----- Game Parameters -----")
        self.log(f"(a) Time limit (t): \nWhite Max Time:{self.max_time_white} seconds \Black Max Time:{self.max_time_black} seconds")
        self.log(f"(b) Max number of turns (n) = {self.max_turns}")
        self.log(f"(c) Play mode => {mode_str}")

        if (self.mode == 2 and self.human_color == 'white'):
            ab_str = "ON" if self.use_alpha_beta_black else "OFF"
            self.log(f"(d) Alpha-beta is {ab_str}")
            self.log(f"(e) Heuristic name = {self.heuristic_name_black}")
        elif (self.mode == 2 and self.human_color == 'black'):
            ab_str = "ON" if self.use_alpha_beta_white else "OFF"
            self.log(f"(d) Alpha-beta is {ab_str}")
            self.log(f"(e) Heuristic name = {self.heuristic_name_white}")
            
        elif (self.mode == 3):
            ab_str = "ON" if self.use_alpha_beta_white else "OFF"
            self.log(f"(d) White Alpha-beta is {ab_str}")
            self.log(f"(e)White Heuristic name = {self.heuristic_name_white}")
            ab_str = "ON" if self.use_alpha_beta_black else "OFF"
            self.log(f"(d) Black Alpha-beta is {ab_str}")
            self.log(f"(e)Black Heuristic name = {self.heuristic_name_black}")

        elif (self.mode == 1):
            # No AI => alpha-beta not applicable
            self.log("(d) No AI in this game => alpha-beta N/A")
        self.log()


    def log_initial_board(self):
        """
        2) The initial board config
        """
        self.log("----- Initial Board Configuration -----")
        self.log_board(self.current_game_state["board"])
        self.log()

    def log_board(self, board):
        """Write board to the log file in a 5x5 layout."""
        for i, row in enumerate(board, start=1):
            row_label = str(6 - i)
            row_str = ' '.join(piece.rjust(2, ' ') for piece in row)
            self.log(f"{row_label}  {row_str}")
        self.log("   A  B  C  D  E")

    def notation_from_indices(self, start_rc, end_rc):
        """
        Convert ((r1,c1),(r2,c2)) to e.g. "C3 C4"
        row0 => '5', row4 => '1'
        col0 => 'A', col4 => 'E'
        """
        (sr, sc) = start_rc
        (er, ec) = end_rc
        s_row_label = str(5 - sr)
        e_row_label = str(5 - er)
        s_col_label = chr(ord('A') + sc)
        e_col_label = chr(ord('A') + ec)
        return f"{s_col_label}{s_row_label} {e_col_label}{e_row_label}"

    def log_move_action(
        self,
        player_name, 
        turn_number,
        move_str,
        is_ai=False,
        move_time=0.0,
        heuristic_score=None,
        search_score=None,
        new_board=None
    ):
        """
        3.1) 
         a) player
         b) turn number
         c) action (move)
         d) AI time
         e) AI heuristic score
         f) AI minimax/Alpha-beta search score
         g) new board config

        3.2) If AI, cumulative info: states explored, etc.
        """
        self.log("----- Action Info -----")
        self.log(f"(a) Player: {player_name}")
        self.log(f"(b) Turn number: #{turn_number}")
        self.log(f"(c) Action: move {move_str}")

        if is_ai:
            self.log(f"(d) Time for action: {move_time:.2f} s")
            self.log(f"(e) Heuristic score: {heuristic_score}")
            self.log(f"(f) Minimax/Alpha-beta result: {search_score}")

        self.log("(g) New board configuration:")
        self.log_board(new_board)

        if is_ai:
            #calculate average branching factor
            total_nodes = sum(self.states_by_depth.values())
            total_children = sum( d * cnt for d, cnt in self.states_by_depth.items())
            average_branching_factor = total_children/total_nodes if total_nodes >0 else 0
            self.log("    [Cumulative AI Info]")
            self.log(f"    (a) States explored so far: {self.cumulative_states_explored:,}")
            depth_details = ", ".join([f"{d}:{cnt}" for d, cnt in sorted(self.states_by_depth.items())])
            self.log(f"    (b) States by depth: {depth_details}")
            total_states = max(self.cumulative_states_explored, 1)
            percentages = ", ".join(
                [f"{d}:{(cnt/total_states)*100:.1f}%" for d, cnt in sorted(self.states_by_depth.items())]
            )
            self.log(f"    (c) % States by depth: {percentages}")
            self.log(f"    (d) Average branching factor: {average_branching_factor:.2f}")
        self.log()

    def log_winner(self, message):
        """
        4) The result (e.g. White won in 8 turns)
        """
        self.log("----- Game Over -----")
        self.log(f"Result: {message}")

    def close_log(self):
        """Close the log file once the game ends."""
        if not self.log_file.closed:
            self.log_file.close()

    def log_invalid_move(self, player, move_input, reason="Unknown"):
        """
        Log a short record of an invalid move attempt.
        """
        self.log("----- Invalid Move Attempt -----")
        self.log(f"Player: {player}")
        self.log(f"Attempted Input: '{move_input}'")
        self.log(f"Reason: {reason}")
        self.log()

    # ----------------------------------------------------------------------
    #                          CHESS LOGIC
    # ----------------------------------------------------------------------

    def display_board(self, game_state):
        """
        Print the board to console only.
        """
        print()
        for i, row in enumerate(game_state["board"], start=1):
            print(str(6 - i) + "  " + ' '.join(piece.rjust(3) for piece in row))
        print()
        print("     A   B   C   D   E")
        print()

    def parse_input(self, move_str):
        """
        Convert a user-entered string like "B3 B5" into ((row1,col1),(row2,col2)).
        Returns None if parsing fails.
        """
        try:
            start, end = move_str.split()
            start_col = ord(start[0].upper()) - ord('A')
            start_row = 5 - int(start[1])
            end_col   = ord(end[0].upper()) - ord('A')
            end_row   = 5 - int(end[1])
            return ((start_row, start_col), (end_row, end_col))
        except:
            return None

    def valid_moves(self, game_state):
        pieceColor = 'w' if game_state["turn"] == "white" else 'b'
        opponentColor = 'b' if pieceColor == 'w' else 'w'
        pieces = []
        for r, row in enumerate(game_state["board"]):
            for c, piece in enumerate(row):
                if piece.startswith(pieceColor):
                    pieces.append((piece, r, c))

        valid_moves = []
        for piece, row, col in pieces:
            piece_type = piece[1]
            frontRow = row - 1 if pieceColor == 'w' else row + 1
            leftCol  = col - 1
            rightCol = col + 1

            if piece_type == 'p':
                # Pawn forward
                if 0 <= frontRow < 5 and game_state["board"][frontRow][col] == '.':
                    valid_moves.append(((row, col), (frontRow, col)))
                # Pawn captures
                if 0 <= frontRow < 5 and 0 <= leftCol < 5:
                    if game_state["board"][frontRow][leftCol].startswith(opponentColor):
                        valid_moves.append(((row, col), (frontRow, leftCol)))
                if 0 <= frontRow < 5 and 0 <= rightCol < 5:
                    if game_state["board"][frontRow][rightCol].startswith(opponentColor):
                        valid_moves.append(((row, col), (frontRow, rightCol)))

            elif piece_type == 'N':
                knight_moves = [
                    (row - 2, col - 1), (row - 2, col + 1),
                    (row + 2, col - 1), (row + 2, col + 1),
                    (row - 1, col - 2), (row - 1, col + 2),
                    (row + 1, col - 2), (row + 1, col + 2),
                ]
                for (r2, c2) in knight_moves:
                    if 0 <= r2 < 5 and 0 <= c2 < 5:
                        target = game_state["board"][r2][c2]
                        if target == '.' or target.startswith(opponentColor):
                            valid_moves.append(((row, col), (r2, c2)))

            elif piece_type == 'B':
                directions = [(-1, -1), (-1,  1), (1, -1), (1,  1)]
                for dr, dc in directions:
                    r2, c2 = row, col
                    while True:
                        r2 += dr
                        c2 += dc
                        if not (0 <= r2 < 5 and 0 <= c2 < 5):
                            break
                        target = game_state["board"][r2][c2]
                        if target == '.':
                            valid_moves.append(((row, col), (r2, c2)))
                        elif target.startswith(opponentColor):
                            valid_moves.append(((row, col), (r2, c2)))
                            break
                        else:
                            break

            elif piece_type == 'Q':
                directions = [
                    (-1,  0), (1,  0),
                    (0, -1),  (0,  1),
                    (-1, -1), (-1,  1),
                    (1, -1),  (1,  1)
                ]
                for dr, dc in directions:
                    r2, c2 = row, col
                    while True:
                        r2 += dr
                        c2 += dc
                        if not (0 <= r2 < 5 and 0 <= c2 < 5):
                            break
                        target = game_state["board"][r2][c2]
                        if target == '.':
                            valid_moves.append(((row, col), (r2, c2)))
                        elif target.startswith(opponentColor):
                            valid_moves.append(((row, col), (r2, c2)))
                            break
                        else:
                            break

            elif piece_type == 'K':
                directions = [
                    (-1, 0), (1, 0),
                    (0, -1), (0, 1),
                    (-1, -1), (-1,  1),
                    (1, -1),  (1,  1)
                ]
                for dr, dc in directions:
                    r2 = row + dr
                    c2 = col + dc
                    if 0 <= r2 < 5 and 0 <= c2 < 5:
                        target = game_state["board"][r2][c2]
                        if target == '.' or target.startswith(opponentColor):
                            valid_moves.append(((row, col), (r2, c2)))

        return valid_moves

    def is_valid_move(self, game_state, move):
        return move in self.valid_moves(game_state)

    def make_move(self, game_state, move):
        (start_row, start_col), (end_row, end_col) = move
        piece = game_state["board"][start_row][start_col]
        game_state["board"][start_row][start_col] = '.'
        game_state["board"][end_row][end_col]     = piece
        
        # Pawn promotion
        if piece[1] == 'p':
            if piece.startswith('w') and end_row == 0:
                game_state["board"][end_row][end_col] = 'wQ'
            elif piece.startswith('b') and end_row == 4:
                game_state["board"][end_row][end_col] = 'bQ'

        # Switch turn
        game_state["turn"] = 'black' if game_state["turn"] == 'white' else 'white'
        return game_state

    def game_over(self, game_state):
        """
        Return True if the game has ended, False otherwise.
        No printing here—just the logic.
        """
        kings = [
            piece for row in game_state["board"] for piece in row
            if piece in ['wK', 'bK']
        ]
        if len(kings) == 1:
            # White or black is missing a king => game is over
            return True

        if self.turn_number >= self.max_turns:
            # Reached the maximum turn limit => draw
            return True

        return False

    def count_pieces(self, game_state):
        """
        Utility to count how many pieces remain on the board.
        """
        return sum(
            1
            for row in game_state['board']
            for piece in row
            if piece != '.'
        )

    def human_move(self):
        """
        Prompt the human player for a move, parse/validate it, and return if valid.
        """
        while True:
            move_input = input(f"{self.current_game_state['turn'].capitalize()} to move (e.g. A# B#), or 'exit': ")
            if move_input.lower() == 'exit':
                print("Game exited by user.")
                self.log_winner("Game exited by user.")
                self.close_log()
                sys.exit(0)
            
            parsed = self.parse_input(move_input)
            if not parsed:
                # Log invalid parse
                self.log_invalid_move(
                    player=self.current_game_state["turn"], 
                    move_input=move_input, 
                    reason="Parsing error"
                )
                print("Invalid move format. Try again.")
                continue
            
            if not self.is_valid_move(self.current_game_state, parsed):
                # Log invalid move
                self.log_invalid_move(
                    player=self.current_game_state["turn"], 
                    move_input=move_input, 
                    reason="Not in valid move list"
                )
                print("Invalid move. Try again.")
                continue
            
            return parsed

    def run_game_loop(self):
        while True:
            print(f"\n===== TURN {self.turn_number} =====")

            # WHITE move
            self.display_board(self.current_game_state)
            if self.game_over(self.current_game_state):
                self.log_end_and_close()
                break

            white_is_ai = (self.white_player == 'ai')
            if white_is_ai:
                print("White AI is thinking...")
                chosen_move, move_time, heur, search_s = self.ai_flow()  # Call ai_flow for AI moves
            else:
                chosen_move = self.human_move()  # Call human_move for human moves
                move_time   = 0.0
                heur        = None
                search_s    = None

            self.make_move(self.current_game_state, chosen_move)
            action_str = self.notation_from_indices(*chosen_move)
            self.log_move_action(
                player_name="white",
                turn_number=self.turn_number,
                move_str=f"from {action_str}",
                is_ai=white_is_ai,
                move_time=move_time,
                heuristic_score=heur,
                search_score=search_s,
                new_board=self.current_game_state["board"]
            )

            if self.game_over(self.current_game_state):
                self.log_end_and_close()
                break

            # BLACK move
            self.display_board(self.current_game_state)
            if self.game_over(self.current_game_state):
                self.log_end_and_close()
                break

            black_is_ai = (self.black_player == 'ai')
            if black_is_ai:
                print("Black AI is thinking...")
                chosen_move, move_time, heur, search_s = self.ai_flow()  # Call ai_flow for AI moves
            else:
                chosen_move = self.human_move()  # Call human_move for human moves
                move_time   = 0.0
                heur        = None
                search_s    = None

            self.make_move(self.current_game_state, chosen_move)
            action_str = self.notation_from_indices(*chosen_move)
            self.log_move_action(
                player_name="black",
                turn_number=self.turn_number,
                move_str=f"from {action_str}",
                is_ai=black_is_ai,
                move_time=move_time,
                heuristic_score=heur,
                search_score=search_s,
                new_board=self.current_game_state["board"]
            )

            # Check if the game is over by normal means
            if self.game_over(self.current_game_state):
                self.display_board(self.current_game_state)
                self.log_end_and_close()
                break

            # -------------------------------------------
            #    Check if piece count has stalled 10 turns
            # -------------------------------------------
            current_piece_count = self.count_pieces(self.current_game_state)
            if current_piece_count == self.last_piece_count:
                self.stall_counter += 1
            else:
                self.stall_counter = 0
            self.last_piece_count = current_piece_count

            if self.stall_counter >= 10:
                # 10 full consecutive turns without a capture => draw
                print("Draw due to no change in piece count for 10 consecutive turns!")
                msg = f"Draw (no captures in 10 consecutive turns) at turn {self.turn_number}"
                self.log_winner(msg)
                self.close_log()
                break

            # Advance to the next turn
            self.turn_number += 1

    def ai_flow(self):
        """
        AI picks a move using Minimax/Alpha-Beta with the chosen heuristic.
        Uses different configurations for White and Black AI.
        Returns (chosen_move, move_time, heuristic_score, search_score).
        """
        # Determine which AI is playing
        is_white_turn = (self.current_game_state['turn'] == 'white')
        
        # Select the correct AI configuration
        max_time = self.max_time_white if is_white_turn else self.max_time_black
        use_alpha_beta = self.use_alpha_beta_white if is_white_turn else self.use_alpha_beta_black
        heuristic_name = self.heuristic_name_white if is_white_turn else self.heuristic_name_black

        start_time = time.time()
        search_depth = 3  # Can be adjusted dynamically later

        # AI selects the best move
        chosen_move, best_score, states_explored_by_depth = choose_best_move(
            game=self,
            game_state=self.current_game_state,
            max_depth=search_depth,
            use_alpha_beta=use_alpha_beta,
            heuristic_name=heuristic_name
        )

        elapsed = time.time() - start_time

        # Handle AI failures or rule violations
        if not chosen_move:
            print(f"{self.current_game_state['turn'].capitalize()} AI has no moves. Game ends.")
            self.log_winner(f"{self.current_game_state['turn'].capitalize()} AI lost: No moves available.")
            self.close_log()
            sys.exit(0)

        if elapsed > max_time:
            print(f"{self.current_game_state['turn'].capitalize()} AI exceeded {max_time}s time limit and loses.")
            self.log_winner(f"{self.current_game_state['turn'].capitalize()} AI lost by time forfeit.")
            self.close_log()
            sys.exit(0)

        if not self.is_valid_move(self.current_game_state, chosen_move):
            print(f"{self.current_game_state['turn'].capitalize()} AI made an illegal move and loses.")
            self.log_winner(f"{self.current_game_state['turn'].capitalize()} AI lost by illegal move.")
            self.close_log()
            sys.exit(0)

        # Update cumulative states explored
        total_states_explored = sum(states_explored_by_depth.values())
        self.cumulative_states_explored += total_states_explored

        # Update states_by_depth
        for depth, count in states_explored_by_depth.items():
            self.states_by_depth[depth] = self.states_by_depth.get(depth, 0) + count

        return (chosen_move, elapsed, best_score, best_score)

    def log_end_and_close(self):
        """
        Called after the game ends to log final result and close the file.
        Prints the final result once, with the # of turns.
        """
        kings = [
            piece for row in self.current_game_state["board"] for piece in row
            if piece in ['wK', 'bK']
        ]
        if len(kings) == 1:
            if 'wK' in kings:
                print(f"White Wins in {self.turn_number} turns!")
                msg = f"White won in {self.turn_number} turns"
            else:
                print(f"Black Wins in {self.turn_number} turns!")
                msg = f"Black won in {self.turn_number} turns"
        else:
            print(f"Draw after {self.turn_number} turns!")
            msg = f"Draw after {self.turn_number} turns"

        self.log_winner(msg)
        self.close_log()

    def play(self):
        print(f"\nStarting MiniChess with: "
              f"white_max_time={self.max_time_white},black_max_time={self.max_time_black} "
              f"max_turns={self.max_turns}, "
              f"white_use_alpha_beta={self.use_alpha_beta_white}, mode={self.mode},black_use_alpha_beta={self.use_alpha_beta_black}, mode={self.mode} "
              f"human_color={self.human_color}, "
              f"white_heuristic_name={self.heuristic_name_white},black_heuristic_name={self.heuristic_name_black}.")
        print(f"Logging to file: {self.log_filename}")
        self.run_game_loop()


def main():
    print("Welcome to MiniChess Setup!")
    
    # 1) MODE
    while True:
        print("Select Play Mode:")
        print("1 - Human vs Human")
        print("2 - Human vs AI")
        print("3 - AI vs AI")
        choice = input("Enter choice (1/2/3): ")
        if choice not in ['1','2','3']:
            print("Invalid choice. Try again.")
        else:
            mode = int(choice)
            break

    # If mode=2 => ask for color
    human_color = 'white'
    if mode == 2:
        while True:
            color = input("Choose your color (white/black): ").lower()
            if color not in ["white", "black"]:
                print("Invalid color. Try again.")
            else:
                human_color = color
                break

    # 2) MAX TURNS
    while True:
        try:
            mt = input("Enter maximum number of full turns (default=10): ")
            if mt.strip() == '':
                max_turns = 10
                break
            else:
                max_turns = int(mt)
                if max_turns <= 0:
                    raise ValueError
                break
        except ValueError:
            print("Invalid integer. Try again.")

    # If mode=1 => no AI parameters needed
    if mode == 1:
        max_time = 3
        use_alpha_beta_white = True
        use_alpha_beta_black = True
        heuristic_name_white = 'e0'
        heuristic_name_black = 'e0'

    elif mode == 2:
        # AI vs Human => ask for single AI settings
        while True:
            t = input("Enter maximum time in seconds for AI (default=3): ")
            if t.strip() == '':
                max_time = 3
                break
            else:
                try:
                    max_time = int(t)
                    if max_time <= 0:
                        raise ValueError
                    break
                except ValueError:
                    print("Invalid integer. Try again.")

        while True:
            ab = input("Use alpha-beta? (True/False, default=True): ").lower()
            if ab.strip() == '':
                use_alpha_beta = True
                break
            elif ab in ['true', 'false']:
                use_alpha_beta = (ab == 'true')
                break
            else:
                print("Invalid input. Enter True or False.")

        while True:
            print("Choose your heuristic (e0/e1/e2). Default = e0.")
            heuristic_choice = input("Heuristic: ").lower()
            if heuristic_choice.strip() == '':
                heuristic_name = 'e0'
                break
            elif heuristic_choice in ['e0', 'e1', 'e2']:
                heuristic_name = heuristic_choice
                break
            else:
                print("Invalid choice. Pick from e0, e1, e2.")

        # Assign AI settings based on color choice
        if human_color == "white":
            use_alpha_beta_white = False  # Human doesn't use AI
            heuristic_name_white = None
            use_alpha_beta_black = use_alpha_beta
            heuristic_name_black = heuristic_name
        else:
            use_alpha_beta_black = False  # Human doesn't use AI
            heuristic_name_black = None
            use_alpha_beta_white = use_alpha_beta
            heuristic_name_white = heuristic_name

    else:  # mode == 3 (AI vs AI)
        print("\n=== AI vs AI Configuration ===")
        
        # White AI settings
        print("\nWhite AI:")
        while True:
            t = input("Enter max time for White AI (default=3): ")
            if t.strip() == '':
                max_time_white = 3
                break
            else:
                try:
                    max_time_white = int(t)
                    if max_time_white <= 0:
                        raise ValueError
                    break
                except ValueError:
                    print("Invalid integer. Try again.")

        while True:
            ab = input("Use alpha-beta for White AI? (True/False, default=True): ").lower()
            if ab.strip() == '':
                use_alpha_beta_white = True
                break
            elif ab in ['true', 'false']:
                use_alpha_beta_white = (ab == 'true')
                break
            else:
                print("Invalid input. Enter True or False.")

        while True:
            print("Choose heuristic for White AI (e0/e1/e2). Default = e0.")
            heuristic_choice = input("Heuristic: ").lower()
            if heuristic_choice.strip() == '':
                heuristic_name_white = 'e0'
                break
            elif heuristic_choice in ['e0', 'e1', 'e2']:
                heuristic_name_white = heuristic_choice
                break
            else:
                print("Invalid choice. Pick from e0, e1, e2.")

        # Black AI settings
        print("\nBlack AI:")
        while True:
            t = input("Enter max time for Black AI (default=3): ")
            if t.strip() == '':
                max_time_black = 3
                break
            else:
                try:
                    max_time_black = int(t)
                    if max_time_black <= 0:
                        raise ValueError
                    break
                except ValueError:
                    print("Invalid integer. Try again.")

        while True:
            ab = input("Use alpha-beta for Black AI? (True/False, default=True): ").lower()
            if ab.strip() == '':
                use_alpha_beta_black = True
                break
            elif ab in ['true', 'false']:
                use_alpha_beta_black = (ab == 'true')
                break
            else:
                print("Invalid input. Enter True or False.")

        while True:
            print("Choose heuristic for Black AI (e0/e1/e2). Default = e0.")
            heuristic_choice = input("Heuristic: ").lower()
            if heuristic_choice.strip() == '':
                heuristic_name_black = 'e0'
                break
            elif heuristic_choice in ['e0', 'e1', 'e2']:
                heuristic_name_black = heuristic_choice
                break
            else:
                print("Invalid choice. Pick from e0, e1, e2.")

    # Create the game object with separate AI settings
    game = MiniChess(
        max_time_white=max_time_white if mode == 3 else max_time,
        max_time_black=max_time_black if mode == 3 else max_time,
        max_turns=max_turns,
        use_alpha_beta_white=use_alpha_beta_white,
        use_alpha_beta_black=use_alpha_beta_black,
        heuristic_name_white=heuristic_name_white,
        heuristic_name_black=heuristic_name_black,
        mode=mode,
        human_color=human_color
    )

    game.play()

if __name__ == "__main__":
    main()
