import math
import copy
import time
import sys
from datetime import datetime

# NEW: We'll import our search logic (with e0, e1, e2 heuristics)
from search import choose_best_move

# NEW: Short number formatting (for large counts)
def short_number_fmt(num):
    """
    Convert a large integer to a short string like:
      1,100 => '1.1k'
      7,300 => '7.3k'
      46,700 => '46.7k'
      286,500 => '286.5k'
      1,800,000 => '1.8M'
    Otherwise just str(num).
    """
    if num >= 1_000_000:
        return f"{num / 1_000_000:.1f}M"
    elif num >= 1_000:
        return f"{num / 1_000:.1f}k"
    else:
        return str(num)


class MiniChess:
    def __init__(
        self,
        max_time_white,
        max_time_black,
        max_turns,
        use_alpha_beta_white,
        use_alpha_beta_black,
        heuristic_name_white,
        heuristic_name_black,
        mode,
        human_color
    ):
        """
        If mode=1 => Human vs Human
        If mode=2 => Human vs AI (human_color='white' or 'black')
        If mode=3 => AI vs AI
        """
        self.max_time_white        = max_time_white
        self.max_time_black        = max_time_black
        self.max_turns             = max_turns
        self.use_alpha_beta_white  = use_alpha_beta_white
        self.use_alpha_beta_black  = use_alpha_beta_black
        self.heuristic_name_white  = heuristic_name_white
        self.heuristic_name_black  = heuristic_name_black
        self.mode                  = mode
        self.human_color           = human_color

        self.current_game_state = self.init_board()
        self.turn_number        = 1
        
        # Decide who is AI vs human
        self.white_player = 'human' if (mode == 1 or (mode == 2 and human_color == 'white')) else 'ai'
        self.black_player = 'human' if (mode == 1 or (mode == 2 and human_color == 'black')) else 'ai'

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

        # Stats for AI
        self.cumulative_states_explored = 0
        self.states_by_depth = {}  # e.g. {0:count, 1:count,...}

        # NEW: expansions data
        self.nodes_visited = 0
        self.total_branchings = 0

        # Write initial log info
        self.log_initial_parameters()
        self.log_initial_board()

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
           e) heuristic name(s)
        """
        def player_desc(player):
            return "AI" if player == 'ai' else "H"

        mode_str = f"Player1={player_desc(self.white_player)} & Player2={player_desc(self.black_player)}"
        self.log("----- Game Parameters -----")
        self.log(f"(a) Time limit (t): ")
        self.log(f"    White Max Time: {self.max_time_white} seconds")
        self.log(f"    Black Max Time: {self.max_time_black} seconds")
        self.log(f"(b) Max number of turns (n) = {self.max_turns}")
        self.log(f"(c) Play mode => {mode_str}")

        if self.mode == 1:
            # No AI => alpha-beta not applicable
            self.log("(d) No AI in this game => alpha-beta N/A")

        elif self.mode == 2:
            # Only one side is AI, so we mention that side’s alpha-beta & heuristic
            if self.white_player == 'ai':
                ab_str = "ON" if self.use_alpha_beta_white else "OFF"
                self.log(f"(d) Alpha-beta is {ab_str}")
                self.log(f"(e) Heuristic name = {self.heuristic_name_white}")
            else:
                ab_str = "ON" if self.use_alpha_beta_black else "OFF"
                self.log(f"(d) Alpha-beta is {ab_str}")
                self.log(f"(e) Heuristic name = {self.heuristic_name_black}")

        else:  # mode == 3 => AI vs AI
            ab_w_str = "ON" if self.use_alpha_beta_white else "OFF"
            ab_b_str = "ON" if self.use_alpha_beta_black else "OFF"
            self.log(f"(d) White Alpha-beta: {ab_w_str}, Black Alpha-beta: {ab_b_str}")
            self.log(f"(e) White Heuristic: {self.heuristic_name_white}, Black Heuristic: {self.heuristic_name_black}")

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

    # NEW: compute average BF
    def compute_average_bf(self):
        """
        Compute average branching factor: 
        total_branchings / (nodes_visited - 1), if nodes_visited > 1.
        """
        if self.nodes_visited > 1:
            return self.total_branchings / (self.nodes_visited - 1)
        else:
            return 0.0

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
            self.log("    [Cumulative AI Info]")
            self.log(f"    (a) States explored so far: {self.cumulative_states_explored:,}")

            # Summaries by depth
            depth_strs = []
            for d in sorted(self.states_by_depth.keys()):
                count = self.states_by_depth[d]
                depth_strs.append(f"{d}={short_number_fmt(count)}")
            depth_details = " ".join(depth_strs)
            self.log(f"    (b) States by depth: {depth_details}")

            total_states = max(self.cumulative_states_explored, 1)
            pct_strs = []
            for d in sorted(self.states_by_depth.keys()):
                cnt = self.states_by_depth[d]
                pct = (cnt / total_states) * 100
                pct_strs.append(f"{d}={pct:.1f}%")
            percentages = " ".join(pct_strs)
            self.log(f"    (c) % states by depth: {percentages}")

            avg_bf = self.compute_average_bf()
            self.log(f"    (d) average branching factor: {avg_bf:.1f}")

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
            backRow  = row + 1 if pieceColor == 'w' else row - 1
            leftCol  = col - 1
            rightCol = col + 1

            if piece_type == 'p':
                # Forward
                if (0 <= frontRow < 5 and 0 <= col < 5 and
                    game_state["board"][frontRow][col] == '.'):
                    valid_moves.append(((row, col), (frontRow, col)))
                # Capture left
                if (0 <= frontRow < 5 and 0 <= leftCol < 5 and
                    game_state["board"][frontRow][leftCol].startswith(opponentColor)):
                    valid_moves.append(((row, col), (frontRow, leftCol)))
                # Capture right
                if (0 <= frontRow < 5 and 0 <= rightCol < 5 and
                    game_state["board"][frontRow][rightCol].startswith(opponentColor)):
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
        """
        # Check if only one King remains
        kings = [
            piece for row in game_state["board"] for piece in row
            if piece in ['wK', 'bK']
        ]
        if len(kings) == 1:
            return True

        # Or if max turns reached
        if self.turn_number > self.max_turns:
            return True

        return False

    def human_move(self):
        """
        Prompt the human player for a move, parse/validate it, and return if valid.
        """
        while True:
            move_input = input(f"{self.current_game_state['turn'].capitalize()} to move (e.g. B3 B5), or 'exit': ")
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
                chosen_move, move_time, heur, search_s = self.ai_flow(is_white=True)
            else:
                chosen_move = self.human_move()
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
                chosen_move, move_time, heur, search_s = self.ai_flow(is_white=False)
            else:
                chosen_move = self.human_move()
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

            if self.game_over(self.current_game_state):
                self.display_board(self.current_game_state)
                self.log_end_and_close()
                break

            self.turn_number += 1

    def ai_flow(self, is_white):
        """
        AI picks a move using Minimax/Alpha-Beta with the chosen heuristic.
        Uses separate config for White vs Black (time, alpha-beta, heuristic).
        Returns (chosen_move, move_time, heuristic_score, search_score).
        """
        # Reset expansions for this search
        self.nodes_visited = 0
        self.total_branchings = 0

        # Decide parameters
        if is_white:
            max_time = self.max_time_white
            use_alpha_beta = self.use_alpha_beta_white
            heuristic_name = self.heuristic_name_white
        else:
            max_time = self.max_time_black
            use_alpha_beta = self.use_alpha_beta_black
            heuristic_name = self.heuristic_name_black

        start_time = time.time()
        search_depth = 3  # Could be set dynamically

        # Perform the search
        chosen_move, best_score = choose_best_move(
            game=self,
            game_state=self.current_game_state,
            max_depth=search_depth,
            use_alpha_beta=use_alpha_beta,
            heuristic_name=heuristic_name
        )

        elapsed = time.time() - start_time

        # Handle no move found
        if not chosen_move:
            print(f"{self.current_game_state['turn'].capitalize()} AI has no moves. Game ends.")
            self.log_winner(f"{self.current_game_state['turn'].capitalize()} AI lost: No moves available.")
            self.close_log()
            sys.exit(0)

        # Handle time forfeit
        if elapsed > max_time:
            print(f"{self.current_game_state['turn'].capitalize()} AI exceeded {max_time}s time limit and loses.")
            self.log_winner(f"{self.current_game_state['turn'].capitalize()} AI lost by time forfeit.")
            self.close_log()
            sys.exit(0)

        # Handle illegal move
        if not self.is_valid_move(self.current_game_state, chosen_move):
            print(f"{self.current_game_state['turn'].capitalize()} AI made an illegal move and loses.")
            self.log_winner(f"{self.current_game_state['turn'].capitalize()} AI lost by illegal move.")
            self.close_log()
            sys.exit(0)

        # We'll treat best_score as both heuristic_score and search_score
        heuristic_score = best_score
        search_score = best_score

        # For demonstration, just say each AI search encountered 10 states
        self.cumulative_states_explored += 10
        self.states_by_depth[search_depth] = self.states_by_depth.get(search_depth, 0) + 10

        return (chosen_move, elapsed, heuristic_score, search_score)

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
        print(
            f"\nStarting MiniChess with:\n"
            f"  White time={self.max_time_white}s, Black time={self.max_time_black}s,\n"
            f"  max_turns={self.max_turns},\n"
            f"  White use_alpha_beta={self.use_alpha_beta_white}, Black use_alpha_beta={self.use_alpha_beta_black},\n"
            f"  mode={self.mode}, human_color={self.human_color},\n"
            f"  White heuristic={self.heuristic_name_white}, Black heuristic={self.heuristic_name_black}.\n"
            f"Logging to file: {self.log_filename}"
        )
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
        max_time_white = 3
        max_time_black = 3
        use_alpha_beta_white = False  # No AI anyway
        use_alpha_beta_black = False
        heuristic_name_white = None
        heuristic_name_black = None

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

        # Assign AI settings based on which color the human chose
        if human_color == "white":
            # White = human => no AI settings
            max_time_white = 0
            use_alpha_beta_white = False
            heuristic_name_white = None

            # Black = AI => use chosen
            max_time_black = max_time
            use_alpha_beta_black = use_alpha_beta
            heuristic_name_black = heuristic_name
        else:
            # Black = human => no AI settings
            max_time_black = 0
            use_alpha_beta_black = False
            heuristic_name_black = None

            # White = AI => use chosen
            max_time_white = max_time
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
        max_time_white=max_time_white,
        max_time_black=max_time_black,
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
