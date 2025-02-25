import math
import copy
import time
import sys

class MiniChess:
    def __init__(
        self,
        max_time=3,            
        max_turns=10,          
        use_alpha_beta=True,   
        mode=1,                
        human_color='white',   
        heuristic_name='e0' 
    ):
        """
        If mode=1 => Human vs Human
        If mode=2 => Human vs AI (human_color='white' or 'black')
        If mode=3 => AI vs AI
        """
        self.current_game_state = self.init_board()
        self.max_time           = max_time
        self.max_turns          = max_turns
        self.use_alpha_beta     = use_alpha_beta
        self.mode               = mode
        self.human_color        = human_color
        self.heuristic_name     = heuristic_name

        # Decide control of White/Black
        self.white_player = 'human'
        self.black_player = 'human'

        if self.mode == 2:  # Human vs AI
            if self.human_color == 'white':
                self.white_player = 'human'
                self.black_player = 'ai'
            else:
                self.white_player = 'ai'
                self.black_player = 'human'
        elif self.mode == 3:  # AI vs AI
            self.white_player = 'ai'
            self.black_player = 'ai'

        # One "turn" = White moves, then Black moves
        self.turn_number = 1

        # Build filename e.g. "gameTrace-true-3-10.txt"
        b_str = str(self.use_alpha_beta).lower()  # "true"/"false"
        t_str = str(self.max_time)
        n_str = str(self.max_turns)
        self.log_filename = f"gameTrace-{b_str}-{t_str}-{n_str}.txt"

        # Open the file for logging
        self.log_file = open(self.log_filename, "w", encoding="utf-8")

        # Placeholder stats for AI
        self.cumulative_states_explored = 0
        self.states_by_depth = {}  # e.g. {0:count, 1:count,...}

        # Write initial log info
        self.log_initial_parameters()
        self.log_initial_board()

    def init_board(self):
        """
        5x5 mini-chess board
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
        self.log(f"(a) Time limit (t) = {self.max_time} seconds")
        self.log(f"(b) Max number of turns (n) = {self.max_turns}")
        self.log(f"(c) Play mode => {mode_str}")

        if (self.white_player == 'ai') or (self.black_player == 'ai'):
            ab_str = "ON" if self.use_alpha_beta else "OFF"
            self.log(f"(d) Alpha-beta is {ab_str}")
            self.log(f"(e) Heuristic name = {self.heuristic_name}")
        else:
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
         f) AI minimax/alpha-beta search score
         g) new board config

        3.2) If AI, cumulative info: states explored, etc. (placeholders)
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
            depth_details = ", ".join([f"{d}:{cnt}" for d, cnt in sorted(self.states_by_depth.items())])
            self.log(f"    (b) States by depth: {depth_details}")
            total_states = max(self.cumulative_states_explored, 1)
            percentages = ", ".join(
                [f"{d}:{(cnt/total_states)*100:.1f}%" for d, cnt in sorted(self.states_by_depth.items())]
            )
            self.log(f"    (c) % states by depth: {percentages}")
            self.log(f"    (d) average branching factor: 2.5 (placeholder)")
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
        Row '1' => row4, row '5' => row0; 
        Col 'A' => col0, 'B' => col1, etc.
        Example: "B3" => (2,1); "B3 B5" => ((2,1),(0,1)).
        Returns None if parsing fails.
        """
        try:
            start, end = move_str.split()
            start_col = ord(start[0].upper()) - ord('A')   # 'A' => 0, 'B' => 1, ...
            start_row = 5 - int(start[1])                  # '1' => row4, '5' => row0
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
                if (0 <= frontRow < 5 and 0 <= col < 5 and
                    game_state["board"][frontRow][col] == '.'):
                    valid_moves.append(((row, col), (frontRow, col)))
                if (0 <= frontRow < 5 and 0 <= leftCol < 5 and
                    game_state["board"][frontRow][leftCol].startswith(opponentColor)):
                    valid_moves.append(((row, col), (frontRow, leftCol)))
                if (0 <= frontRow < 5 and 0 <= rightCol < 5 and
                    game_state["board"][frontRow][rightCol].startswith(opponentColor)):
                    valid_moves.append(((row, col), (frontRow, rightCol)))

            elif piece_type == 'N':
                front2Row = row - 2 if pieceColor == 'w' else row + 2
                back2Row  = row + 2 if pieceColor == 'w' else row - 2
                left2Col  = col - 2
                right2Col = col + 2

                knight_moves = [
                    (front2Row, rightCol),
                    (front2Row, leftCol),
                    (row - 1 if pieceColor == 'w' else row + 1, right2Col),
                    (row - 1 if pieceColor == 'w' else row + 1, left2Col),
                    (back2Row,  rightCol),
                    (back2Row,  leftCol),
                    (backRow,   right2Col),
                    (backRow,   left2Col),
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

        game_state["turn"] = 'black' if game_state["turn"] == 'white' else 'white'
        return game_state

    def game_over(self, game_state):
        kings = [
            piece for row in game_state["board"] for piece in row
            if piece in ['wK', 'bK']
        ]
        if len(kings) == 1:
            if 'wK' in kings:
                print("White Wins!")
                return True
            else:
                print("Black Wins!")
                return True

        if self.turn_number > self.max_turns:
            print(f"Reached max turn limit = {self.max_turns}. Game is a draw.")
            return True
        
        return False

    def human_move(self):
        """
        Prompt the human player for a move (e.g. "B3 B5"), 
        parse/validate it, and return the move if valid.
        """
        while True:
            move_input = input(f"{self.current_game_state['turn'].capitalize()} to move (e.g. B3 B5), or 'exit': ")
            if move_input.lower() == 'exit':
                print("Game exited by user.")
                self.log_winner("Game exited by user.")
                self.close_log()
                sys.exit(0)
            
            parsed = self.parse_input(move_input)
            if parsed and self.is_valid_move(self.current_game_state, parsed):
                return parsed
            else:
                print("Invalid move. Try again.")

    def calculate_heuristic_e0(self, game_state):
        """
        Calculate the heuristic value e0 for the given game state.
        e0 = (#wp + 3·#wB + 3·#wN + 9·#wQ + 999·wK) − (#bp + 3·#bB + 3·#bN + 9·#bQ + 999·bK)
        """
        white_pieces = {
            'p': 0,
            'B': 0,
            'N': 0,
            'Q': 0,
            'K': 0
        }
        black_pieces = {
            'p': 0,
            'B': 0,
            'N': 0,
            'Q': 0,
            'K': 0
        }

        for row in game_state["board"]:
            for piece in row:
                if piece.startswith('w'):
                    piece_type = piece[1]
                    if piece_type in white_pieces:
                        white_pieces[piece_type] += 1
                elif piece.startswith('b'):
                    piece_type = piece[1]
                    if piece_type in black_pieces:
                        black_pieces[piece_type] += 1

        white_score = (
            white_pieces['p'] +
            3 * white_pieces['B'] +
            3 * white_pieces['N'] +
            9 * white_pieces['Q'] +
            999 * white_pieces['K']
        )
        black_score = (
            black_pieces['p'] +
            3 * black_pieces['B'] +
            3 * black_pieces['N'] +
            9 * black_pieces['Q'] +
            999 * black_pieces['K']
        )

        return white_score - black_score

    def ai_flow(self):
        """
        AI decision-making process.
        """
        start_time = time.time()
        best_move = None
        best_value = -math.inf if self.current_game_state["turn"] == "white" else math.inf

        # Get all valid moves for the current player
        valid_moves = self.valid_moves(self.current_game_state)

        for move in valid_moves:
            # Make a copy of the game state and apply the move
            new_state = copy.deepcopy(self.current_game_state)
            new_state = self.make_move(new_state, move)

            # Calculate the heuristic value for the new state
            heuristic_value = self.calculate_heuristic_e0(new_state)

            # Update the best move based on the heuristic value
            if self.current_game_state["turn"] == "white":
                if heuristic_value > best_value:
                    best_value = heuristic_value
                    best_move = move
            else:
                if heuristic_value < best_value:
                    best_value = heuristic_value
                    best_move = move

        move_time = time.time() - start_time
        return best_move, move_time, best_value, best_value

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
                chosen_move, move_time, heur, search_s = self.ai_flow()
            else:
                chosen_move = self.human_move()
                move_time   = 0.0
                heur        = None
                search_s    = None

            # Perform the move
            self.make_move(self.current_game_state, chosen_move)
            # Log the move
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
                chosen_move, move_time, heur, search_s = self.ai_flow()
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
                self.log_end_and_close()
                break

            self.turn_number += 1

    def log_end_and_close(self):
        """
        Called after the game ends to log final result and close the file.
        """
        kings = [
            piece for row in self.current_game_state["board"] for piece in row
            if piece in ['wK', 'bK']
        ]
        if len(kings) == 1:
            if 'wK' in kings:
                msg = f"White won in {self.turn_number} turns"
            else:
                msg = f"Black won in {self.turn_number} turns"
        else:
            msg = f"Draw after {self.turn_number} turns"

        self.log_winner(msg)
        self.close_log()

    def play(self):
        print(f"\nStarting MiniChess with: "
              f"max_time={self.max_time}, max_turns={self.max_turns}, "
              f"use_alpha_beta={self.use_alpha_beta}, mode={self.mode}, "
              f"human_color={self.human_color}.")
        print(f"Logging to file: {self.log_filename}")
        self.run_game_loop()


def main():
    print("Welcome to MiniChess Setup!")
    
    # 1) MODE
    while True:
        print("Select Play Mode:")
        print("1 - Human vs Human")
        print("2 - Human vs AI (Not Implemented Yet)")
        print("3 - AI vs AI (Not Implemented Yet)")
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

    # If mode=1 => no AI => skip time/alpha-beta
    if mode in [2, 3]:
        # 3) MAX TIME
        while True:
            try:
                t = input("Enter maximum time in seconds for AI (default=3): ")
                if t.strip() == '':
                    max_time = 3
                    break
                else:
                    max_time = int(t)
                    if max_time <= 0:
                        raise ValueError
                    break
            except ValueError:
                print("Invalid integer. Try again.")

        # 4) ALPHA-BETA or MINIMAX
        while True:
            ab = input("Use alpha-beta? (True/False, default=True): ").lower()
            if ab.strip() == '':
                use_alpha_beta = True
                break
            elif ab in ['true','false']:
                use_alpha_beta = (ab == 'true')
                break
            else:
                print("Invalid input. Enter True or False.")
    else:
        # Human vs Human => no AI parameters
        max_time       = 3
        use_alpha_beta = True

    heuristic_name = 'e0'

    # Create the game object
    game = MiniChess(
        max_time=max_time,
        max_turns=max_turns,
        use_alpha_beta=use_alpha_beta,
        mode=mode,
        human_color=human_color,
        heuristic_name=heuristic_name
    )
    game.play()

if __name__ == "__main__":
    main()