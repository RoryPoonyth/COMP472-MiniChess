import math
import copy
import time
import argparse

class MiniChess:
    def __init__(self):
        self.current_game_state = self.init_board()

    """
    Initialize the board

    Args:
        - None
    Returns:
        - state: A dictionary representing the state of the game
    """
    def init_board(self):
        state = {
                "board": 
                [['bK', 'bQ', 'bB', 'bN', '.'],
                ['.', '.', 'bp', 'bp', '.'],
                ['.', '.', '.', '.', '.'],
                ['.', 'wp', 'wp', '.', '.'],
                ['.', 'wN', 'wB', 'wQ', 'wK']],
                "turn": 'white',
                }
        return state

    """
    Prints the board
    
    Args:
        - game_state: Dictionary representing the current game state
    Returns:
        - None
    """
    def display_board(self, game_state):
        print()
        for i, row in enumerate(game_state["board"], start=1):
            print(str(6-i) + "  " + ' '.join(piece.rjust(3) for piece in row))
        print()
        print("     A   B   C   D   E")
        print()

    """
    Returns a list of valid moves

    Args:
        - game_state:   dictionary | Dictionary representing the current game state
    Returns:
        - valid moves:   list | A list of nested tuples corresponding to valid moves [((start_row, start_col),(end_row, end_col)),((start_row, start_col),(end_row, end_col))]
    """
    def valid_moves(self, game_state):
        
        pieceColor = 'w' if game_state["turn"] == "white" else 'b'
        """
        Get all pieces of the current player and their positions 
        Example of desired format of the list corresponding to the base board
        [['wp', 2, 1], ['wp', 2, 2]] 
        @Omar
        """
        pieces = [
            [piece, rowIndex, colIndex]
            for rowIndex, row in enumerate(game_state["board"])
            for colIndex, piece in enumerate(row)
            if piece.startswith(pieceColor)
        ]
                
        valid_moves = []
        """
        Iterate through all pieces from the player and compose list of possible moves
        The expected output should be an array of start end moves, based from the parse
        input move for instance 
        valid_moves = [[start] [end], [start] [end], [start] [end]]
        valid_moves = [ [1,1]  [1,2],  [2,2]  [2,4],  [2,2]  [2,5]]
        @Omar
        """
        for piece in pieces:
            """
            Get current position of the piece reminder:
            [    'wp',      2,       1     ]
            [    piece,     row,    col    ]
            [  piece[0], piece[1], piece[2]]
            @Omar
            """
            # Variable Declaration for Improving Code Readability
            opponentColor = 'b' if pieceColor == 'w' else 'w'
            row, col = piece[1], piece[2]  
            frontRow = row - 1 if pieceColor == 'w' else row + 1
            backRow = row + 1 if pieceColor == 'w' else row - 1
            leftCol = col - 1
            rightCol = col + 1

            match piece[0][1]:
                case 'p':  # Pawn
                    """
                    Moving Pawn Forward
                    ['.']
                    ['wp']
                    A pawn can move forward only when there is an empty space '.' in front of it. 
                    The following logic will be
                    if empty space and row is inside board and column is inside board 
                        then append to valid moves

                    @Omar
                    """
                    # Move Forward                   
                    if ( game_state["board"][frontRow][col] == '.' and
                        0 <= frontRow < 5 and 
                        0 <= col < 5
                    ):
                        move = ((row, col),(frontRow, col))
                        valid_moves.append(move)
                    """
                    Pawn Attacking
                    ['bp','.','bp']
                    ['.','wp','.']
                    A pawn can move attack diagonally only when there is an opponent digonally infront of
                    The following logic will be
                    Left Attack
                    if not empty space '.' infront on the left -1 row and row is inside board and left column is inside board 
                        then append to valid moves
                    
                    Right Attack
                    if not empty space '.' infront on the right +1  row and row is inside board and left column is inside board 
                        then append to valid moves
                    @Omar
                    """
                    # Left Attack
                    if ( game_state["board"][frontRow][leftCol].startswith(opponentColor) and
                        0 <= frontRow < 5 and 
                        0 <= leftCol < 5
                    ):
                        move = ((row, col),(frontRow, leftCol))
                        valid_moves.append(move)
                        
                    # Right Attack
                    if ( game_state["board"][frontRow][rightCol].startswith(opponentColor) and
                        0 <= frontRow < 5 and 
                        0 <= rightCol < 5
                    ):
                        move = ((row, col),(frontRow, rightCol))
                        valid_moves.append(move)
                        
                case 'N':  # Knight
                    front2Row = row - 2 if pieceColor == 'w' else row + 2
                    back2Row = row + 2 if pieceColor == 'w' else row - 2
                    left2Col = col - 2
                    right2Col = col + 2

                    # Move 2 Forward and 1 Right                 
                    if (0 <= front2Row < 5 and 
                        0 <= rightCol < 5 and (
                        game_state["board"][front2Row][rightCol] == '.' or
                        game_state["board"][front2Row][rightCol].startswith(opponentColor))
                    ):
                        move = ((row, col),(front2Row, rightCol))
                        valid_moves.append(move)
                        
                    # Move 2 Forward and 1 Left                 
                    if (0 <= front2Row < 5 and 
                        0 <= leftCol < 5 and (
                        game_state["board"][front2Row][leftCol] == '.' or
                        game_state["board"][front2Row][leftCol].startswith(opponentColor))
                    ):
                        move = ((row, col),(front2Row, leftCol))
                        valid_moves.append(move)
                        
                    # Move 1 Forward and 2 Right                 
                    if (0 <= frontRow < 5 and 
                        0 <= right2Col < 5 and (
                        game_state["board"][frontRow][right2Col] == '.' or
                        game_state["board"][frontRow][right2Col].startswith(opponentColor))
                    ):
                        move = ((row, col),(frontRow, right2Col))
                        valid_moves.append(move)
                        
                    # Move 1 Forward and 2 Left                 
                    if (0 <= frontRow < 5 and 
                        0 <= left2Col < 5 and (
                        game_state["board"][frontRow][left2Col] == '.' or
                        game_state["board"][frontRow][left2Col].startswith(opponentColor))
                    ):
                        move = ((row, col),(frontRow, left2Col))
                        valid_moves.append(move)
                        
                    # Move 2 Backward and 1 Right                 
                    if (0 <= back2Row < 5 and 
                        0 <= rightCol < 5 and (
                        game_state["board"][back2Row][rightCol] == '.' or
                        game_state["board"][back2Row][rightCol].startswith(opponentColor))
                    ):
                        move = ((row, col),(back2Row, rightCol))
                        valid_moves.append(move)
                        
                    # Move 2 Backward and 1 Left                 
                    if (0 <= back2Row  < 5 and 
                        0 <= leftCol < 5 and (
                        game_state["board"][back2Row][leftCol] == '.' or
                        game_state["board"][back2Row][leftCol].startswith(opponentColor))
                    ):
                        move = ((row, col),(back2Row, leftCol))
                        valid_moves.append(move)
                        
                    # Move 1 Backward and 2 Right                 
                    if (0 <= backRow < 5 and 
                        0 <= right2Col < 5 and (
                        game_state["board"][backRow][right2Col] == '.' or
                        game_state["board"][backRow][right2Col].startswith(opponentColor))
                    ):
                        move = ((row, col),(backRow, right2Col))
                        valid_moves.append(move)
                        
                    # Move 1 Backward and 2 Left                 
                    if (0 <= backRow < 5 and 
                        0 <= left2Col < 5 and (
                        game_state["board"][backRow][left2Col] == '.' or
                        game_state["board"][backRow][left2Col].startswith(opponentColor))
                    ):
                        move = ((row, col),(backRow, left2Col))
                        valid_moves.append(move)
                        
                case 'B':  # Bishop
                    print(f"Bishop")
                case 'Q':  # Queen
                    print(f"Queen")
                case 'K':  # King
                    print(f"King")

        return valid_moves


    """
    Check if the move is valid    
    
    Args: 
        - game_state:   dictionary | Dictionary representing the current game state
        - move          tuple | the move which we check the validity of ((start_row, start_col),(end_row, end_col))
    Returns:
        - boolean representing the validity of the move
    """
    def is_valid_move(self, game_state, move):
        # Check if move is in list of valid moves
        if move in self.valid_moves(game_state):
            return True

    """
    Modify to board to make a move

    Args: 
        - game_state:   dictionary | Dictionary representing the current game state
        - move          tuple | the move to perform ((start_row, start_col),(end_row, end_col))
    Returns:
        - game_state:   dictionary | Dictionary representing the modified game state
    """
    def make_move(self, game_state, move):
        start = move[0]
        end = move[1]
        start_row, start_col = start
        end_row, end_col = end
        piece = game_state["board"][start_row][start_col]
        game_state["board"][start_row][start_col] = '.'
        game_state["board"][end_row][end_col] = piece
        game_state["turn"] = "black" if game_state["turn"] == "white" else "white"
        
        return game_state

    """
    Game Over: Game Ends if the oponnents king is capture or if the maximum number of turns has been reached
    @Omar
    """

    def game_over(self, game_state):
        # Game Over Knight Captured
        kings = [
            piece
            for row in game_state["board"]
            for piece in row
            if piece == 'wK' or piece == 'bK'
        ]
        
        if(len(kings) == 1 ):
            if 'wK' in kings:
                print("White Wins") 
            else:
                print("Black Wins") 
            exit(1)
        
        # Game Over Maximum Turns Draw 
        """
        Todo
        if(game_state["turnNumber"] > int(maxTurns) ):
            print(f"Draw Maximum Turns ({maxTurns}) Reached ") 
            exit(1)
        """ 
        
    

    """
    Parse the input string and modify it into board coordinates

    Args:
        - move: string representing a move "B2 B3"
    Returns:
        - (start, end)  tuple | the move to perform ((start_row, start_col),(end_row, end_col))
    """
    def parse_input(self, move):
        try:
            start, end = move.split()
            start = (5-int(start[1]), ord(start[0].upper()) - ord('A'))
            end = (5-int(end[1]), ord(end[0].upper()) - ord('A'))
            return (start, end)
        except:
            return None

    """
    Game loop

    Args:
        - None
    Returns:
        - None
    """
    def play(self):
        print("Welcome to Mini Chess! Enter moves as 'B2 B3'. Type 'exit' to quit.")

        while True:
            self.display_board(self.current_game_state)
            self.game_over(self.current_game_state)

            move = input(f"{self.current_game_state['turn'].capitalize()} to move: ")
            if move.lower() == 'exit':
                print("Game exited.")
                exit(1)

            move = self.parse_input(move)
            if not move or not self.is_valid_move(self.current_game_state, move):
                print("Invalid move. Try again.")
                continue

            self.make_move(self.current_game_state, move)

if __name__ == "__main__":
    game = MiniChess()
    game.play()