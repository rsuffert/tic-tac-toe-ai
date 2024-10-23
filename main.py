from abc import ABC, abstractmethod
from enum import Enum
from typing import List
import tictactoeutils
import random

class Player(Enum):
    '''Enum for player types.'''
    EMPTY =  '0'
    X     =  '1'
    O     = '-1'

class AIPlayer(ABC):
    '''Abstract class for an AI player.'''
    @abstractmethod
    def play(board: List[List[Player]]) -> List[List[Player]]: pass

# Tic-tac-toe board
board: List[List[Player]] = [[Player.EMPTY for _ in range(3)] for _ in range(3)]

def main(ai_player: AIPlayer) -> None:
    '''Entry point for the program.'''
    global board
    c = tictactoeutils.Classifier(Player.X.value, Player.O.value, Player.EMPTY.value)
    
    result: str = c.ONGOING
    while result == c.ONGOING:
        show_board()
        user_plays()
        board  = ai_player.play(board)
        result = c.classify(board_to_string())
        print("=================================================================\n")

    show_board()
    print("GAME OVER! Result: " + 
        result.replace(Player.X.value, Player.X.name).
               replace(Player.O.value, Player.O.name)
    )
        
def show_board() -> None:
    '''Displays the board.'''
    print('\t' + '\t'.join(map(str, range(len(board)))))
    for i, row in enumerate(board):
        print(i, end='\t')
        print('\t'.join('_' if cell == Player.EMPTY else cell.name for cell in row))

def user_plays() -> None:
    '''Handles the human player play.'''
    while True:
        try:
            user_input: str = input("Enter the row and column (separated by a space): ")
            r, c = map(int, user_input.split())
            if r not in range(3) or c not in range(3):
                print("\tInvalid input. Please enter two numbers between 0 and 2.")
                continue
            if board[r][c] != Player.EMPTY:
                print("\tInvalid input. The cell is already occupied.")
                continue
            board[r][c] = Player.X
            break
        except ValueError:
            print("\tInvalid input. Please enter two integers separated by a space.")

def board_to_string() -> str:
    '''Converts the board to a string.'''
    return ','.join(str(cell.value) for row in board for cell in row)

class RandomAIPlayer(AIPlayer):
    '''AI player that makes random moves.'''
    def play(self, board: List[List[Player]]) -> List[List[Player]]:
        empty_cells = [(r, c) for r in range(3) for c in range(3) if board[r][c] == Player.EMPTY]
        if empty_cells:
            r, c = random.choice(empty_cells)
            board[r][c] = Player.O
        return board

if __name__ == '__main__':
    main(RandomAIPlayer()) # TODO: add AI player implementation