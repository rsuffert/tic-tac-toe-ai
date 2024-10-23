from typing import List
from tictactoeutils import Classifier
from aiplayers import Player, Difficulty, BaseAIPlayer, MinimaxAIPlayer

# Tic-tac-toe board
board: List[List[Player]] = [[Player.EMPTY for _ in range(3)] for _ in range(3)]
difficulty: Difficulty = None

def main(ai_player: BaseAIPlayer) -> None:
    '''Entry point for the program.'''
    global board
    c = Classifier(Player.X.value, Player.O.value, Player.EMPTY.value)
    
    show_menu()

    result: str = c.ONGOING
    while result == c.ONGOING:
        #flush_terminal()
        show_board()
        user_plays()
        board  = ai_player.play(difficulty, board)
        result = c.classify(board_to_string())

    show_board()
    print("GAME OVER! Result: " + 
               result.replace(Player.O.value, Player.O.name).
                      replace(Player.X.value, Player.X.name)
    )

def show_menu() -> None:
    '''Displays the main (initial) menu.'''
    global difficulty
    print("=========== Welcome to the AI Tic Tac Toe game! ===========")
    print("You are playing as X. The AI is playing as O.")
    print("Please select the difficulty level you would like:")
    print("\t1. Easy")
    print("\t2. Medium")
    print("\t3. Hard")
    while True:
        try:
            level = int(input("Your choice: "))
            if level not in range(1, 4):
                print("\tInvalid input. Please enter a number between 1 and 3.")
                continue
            if level == 1:   difficulty = Difficulty.EASY
            elif level == 2: difficulty = Difficulty.MEDIUM
            else:            difficulty = Difficulty.HARD
            break
        except ValueError:
            print("\tInvalid input. Please enter an integer.")
    

def show_board() -> None:
    '''Displays the board.'''
    print("======================= BOARD =======================")
    print('\t' + '\t'.join(map(str, range(len(board)))))
    for i, row in enumerate(board):
        print(i, end='\t')
        print('\t'.join('_' if cell == Player.EMPTY else cell.name for cell in row))
    print()

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

def flush_terminal() -> None:
    '''Clears the terminal screen.'''
    import os
    os.system('cls||clear')

if __name__ == '__main__':
    main(MinimaxAIPlayer())