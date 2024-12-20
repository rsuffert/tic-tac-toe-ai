# aiplayers

from abc import ABC, abstractmethod
from copy import deepcopy
from enum import Enum
from typing import List
import random
from tictactoeutils import Classifier
from neural_network import NeuralNetwork

class Player(Enum):
    '''Enum for player types.'''
    EMPTY =  '0'
    X     =  '1'
    O     = '-1'

class Difficulty(Enum):
    '''
    Enum for AI difficulty levels. The values of the enum indicate the percentage of the time
    the AI will make a random move.
    '''
    EASY   = 0.75
    MEDIUM = 0.5
    HARD   = 0.0

class BaseAIPlayer(ABC):
    '''Abstract class for an AI player.'''
    @abstractmethod
    def play(self, difficulty: Difficulty, board: List[List[Player]]) -> List[List[Player]]:
        '''
        Makes the AI player do the next move on the given board with the requested difficulty.
        difficulty.value represents the probability with which the AI player should play randomly.
        '''
        pass

class RandomAIPlayer(BaseAIPlayer):
    '''AI player that makes random moves.'''
    def play(self, difficulty: Difficulty, board: List[List[Player]]) -> List[List[Player]]:
        empty_cells = [(r, c) for r in range(3) for c in range(3) if board[r][c] == Player.EMPTY]
        if empty_cells:
            r, c = random.choice(empty_cells)
            board[r][c] = Player.O
        return board

class MinimaxAIPlayer(BaseAIPlayer):
    '''AI player that uses the minimax algorithm to make moves.'''
    def __init__(self):
        self._classifier = Classifier(Player.X.value, Player.O.value, Player.EMPTY.value)
        self._random_player = RandomAIPlayer()

    def play(self, difficulty: Difficulty, board: List[List[Player]]) -> List[List[Player]]:
        rnd: int = random.randint(1, 10)
        if rnd <= difficulty.value * 10:
            return self._random_player.play(difficulty, board)
        
        # use minimax to find the best move
        board_str: List[List[str]] = board_to_string_list(board)

        candidate_moves: List[List[List[str]]] = self._candidates(board_str, Player.O.value)
        if not candidate_moves: return board # no more moves to make
        new_board: List[List[str]] = max(candidate_moves, key=lambda b: self._minimax(b, False))

        return board_to_player_list(new_board)
    
    def _minimax(self, board: List[List[str]], maximizing=False, alpha=float("-inf"), beta=float("+inf")) -> float:
        '''Applies the Minimax algorithm on the given board, returning the best value.'''      
        if self._is_terminal(board):
            return self._evaluate(board)
        if maximizing:
            value = float("-inf")
            for child in self._candidates(board, Player.O.value):
                value = max(value, self._minimax(child, False))
                if value >= beta: break
                alpha = max(alpha, value)
        else:
            value = float("+inf")
            for child in self._candidates(board, Player.X.value):
                value = min(value, self._minimax(child, True))
                if value <= alpha: break
                beta = min(beta, value)
        return value

    def _candidates(self, board: List[List[str]], player: str) -> List[List[List[str]]]:
        '''Given a board and a player, returns all possibilities of boards after the player plays.'''
        candidate_moves = []

        for i in range(len(board)):
            for j in range(len(board)):
                if board[i][j] != Player.EMPTY.value:
                    continue

                candidate = deepcopy(board)
                candidate[i][j] = player

                candidate_moves.append(candidate)
        return candidate_moves

    def _evaluate(self, board: List[List[str]]) -> int:
        '''Returns the evaluation of the board.'''
        if self._classifier.is_winner(board_to_string(board), Player.O.value):
            return 1
        if self._classifier.is_winner(board_to_string(board), Player.X.value):
            return -1
        return 0

    def _is_terminal(self, board: List[List[str]]):
        '''Returns True if the game ended; False otherwise.'''
        board_str: str = board_to_string(board)
        return self._classifier.is_winner(board_str, Player.O.value) or self._classifier.is_winner(board_str, Player.X.value) or self._classifier.is_tie(board_str)

class NeuralNetworkAIPlayer(BaseAIPlayer):
    def __init__(self, model_file: str):
        self._network = NeuralNetwork.from_file(model_file)
        self._random_player = RandomAIPlayer()
    
    def play(self, difficulty: Difficulty, board: List[List[Player]]) -> List[List[Player]]:
        rnd: int = random.randint(1, 10)
        if rnd <= difficulty.value * 10:
            return self._random_player.play(difficulty, board)
        
        # use neural network to find the best move
        # need to revert the board because the network has been trained to play as X
        board_reverted: List[List[Player]] = reverse_board(board)

        board_floats: List[List[float]] = [[float(cell.value) for cell in row] for row in board_reverted]
        row, col = self._network.predict(board_floats)
        
        board_reverted[row][col] = Player.X

        return reverse_board(board_reverted)

def reverse_board(board: List[List[Player]]) -> List[List[Player]]:
    '''Reverses players of the board.'''
    return [
            [
                Player.X if cell == Player.O 
                else Player.O if cell == Player.X
                else Player.EMPTY 
                for cell in row
            ] 
            for row in board
        ]

def board_to_string(board: List[List[str]]) -> str:
    '''Converts the board to a string.'''
    return ','.join(cell for row in board for cell in row)

def board_to_string_list(board: List[List[Player]]) -> List[List[str]]:
    '''Converts the list of lists board into a list of lists of strings.'''
    result: List[List[str]] = []
    for row in board:
        result.append([cell.value for cell in row])
    return result

def board_to_player_list(board: List[List[str]]) -> List[List[Player]]:
    '''Converts the list of lists of strings into a list of lists of players.'''
    result: List[List[Player]] = []
    value_to_player= {
        Player.EMPTY.value: Player.EMPTY,
        Player.X.value:     Player.X,
        Player.O.value:     Player.O
    }
    for row in board:
        result.append([value_to_player[cell] for cell in row])
    return result