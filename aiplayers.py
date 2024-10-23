# aiplayers

from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Tuple
import random

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
    def play(self, difficulty: Difficulty, board: List[List[Player]]) -> List[List[Player]]: pass

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
    def play(self, difficulty: Difficulty, board: List[List[Player]]) -> List[List[Player]]:
        rnd: int = random.randint(1, 10)
        if rnd <= difficulty.value * 10:
            return RandomAIPlayer().play(difficulty, board)
        
        # TODO: Replace random play with Minimax play here
        return RandomAIPlayer.play(difficulty, board)