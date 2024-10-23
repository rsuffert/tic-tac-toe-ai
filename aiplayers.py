# aiplayers

from abc import ABC, abstractmethod
from enum import Enum
from typing import List
import random

class Player(Enum):
    '''Enum for player types.'''
    EMPTY =  '0'
    X     =  '1'
    O     = '-1'

class BaseAIPlayer(ABC):
    '''Abstract class for an AI player.'''
    @abstractmethod
    def play(board: List[List[Player]]) -> List[List[Player]]: pass

class RandomAIPlayer(BaseAIPlayer):
    '''AI player that makes random moves.'''
    def play(self, board: List[List[Player]]) -> List[List[Player]]:
        empty_cells = [(r, c) for r in range(3) for c in range(3) if board[r][c] == Player.EMPTY]
        if empty_cells:
            r, c = random.choice(empty_cells)
            board[r][c] = Player.O
        return board