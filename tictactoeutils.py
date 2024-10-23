# tictactoeutils

import regex
import pandas as pd
import itertools
import csv
from typing import Tuple, Union

class Classifier:
    '''
    This is a simple algorithmic classifier which is capable of determining the current state of a tic-tac-toe board.
    '''
    # if a player has all of the positions in any of the sub-lists then they are a winner
    _WIN_CONDITIONS = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Rows
        [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Columns
        [0, 4, 8], [2, 4, 6]              # Diagonals
    ]

    def __init__(self, x_symbol: str, o_symbol: str, blank_symbol: str) -> None:
        '''
        Params:
            x_symbol: the symbol that represents player x in the boards instances.
            o_symbol: the symbol that represents player o in the boards instances.
            blank_symbol: the symbol that represents blank cells in the boards instances.
        '''
        if not isinstance(x_symbol, str) or not isinstance(o_symbol, str) or not isinstance(blank_symbol, str):
            raise TypeError("Please make sure all of your parameters are of type 'str'")
        self._X = x_symbol
        self._O = o_symbol
        self._B = blank_symbol
        self.X_WON = f"{self._X} won"
        self.O_WON = f"{self._O} won"
        self.TIE = "Tie"
        self.ONGOING = "Ongoing game"

    def is_winner(self, board: str, player: str) -> bool:
        '''
        This checks whether or not a given player is a winner of a specific board instance.
        Notice that for an (invalid) board where both x and o are winners this will return True for both players.
        Thus, it is recommended to check first with the `is_tie` function, which will return True for this.
        '''
        if not isinstance(board, str) or not isinstance(player, str):
            raise TypeError("Please make sure all of your parameters are of type 'str'")
        if player != self._X and player != self._O:
            raise ValueError(f"Unknown player symbol: {player}. Should be either {self._X} or {self._O}")
        board_split = regex.split('\s*,\s*', board)
        return any(all(board_split[cell] == player for cell in combo) for combo in self._WIN_CONDITIONS)

    def is_tie(self, board: str) -> bool:
        '''
        This checks whether a given instance of a board is or will inevitably end tied.
        For an (invalid) board where both x and o are winners, this will return True.
        '''
        board_split = regex.split('\s*,\s*', board)
        if self._B in board_split:         return False
        if self.is_winner(board, self._X): return False
        if self.is_winner(board, self._O): return False
        return True
    
    def is_valid(self, board: Union[str, Tuple[str]]) -> bool:
        '''
        This tells whether or not the given board is valid - i.e., if it is a configuration that can be reached during a tic-tac-toe match.
        '''
        if not isinstance(board, str) and not (isinstance(board, tuple) and all(isinstance(e, str) for e in board)):
            raise TypeError("Please make sure 'board' is either a string or a tuple of strings representing a board")
        x_count = board.count(self._X)
        o_count = board.count(self._O)
        if not(x_count == o_count or x_count == 1+o_count): return False
        board_join = ','.join(board)
        x_won = self.is_winner(board_join, self._X)
        o_won = self.is_winner(board_join, self._O)
        if   x_won and o_won:                return False
        if   x_won and x_count != o_count+1: return False
        elif o_won and o_count != x_count:   return False
        return True
    
    def classify(self, board: str) -> str:
        '''
        This classifies the state of the given board, returning a string representing it.
        '''
        if not isinstance(board, str):
            raise TypeError("Please make sure all of your parameters are of type 'str'")
        if self.is_tie(board):             return self.TIE
        if self.is_winner(board, self._X): return self.X_WON
        if self.is_winner(board, self._O): return self.O_WON
        return self.ONGOING
    
    def classify_dataset(self, origin_dataset_path: str, destination_dataset_path: str) -> None:
        '''
        This classifies all boards of the origin dataset and writes the result to the detination dataset.
        '''
        if not isinstance(origin_dataset_path, str) or not isinstance(destination_dataset_path, str):
            raise TypeError("Please make sure all of your parameters are of type 'str'")
        df = pd.read_csv(origin_dataset_path)
        new_rows = []
        for _, row in df.iterrows():
            row_csv = ','.join(map(str, row.values))
            new_class = self.classify(row_csv)
            new_rows.append(row_csv + ',' + new_class)
        with open(destination_dataset_path, 'w', newline='\n') as f:
            header = ','.join(map(str, df.columns))+",Class"
            f.write(header + '\n')
            for r in new_rows:
                f.write(r + '\n')

class Generator:
    '''
    This generates all possible instances of a tic-tac-toe board.
    '''
    def __init__(self, x_symbol: str, o_symbol: str, blank_symbol: str) -> None:
        '''
        Params:
            x_symbol: the symbol that represents player x in the boards instances.
            o_symbol: the symbol that represents player o in the boards instances.
            blank_symbol: the symbol that represents blank cells in the boards instances.
        '''
        if not isinstance(x_symbol, str) or not isinstance(o_symbol, str) or not isinstance(blank_symbol, str):
            raise TypeError("Please make sure all of your parameters are of type 'str'")
        self._X = x_symbol
        self._O = o_symbol
        self._B = blank_symbol
        self._classifier = Classifier(self._X, self._O, self._B)
    
    def generate_all(self, destination_dataset_name: str, valid_only: bool = True) -> int:
        '''
        This generates all possible ways to arrange Xs, Os, and blanks on a tic-tac-toe board, writing them all
        (or only the valid ones, according to the `valid_only` parameter) to the given dataset path and returning
        the number of boards written to the dataset.
        '''
        if not isinstance(destination_dataset_name, str):
            raise TypeError("Parameter 'destination_dataset_name' must be a string")
        if not isinstance(valid_only, bool):
            raise TypeError("Parameter 'valid_only' must be a boolean value")
        cells = [self._X, self._O, self._B]
        boards = itertools.product(cells, repeat=9)
        with open(destination_dataset_name, 'w', newline='\n') as f:
            writer = csv.writer(f)
            header = ["1A", "1B", "1C", "2A", "2B", "2C", "3A", "3B", "3C"]
            writer.writerow(header)
            count = 0
            for b in boards:
                if not valid_only or self._classifier.is_valid(b):
                    writer.writerow(b)
                    count += 1
        return count