# genetic.py

from typing import List, Tuple
from aiplayers import Player, Difficulty, MinimaxAIPlayer
from tictactoeutils import Classifier
from neural_network import NeuralNetwork

NN_TOPOLOGY: Tuple[int, ...] = (9, 9, 9)
classifier: Classifier = Classifier(Player.X.value, Player.O.value, Player.EMPTY.value)

def fitness(population: List[float], difficulty: Difficulty) -> float:
    """
    Calculates the fitness of a population.
    Args:
        population (List[float]): The weights for the neural network.
        difficulty (Difficulty): The difficulty level to which the population will be evaluated against.
    Returns:
        float: A number representing the fitness of the population. The higher the number, the better the population.
    """
    fitness: float = 0.0
    network: NeuralNetwork   = NeuralNetwork(NN_TOPOLOGY, population)
    minimax: MinimaxAIPlayer = MinimaxAIPlayer()

    board: List[List[Player]] = [[Player.EMPTY for _ in range(3)] for _ in range(3)]
    result: str = classifier.ONGOING
    
    # Simulate a game between the neural network and the minimax AI
    while result == classifier.ONGOING:
        # neural network plays as X
        row, col = network.predict(_to_float_board(board))
        if board[row][col] != Player.EMPTY:
            return float('-inf') # terminate with mininmal fitness if the network makes an invalid move
        fitness += 1.0 # grant one point for each valid move by the network
        board[row][col] = Player.X

        # minimax plays as O
        board = minimax.play(difficulty, board)

        # classify the board
        result = classifier.classify(_board_to_string(board)) # TODO: probably need to check the result right after each player plays

    # grant bonus or penalty based on the result of the match
    match result:
        case classifier.X_WON:
            fitness *= 1.25 # grant 25% bonus if the neural network wins
        case classifier.O_WON:
            fitness *= 0.75 # penalize 25% if minimax wins
        case classifier.TIE:
            fitness *= 1.0 # no bonus or penalty for a tie

    return fitness

def _board_to_string(board: List[List[Player]]) -> str:
    """Converts the board to a string."""
    return ','.join(str(cell.value) for row in board for cell in row)

def _to_float_board(board: List[List[Player]]) -> List[List[float]]:
    """Converts the board to a list of list of floats."""
    return [[float(cell.value) for cell in row] for row in board]