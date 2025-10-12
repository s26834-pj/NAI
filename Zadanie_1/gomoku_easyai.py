#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Gomoku (Five in a Row)
Authors: Hanna Paczoska, Wiktor Rapacz

Description:
Implementation of the Gomoku game using the easyAI library. The game features one human player and one AI opponent

Usage:
    Run the program and enter moves in the format "x y" (e.g., "4 4").

References:
    easyAI documentation: https://pypi.org/project/easyAI/

Game rules:
    See the README.md file in this repository.
    https://github.com/s26834-pj/NAI/blob/main/Zadanie_1/README.md"""

from easyAI import TwoPlayerGame, AI_Player, Human_Player, Negamax
import numpy as np


class Gomoku(TwoPlayerGame):
    """
    A class that represents the Gomoku game (Five in a Row).

    Inherits from:
        easyAI.TwoPlayerGame

    Attributes:
        players (list): List containing the two players.
        size (int): Board size (default: 9x9).
        board (numpy.ndarray): 2D array representing the game board.
        current_player (int): Number of the player whose turn it is (1 or 2).
        last_move (tuple): Coordinates of the last move made.
    """

    def __init__(self, players, size=9):
        """
        Initialize the Gomoku game board and players.

        Args:
            players (list): [Human_Player(), AI_Player()]
            size (int, optional): Size of the square board (default is 9).
        """
        self.players = players
        self.size = size
        self.board = np.zeros((size, size), dtype=int)
        self.current_player = 1
        self.last_move = None

    # Methods required by the easyAI TwoPlayerGame interface

    def possible_moves(self):
        """
        Return a list of all legal moves available on the board.

        Returns:
            list[str]: Each move represented as "x y" string coordinates.
        """
        return [f"{x} {y}" for x in range(self.size) for y in range(self.size) if self.board[y, x] == 0]

    def make_move(self, move):
        """
        Execute a move on the board.

        Args:
            move (str): Coordinates of the move in the format "x y".
        """
        x, y = map(int, move.split())
        self.board[y, x] = self.current_player
        self.last_move = (x, y)

    def unmake_move(self, move):
        """
        Undo the last move (used internally by easyAI during search).

        Args:
            move (str): Coordinates of the move to undo.
        """
        x, y = map(int, move.split())
        self.board[y, x] = 0

    def is_over(self):
        """
        Determine whether the game is over.

        Returns:
            bool: True if there is a winner or the board is full, False otherwise.
        """
        return self.winner() is not None or not any(self.board.flatten() == 0)

    def show(self):
        """
        Display the current state of the board in the console.
        """
        print("\n   " + " ".join(f"{i:2}" for i in range(self.size)))
        for y in range(self.size):
            row = ""
            for x in range(self.size):
                cell = self.board[y, x]
                row += " . " if cell == 0 else (" X " if cell == 1 else " O ")
            print(f"{y:2} {row}")

    def winner(self):
        """
        Check if any player has achieved five in a row.

        Returns:
            int or None: The player number (1 or 2) if there is a winner, else None.
        """
        for y in range(self.size):
            for x in range(self.size):
                if self.board[y, x] != 0 and self._check_five(x, y):
                    return self.board[y, x]
        return None

    def _check_five(self, x, y):
        """
        Check if there are five consecutive marks starting from position (x, y).

        Args:
            x (int): X-coordinate.
            y (int): Y-coordinate.

        Returns:
            bool: True if there are five in a row, False otherwise.
        """
        player = self.board[y, x]
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        for dx, dy in directions:
            count = 1
            # Check forward direction
            nx, ny = x + dx, y + dy
            while 0 <= nx < self.size and 0 <= ny < self.size and self.board[ny, nx] == player:
                count += 1
                nx += dx
                ny += dy
            # Check backward direction
            nx, ny = x - dx, y - dy
            while 0 <= nx < self.size and 0 <= ny < self.size and self.board[ny, nx] == player:
                count += 1
                nx -= dx
                ny -= dy
            if count >= 5:
                return True
        return False

    def scoring(self):
        """
        Evaluate the game state.

        Returns:
            int: A positive score if the current player is winning,
                 negative if losing, or 0 if neutral.
        """
        if self.winner() == self.current_player:
            return 1000
        elif self.winner() == 3 - self.current_player:
            return -1000
        return 0


# Main program section

if __name__ == "__main__":
    print("=== GOMOKU (easyAI) ===")
    print("Enter your move in the format: x y (e.g., '4 4')")
    print("See RULES.md for detailed game rules.\n")

    # Create the AI engine using Negamax with depth = 2
    ai_algo = Negamax(2)

    # Player 1: Human, Player 2: AI
    game = Gomoku([Human_Player(), AI_Player(ai_algo)], size=9)

    # Start the game
    game.play()

    # Display final result
    if game.winner():
        print(f"\nWinner: {'X' if game.winner() == 1 else 'O'}")
    else:
        print("\nDraw.")
