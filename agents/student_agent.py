# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time
import logging
import math
import random


@register_agent("student_agent")
class StudentAgent(Agent):
    """
        A dummy class for your implementation. Feel free to use this class to
        add any helper functionalities needed for your agent.
        """

    def __init__(self):
        super(StudentAgent, self).__init__()
        self.name = "StudentAgent"
        self.dir_map = {
            "u": 0,
            "r": 1,
            "d": 2,
            "l": 3,
        }
        self.num_to_dir = {
            0: "u",
            1: "r",
            2: "d",
            3: "l"
        }
        self.opposites = {0: 2, 1: 3, 2: 0, 3: 1}
        self.moves = ((-1, 0), (0, 1), (1, 0), (0, -1))

    def get_possible_moves(self, my_pos, max_step, chess_board, adv_pos):
        # Moves (Up, Right, Down, Left)
        moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
        seen = set()
        bfs = [my_pos]
        x, y = my_pos
        seen.add((x, y))
        dist = 0
        counter = 1
        next_count = 0
        # board_size = len(chess_board)
        # distances = np.ones([board_size, board_size], dtype=int) * (2 * board_size + 2)
        # distances[x, y] = 0
        # print(distances)

        allowed_moves = []
        # Pick steps random but allowable moves
        while bfs:
            # print("stuck")
            r, c = bfs.pop(0)
            # Build a list of the moves we can make

            if counter == 0:
                dist += 1
                counter = next_count
                next_count = 0

            for i in range(0, 4):
                if not chess_board[r, c, i]:
                    allowed_moves.append(((r, c), i))
                    # distances[r, c] += 1

            for d in range(0, 4):
                if dist + 1 > max_step:
                    break
                tmp_pos = (r + moves[d][0], c + moves[d][1])
                a, b = tmp_pos

                if not chess_board[r, c, d] and not adv_pos == tmp_pos and tmp_pos not in seen:
                    bfs.append(tmp_pos)
                    next_count += 1
                    seen.add(tmp_pos)
                    # distances[a, b] = 0

            counter -= 1
        # print(distances)
        return allowed_moves

    def distance_to_all_cells(self, my_pos, chess_board, adv_pos):
        # Moves (Up, Right, Down, Left)
        moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
        seen = set()
        bfs = [my_pos]
        x, y = my_pos
        seen.add((x, y))
        dist = 0
        counter = 1
        next_count = 0
        board_size = len(chess_board)
        distances = np.ones([board_size, board_size], dtype=int) * (2 * board_size + 2)
        distances[x, y] = 0
        # print(distances)
        # Pick steps random but allowable moves
        while bfs:

            r, c = bfs.pop(0)
            # Build a list of the moves we can make
            if counter == 0:
                dist += 1
                counter = next_count
                next_count = 0

            for d in range(0, 4):
                tmp_pos = (r + moves[d][0], c + moves[d][1])
                # print(not chess_board[r, c, d] and not adv_pos == tmp_pos and tmp_pos not in seen)
                if not chess_board[r, c, d] and not adv_pos == tmp_pos and tmp_pos not in seen:
                    bfs.append(tmp_pos)
                    distances[tmp_pos] = dist + 1
                    next_count += 1
                    seen.add(tmp_pos)
            counter -= 1
        return distances

    def calc_heuristic(self, my_pos, chess_board, adv_pos):
        my_potential = self.distance_to_all_cells(my_pos, chess_board, adv_pos)
        opp_potential = self.distance_to_all_cells(adv_pos, chess_board, my_pos)
        # print("my pot:", my_potential)
        # print("opp pot:" , opp_potential)
        return np.sum(my_potential - opp_potential)

    def set_barrier(self, r, c, dir, chess_board):
        # Set the barrier to True
        chess_board[r, c, dir] = True
        # Set the opposite barrier to True
        move = self.moves[dir]
        chess_board[r + move[0], c + move[1], self.opposites[dir]] = True

    def undo(self, r, c, dir, chess_board):
        # Set the barrier to True
        chess_board[r, c, dir] = False
        # Set the opposite barrier to True
        move = self.moves[dir]
        chess_board[r + move[0], c + move[1], self.opposites[dir]] = False

    def best_move(self, my_pos, max_step, chess_board, adv_pos):

        moves = self.get_possible_moves(my_pos, max_step, chess_board, adv_pos)

        # game_enders = game_ending_moves()

        # if game_enders["win"]:
        #    return game_enders["win"][0]
        game_enders = {"neither": moves}
        best_moves = []
        heapq.heapify(best_moves)
        chess_board_copy = copy.deepcopy(chess_board)
        for m in game_enders["neither"]:
            coord, dir = m
            x, y = coord
            self.set_barrier(x, y, dir, chess_board_copy)
            heuristic = self.calc_heuristic(coord, chess_board_copy, adv_pos)
            heapq.heappush(best_moves, (heuristic, m))
            self.undo(x, y, dir, chess_board_copy)
        top = heapq.heappop(best_moves)
        # print(top)
        # while best_moves:
        # print("heap:", heapq.heappop(best_moves))
        return top[1]

    def step(self, chess_board, my_pos, adv_pos, max_step):
        """
        Implement the step function of your agent here.
        You can use the following variables to access the chess board:
        - chess_board: a numpy array of shape (x_max, y_max, 4)
        - my_pos: a tuple of (x, y)
        - adv_pos: a tuple of (x, y)
        - max_step: an integer

        You should return a tuple of ((x, y), dir),
        where (x, y) is the next position of your agent and dir is the direction of the wall
        you want to put on.

        Please check the sample implementation in agents/random_agent.py or agents/human_agent.py for more details.
        """

        # Some simple code to help you with timing. Consider checking
        # time_taken during your search and breaking with the best answer
        # so far when it nears 2 seconds.
        # print(self.get_untried_actions(my_pos, max_step, chess_board, adv_pos))
        start_time = time.time()
        time_taken = time.time() - start_time

        # print()
        my_pos, a = self.best_move(my_pos, max_step, chess_board, adv_pos)
        # print("RETURNED MOVES", my_pos, a)
        print("My AI's turn took ", time_taken, "seconds.")

        # dummy return
        return my_pos, a
