# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import numpy as np
import time
import copy
import heapq


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

    def check_endgame(self, my_pos, chess_board, adv_pos):
        """
        Check if the game ends and compute the current score of the agents.

        Returns
        -------
        is_endgame : bool
            Whether the game ends.
        player_0_score : int
            The score of player 0, student_agent.
        player_1_score : int
            The score of player 1, student_agent's opponent.
        """
        board_size = len(chess_board)
        print("the board size is", board_size)

        my_coord, my_dir = my_pos
        my_pos = my_coord
        
        adv_coord, adv_dir = adv_pos
        adv_pos = adv_coord
        
        
        # Union-Find
        father = dict()
        for r in range(board_size):
            for c in range(board_size):
                father[(r, c)] = (r, c)

        def find(pos):
            if father[pos] != pos:
                father[pos] = find(father[pos])
            return father[pos]

        def union(pos1, pos2):
            father[pos1] = pos2

        for r in range(board_size):
            for c in range(board_size):
                for dir, move in enumerate(
                    self.moves[1:3]
                ):  # Only check down and right
                    if chess_board[r, c, dir + 1]:
                        continue
                    pos_a = find((r, c))
                    pos_b = find((r + move[0], c + move[1]))
                    if pos_a != pos_b:
                        union(pos_a, pos_b)

        for r in range(board_size):
            for c in range(board_size):
                find((r, c))
        p0_r = find(tuple(my_pos))
        p1_r = find(tuple(adv_pos))
        p0_score = list(father.values()).count(p0_r)
        p1_score = list(father.values()).count(p1_r)
        if p0_r == p1_r:
            return False, p0_score, p1_score
        player_win = None
        win_blocks = -1
        if p0_score > p1_score:
            player_win = 0
            win_blocks = p0_score
        elif p0_score < p1_score:
            player_win = 1
            win_blocks = p1_score
        else:
            player_win = -1  # Tie
        '''
        if player_win >= 0:
            # someone won
        else:
            # tie
        '''
        return True, p0_score, p1_score
    
    def game_ending_moves(self, valid_moves, my_pos, chess_board, adv_pos):
        """
        Sorts all_possible_moves by running check_endgame() on each one
        Input: current game
        Output: wins, losses, ties, neither (all arrays)
        """
        wins = []
        losses = []
        ties = []
        neither = []
        (adv_coord,dir) = adv_pos
        adv_pos = adv_coord

        for move in valid_moves:
            (my_coord,dir) = move
            move = my_coord
            # play move on board
            is_endgame, p0_score, p1_score = self.check_endgame(move, chess_board, adv_pos)
            # check/sort
            if not is_endgame:
                neither.append(move)
            else:
                win_blocks = -1
                if p0_score > p1_score: # student_agent wins
                    win_blocks = p0_score
                    self.insert(wins, win_blocks, move) # not putting the actual move in list
                elif p0_score < p1_score: # student_agent loses
                    win_blocks = p1_score
                    self.insert(losses, win_blocks, move, ascending=False)
                else:
                    ties.append(move)
            # undo move on board
            coord, dir = move
            r, c = coord
            self.undo(r, c, dir, chess_board)

        return wins, losses, ties, neither
    
    def insert(self, mlist, n, move, ascending = True):
        index = len(mlist)
        # Searching for the position
        if ascending:
            for i in range(len(mlist)):
                if mlist[i] > n:
                    index = i
                    break
        else:
            for i in range(len(mlist)):
                if mlist[i] < n:
                    index = i
                    break
        # Inserting n in the list
        if index == len(mlist):
            mlist = mlist[:index] + [n]
        else:
            mlist = mlist[:index] + [n] + mlist[index:]
        return mlist
    
    def best_move(self, my_pos, max_step, chess_board, adv_pos):

        moves = self.get_possible_moves(my_pos, max_step, chess_board, adv_pos)
        wins, losses, ties, neither = self.game_ending_moves(moves, my_pos, chess_board, adv_pos)
        game_enders = {"wins": wins, "losses": losses, "ties": ties, "neither": neither}
        my_move = None
        # if game_enders["win"]:
        #    return game_enders["win"][0]

        if wins:
            coord, dir = wins[0]
            return coord
        elif not neither:
            coord, dir = ties[0]
            return coord

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
        #print("My AI's turn took ", time_taken, "seconds.")

        # dummy return
        return my_pos, a
