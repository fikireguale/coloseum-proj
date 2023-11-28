# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time

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

    class Node:
        def __init__(self, state, parent=None):
            self.state = state
            self.parent = parent
            self.children = []
            self.wins = 0
            self.visits = 0
            self.untried_actions = self.get_untried_actions()
            self.chess_board = state[0]
            self.my_pos = state[1]
            self.adv_pos = state[2]
            self.max_step = state[3]
            self.board_size = len(self.chess_board)

        def get_untried_actions(self):
            # Moves (Up, Right, Down, Left)
            moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
            seen = ()
            bfs = [self.my_pos]
            allowed_moves = []
            # Pick steps random but allowable moves
            while bfs:
                r, c = bfs.pop(0)
                # Build a list of the moves we can make
                if math.dist(self.my_pos, [r, c]) + 1 > self.max_step:
                    break

                for d in range(0, 4):
                    tmp_pos = (r + moves[d][0], c + moves[d][1])
                    if not self.chess_board[r, c, d] and not self.adv_pos == tmp_pos and tmp_pos not in seen:
                        bfs.append(tmp_pos)
                        for i in range(0, 4):
                            if not self.chess_board[tmp_pos[0], tmp_pos[1], i]:
                                allowed_moves.append([tmp_pos, i])

            return allowed_moves

        def endgame(self):
            """
            Check if the game ends and compute the current score of the agents.

            Returns
            -------
            is_endgame : bool
                Whether the game ends.
            player_1_score : int
                The score of player 1.
            player_2_score : int
                The score of player 2.
            """
            # Union-Find
            moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
            father = dict()
            for r in range(self.board_size):
                for c in range(self.board_size):
                    father[(r, c)] = (r, c)

            def find(pos):
                if father[pos] != pos:
                    father[pos] = find(father[pos])
                return father[pos]

            def union(pos1, pos2):
                father[pos1] = pos2

            for r in range(self.board_size):
                for c in range(self.board_size):
                    for dir, move in enumerate(
                            moves[1:3]
                    ):  # Only check down and right
                        if self.chess_board[r, c, dir + 1]:
                            continue
                        pos_a = find((r, c))
                        pos_b = find((r + move[0], c + move[1]))
                        if pos_a != pos_b:
                            union(pos_a, pos_b)

            for r in range(self.board_size):
                for c in range(self.board_size):
                    find((r, c))
            p0_r = find(tuple(self.my_pos))
            p1_r = find(tuple(self.adv_pos))
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
            return True, p0_score, p1_score

        def rollout_policy(self):
            # Define the policy for the rollout phase, typically random
            pass

        def expand(self):
            # Expand the node by creating a new child node
            pass

        def best_child(self, c_param=1.41):
            # Select the best child using the UCB1 formula
            pass

        def update(self, result):
            # Update this node's data with the simulation result
            pass

    def ucb1(parent, child, c_param=1.41):
            """Calculate the UCB1 value for a child node."""
            return (child.wins / child.visits) + c_param * math.sqrt(math.log(parent.visits) / child.visits)

    def select_node(root):
        """Select a node in the tree to perform a rollout."""
        current_node = root
        while not current_node.is_terminal():
            if not current_node.untried_actions:
                current_node = current_node.best_child()
            else:
                return current_node.expand()
        return current_node

    def rollout(node):
        """Perform a rollout from the given node to the end of the game."""
        current_state = node.state
        while not current_state.is_terminal():
            current_state = current_state.rollout_policy()
        return current_state.get_result()

    def backpropagate(node, result):
        """Backpropagate the result of a rollout up the tree."""
        while node is not None:
            node.visits += 1
            node.wins += result
            node = node.parent

    def monte_carlo_tree_search(root, iterations=1000):
        """Perform the MCTS algorithm."""
        for _ in range(iterations):
            node = select_node(root)
            result = rollout(node)
            backpropagate(node, result)
        
        return max(root.children, key=lambda x: x.visits).state

    def step_simulation(self):
        """
        Take a step in the game world.
        Runs the agents' step function and update the game board accordingly.
        If the agents' step function raises an exception, the step will be replaced by a Random Walk.

        Returns
        -------
        results: tuple
            The results of the step containing (is_endgame, player_1_score, player_2_score)
        """
        cur_player, cur_pos, adv_pos = self.get_current_player()

        try:
            # Run the agents step function
            start_time = time()
            next_pos, dir = cur_player.step(
                deepcopy(self.chess_board),
                tuple(cur_pos),
                tuple(adv_pos),
                self.max_step,
            )
            time_taken = time() - start_time
            self.update_player_time(time_taken)

            next_pos = np.asarray(next_pos, dtype=cur_pos.dtype)
            if not self.check_boundary(next_pos):
                raise ValueError("End position {} is out of boundary".format(next_pos))
            if not 0 <= dir <= 3:
                raise ValueError(
                    "Barrier dir should reside in [0, 3], but your dir is {}".format(
                        dir
                    )
                )
            if not self.check_valid_step(cur_pos, next_pos, dir):
                raise ValueError(
                    "Not a valid step from {} to {} and put barrier at {}, with max steps = {}".format(
                        cur_pos, next_pos, dir, self.max_step
                    )
                )
        except BaseException as e:
            ex_type = type(e).__name__
            if (
                    "SystemExit" in ex_type and isinstance(cur_player, HumanAgent)
            ) or "KeyboardInterrupt" in ex_type:
                sys.exit(0)
            print(
                "An exception raised. The traceback is as follows:\n{}".format(
                    traceback.format_exc()
                )
            )
            print("Execute Random Walk!")
            next_pos, dir = self.random_walk(tuple(cur_pos), tuple(adv_pos))
            next_pos = np.asarray(next_pos, dtype=cur_pos.dtype)

        # Print out each step
        # print(self.turn, next_pos, dir)
        logger.info(
            f"Player {self.player_names[self.turn]} moves to {next_pos} facing {self.dir_names[dir]}. Time taken this turn (in seconds): {time_taken}"
        )
        if not self.turn:
            self.p0_pos = next_pos
        else:
            self.p1_pos = next_pos
        # Set the barrier to True
        r, c = next_pos
        self.set_barrier(r, c, dir)

        # Change turn
        self.turn = 1 - self.turn

        results = self.check_endgame()
        self.results_cache = results

        # Print out Chessboard for visualization
        if self.display_ui:
            self.render()
            if results[0]:
                # If game ends and displaying the ui, wait for user input
                click.echo("Press a button to exit the game.")
                try:
                    _ = click.getchar()
                except:
                    _ = input()
        return results

    def run(self, swap_players=False, board_size=None):
        self.reset(swap_players=swap_players, board_size=board_size)
        is_end, p0_score, p1_score = self.world.step()
        while not is_end:
            is_end, p0_score, p1_score = self.world.step_simulation()
        logger.info(
            f"Run finished. Player {PLAYER_1_NAME}: {p0_score}, Player {PLAYER_2_NAME}: {p1_score}"
        )
        return p0_score, p1_score, self.world.p0_time, self.world.p1_time

    def autoplay(self):
        """
        Run multiple simulations of the gameplay and aggregate win %
        """
        p1_win_count = 0
        p2_win_count = 0
        p1_times = []
        p2_times = []
        if self.args.display:
            logger.warning("Since running autoplay mode, display will be disabled")
        self.args.display = False
        with all_logging_disabled():
            for i in tqdm(range(self.args.autoplay_runs)):
                swap_players = i % 2 == 0
                board_size = np.random.randint(args.board_size_min, args.board_size_max)
                p0_score, p1_score, p0_time, p1_time = self.run(
                    swap_players=swap_players, board_size=board_size
                )
                if swap_players:
                    p0_score, p1_score, p0_time, p1_time = (
                        p1_score,
                        p0_score,
                        p1_time,
                        p0_time,
                    )
                if p0_score > p1_score:
                    p1_win_count += 1
                elif p0_score < p1_score:
                    p2_win_count += 1
                else:  # Tie
                    p1_win_count += 1
                    p2_win_count += 1
                p1_times.extend(p0_time)
                p2_times.extend(p1_time)

        logger.info(
            f"Player {PLAYER_1_NAME} win percentage: {p1_win_count / self.args.autoplay_runs}. Maxium turn time was {np.round(np.max(p1_times), 5)} seconds.")
        logger.info(
            f"Player {PLAYER_2_NAME} win percentage: {p2_win_count / self.args.autoplay_runs}. Maxium turn time was {np.round(np.max(p2_times), 5)} seconds.")

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
        start_time = time.time()
        time_taken = time.time() - start_time
        
        print("My AI's turn took ", time_taken, "seconds.")

        # dummy return
        return my_pos, self.dir_map["u"]
