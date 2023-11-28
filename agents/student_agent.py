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

        def get_untried_actions(self):
            # Return a list of all possible actions from this state
            pass

        def is_terminal(self):
            # Check if the state is a terminal state
            pass

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
