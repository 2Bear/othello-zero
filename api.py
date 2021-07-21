import subprocess
from subprocess import PIPE, STDOUT, Popen

import config
from util import plane_2_line

import numpy as np


class HumanPlayer:
    def make_move(self, current_node):
        human_input = -1
        while True:
            human_input_str = input(">")
            if human_input_str == "pass":
                human_input = config.pass_move
            else:
                human_input = plane_2_line(human_input_str)

            if human_input is None or current_node.legal_moves[human_input] == 0:
                print("illegal.")
            else:
                return human_input
