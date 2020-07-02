import subprocess
from subprocess import PIPE, STDOUT, Popen

import config
from util import line_2_plane, log, plane_2_line

import numpy as np


class EdaxPlayer:
    def __init__(self, level):
        edax_exec = config.edax_path + " -q -eval-file " + config.edax_eval_path \
            + " -book-file " + config.edax_book_path + " --level " + str(level) + " -book-randomness 10"
        self.edax = Popen(edax_exec, shell=True, stdout=PIPE, stdin=PIPE, stderr=STDOUT)
        self.read_stdout()

    def make_move(self, current_node):
        if current_node.move == config.pass_move:
            self.write_stdin("pass")
        else:
            self.write_stdin(line_2_plane(current_node.move))
        self.read_stdout()

        self.write_stdin("go")
        edax_move_plane = self.read_stdout().split("plays ")[-1][:2]
        if edax_move_plane == "PS":
            return config.pass_move
        else:
            return plane_2_line(edax_move_plane)

    def write_stdin(self, command):
        self.edax.stdin.write(str.encode(command + "\n"))
        self.edax.stdin.flush()

    def read_stdout(self):
        out = b''
        while True:
            next_b = self.edax.stdout.read(1)
            if next_b == b'>' and ((len(out) > 0 and out[-1] == 10) or len(out) == 0):
                break
            else:
                out += next_b
        return out.decode("utf-8")

    def close(self):
        self.edax.terminate()


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
