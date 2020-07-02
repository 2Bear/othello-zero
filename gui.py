import copy

import board
import config
from util import line_2_plane

string_board = [
    list("  A B C D E F G H"),
    list("1 ┌─┬─┬─┬─┬─┬─┬─┐"),
    list("2 ├─┼─┼─┼─┼─┼─┼─┤"),
    list("3 ├─┼─┼─┼─┼─┼─┼─┤"),
    list("4 ├─┼─┼─┼─┼─┼─┼─┤"),
    list("5 ├─┼─┼─┼─┼─┼─┼─┤"),
    list("6 ├─┼─┼─┼─┼─┼─┼─┤"),
    list("7 ├─┼─┼─┼─┼─┼─┼─┤"),
    list("8 └─┴─┴─┴─┴─┴─┴─┘")
]
black_sign = "●"
white_sign = "○"
legal_move_sign = "×"

def bit_to_sign(bit, sign, cur_string_board):
    bit_list = list(reversed((("0" * config.board_length) + bin(bit)[2:])[-config.board_length:]))
    for i, v in enumerate(bit_list):
        if v == "1":
            x = i % config.N
            y = i // config.N
            x += 2 + x
            y += 1
            cur_string_board[y][x] = sign
    
def array_to_sign(array, sign, cur_string_board):
    for i, v in enumerate(array):
        if i == config.pass_move:
            return
        
        if v == 1:
            x = i % config.N
            y = i // config.N
            x += 2 + x
            y += 1
            cur_string_board[y][x] = sign

def print_node(node):
    cur_string_board = copy.deepcopy(string_board)
    bit_to_sign(node.board.black, black_sign, cur_string_board)
    bit_to_sign(node.board.white, white_sign, cur_string_board)
    array_to_sign(node.legal_moves, legal_move_sign, cur_string_board)

    print("=================")

    for line in cur_string_board:
        print("".join(line))

    last_move = ""
    if node.move == config.pass_move:
        last_move = "pass"
    else:
        last_move = line_2_plane(node.move)

    player = ""
    opponent = ""
    if node.player == config.black:
        player = black_sign + " black"
        opponent = white_sign + " white"
    else:
        player = white_sign + " white"
        opponent = black_sign + " black"

    print("")

    if node.is_game_root is not True:
        print(opponent + " plays " + last_move + ".")
    if node.is_terminal is not True:
        print("it's " + player + " turn.")
