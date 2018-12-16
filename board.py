import numpy as np

import config


initial_black = np.uint64(0b00010000 << 24 | 0b00001000 << 32)
initial_white = np.uint64(0b00001000 << 24 | 0b00010000 << 32)


class Board:
    def __init__(self, black=initial_black, white=initial_white):
        self.black = black
        self.white = white
        self.black_array2d = bit_to_array(self.black, config.board_length).reshape((config.N, config.N))
        self.white_array2d = bit_to_array(self.white, config.board_length).reshape((config.N, config.N))

    def get_own_and_enemy(self, player):
        if player == config.black:
            return self.black, self.white
        else:
            return self.white, self.black

    def get_own_and_enemy_array2d(self, player):
        if player == config.black:
            return self.black_array2d, self.white_array2d
        else:
            return self.white_array2d, self.black_array2d

    def make_move(self, player, move):
        if move == config.pass_move:
            return Board(self.black, self.white)
        bit_move = np.uint64(0b1 << move)
        own, enemy = self.get_own_and_enemy(player)
        flipped_stones = get_flipped_stones_bit(bit_move, own, enemy)
        own |= flipped_stones | bit_move
        enemy &= ~flipped_stones
        if player == config.black:
            return Board(own, enemy)
        else:
            return Board(enemy, own)

    def get_legal_moves(self, player):
        own, enemy = self.get_own_and_enemy(player)
        legal_moves_without_pass = bit_to_array(get_legal_moves_bit(own, enemy), config.board_length)
        if np.sum(legal_moves_without_pass) == 0:
            return np.concatenate((legal_moves_without_pass, [1]))
        else:
            return np.concatenate((legal_moves_without_pass, [0]))


left_right_mask = np.uint64(0x7e7e7e7e7e7e7e7e)
top_bottom_mask = np.uint64(0x00ffffffffffff00)
corner_mask = left_right_mask & top_bottom_mask


def get_legal_moves_bit(own, enemy):
    legal_moves = np.uint64(0)
    legal_moves |= search_legal_moves_left(own, enemy, left_right_mask, np.uint64(1))
    legal_moves |= search_legal_moves_left(own, enemy, corner_mask, np.uint64(9))
    legal_moves |= search_legal_moves_left(own, enemy, top_bottom_mask, np.uint64(8))
    legal_moves |= search_legal_moves_left(own, enemy, corner_mask, np.uint64(7))
    legal_moves |= search_legal_moves_right(own, enemy, left_right_mask, np.uint64(1))
    legal_moves |= search_legal_moves_right(own, enemy, corner_mask, np.uint64(9))
    legal_moves |= search_legal_moves_right(own, enemy, top_bottom_mask, np.uint64(8))
    legal_moves |= search_legal_moves_right(own, enemy, corner_mask, np.uint64(7))
    legal_moves &= ~(own | enemy)
    return legal_moves


def search_legal_moves_left(own, enemy, mask, offset):
    return search_contiguous_stones_left(own, enemy, mask, offset) >> offset


def search_legal_moves_right(own, enemy, mask, offset):
    return search_contiguous_stones_right(own, enemy, mask, offset) << offset


def get_flipped_stones_bit(bit_move, own, enemy):
    flipped_stones = np.uint64(0)
    flipped_stones |= search_flipped_stones_left(bit_move, own, enemy, left_right_mask, np.uint64(1))
    flipped_stones |= search_flipped_stones_left(bit_move, own, enemy, corner_mask, np.uint64(9))
    flipped_stones |= search_flipped_stones_left(bit_move, own, enemy, top_bottom_mask, np.uint64(8))
    flipped_stones |= search_flipped_stones_left(bit_move, own, enemy, corner_mask, np.uint64(7))
    flipped_stones |= search_flipped_stones_right(bit_move, own, enemy, left_right_mask, np.uint64(1))
    flipped_stones |= search_flipped_stones_right(bit_move, own, enemy, corner_mask, np.uint64(9))
    flipped_stones |= search_flipped_stones_right(bit_move, own, enemy, top_bottom_mask, np.uint64(8))
    flipped_stones |= search_flipped_stones_right(bit_move, own, enemy, corner_mask, np.uint64(7))
    return flipped_stones


def search_flipped_stones_left(bit_move, own, enemy, mask, offset):
    flipped_stones = search_contiguous_stones_left(bit_move, enemy, mask, offset)
    if own & (flipped_stones >> offset) == np.uint64(0):
        return np.uint64(0)
    else:
        return flipped_stones


def search_flipped_stones_right(bit_move, own, enemy, mask, offset):
    flipped_stones = search_contiguous_stones_right(bit_move, enemy, mask, offset)
    if own & (flipped_stones << offset) == np.uint64(0):
        return np.uint64(0)
    else:
        return flipped_stones


def search_contiguous_stones_left(own, enemy, mask, offset):
    e = enemy & mask
    s = e & (own >> offset)
    s |= e & (s >> offset)
    s |= e & (s >> offset)
    s |= e & (s >> offset)
    s |= e & (s >> offset)
    s |= e & (s >> offset)
    return s


def search_contiguous_stones_right(own, enemy, mask, offset):
    e = enemy & mask
    s = e & (own << offset)
    s |= e & (s << offset)
    s |= e & (s << offset)
    s |= e & (s << offset)
    s |= e & (s << offset)
    s |= e & (s << offset)
    return s


def bit_count(bit):
    return bin(bit).count('1')


def bit_to_array(bit, size):
    return np.array(list(reversed((("0" * size) + bin(bit)[2:])[-size:])), dtype=np.uint8)
