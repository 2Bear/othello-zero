import math

import numpy as np

import board
import config


class FakeNode:
    def __init__(self):
        self.parent = None
        self.edge_N = np.zeros([config.all_moves_num], dtype=np.float)
        self.edge_W = np.zeros([config.all_moves_num], dtype=np.float)


class Node:
    def __init__(self, parent, move, player, board:board.Board=None):
        self.parent = parent
        self.expanded = False
        self.move = move
        self.player = player
        self.board = board or parent.board.make_move(parent.player, move)
        self.legal_moves = self.board.get_legal_moves(self.player)
        self.child_nodes = {}
        self.is_game_root = False
        self.is_search_root = False
        self.is_terminal = False
        self.pi = np.zeros([config.all_moves_num], dtype=np.float)
        self.edge_N = np.zeros([config.all_moves_num], dtype=np.float)
        self.edge_W = np.zeros([config.all_moves_num], dtype=np.float)
        self.edge_P = np.zeros([config.all_moves_num], dtype=np.float)

    @property
    def edge_Q(self):
        return self.edge_W / (self.edge_N + (self.edge_N == 0))

    @property
    def edge_U(self):
        return config.c_puct * self.edge_P * math.sqrt(max(1, self.self_N)) / (1 + self.edge_N)

    @property
    def edge_U_with_noise(self):
        noise = normalize_with_mask(np.random.dirichlet([config.noise_alpha] * config.all_moves_num), self.legal_moves)
        p_with_noise = self.edge_P * (1 - config.noise_weight) + noise * config.noise_weight
        return config.c_puct * p_with_noise * math.sqrt(max(1, self.self_N)) / (1 + self.edge_N)

    @property
    def edge_Q_plus_U(self):
        if self.is_search_root:
            return self.edge_Q * self.player + self.edge_U_with_noise + self.legal_moves * 1000
        else:
            return self.edge_Q * self.player + self.edge_U + self.legal_moves * 1000

    @property
    def self_N(self):
        return self.parent.edge_N[self.move]

    @self_N.setter
    def self_N(self, n):
        self.parent.edge_N[self.move] = n

    @property
    def self_W(self):
        return self.parent.edge_W[self.move]

    @self_W.setter
    def self_W(self, w):
        self.parent.edge_W[self.move] = w

    def to_features(self):
        features = np.zeros([config.history_num * 2 + 1, config.N, config.N], dtype=np.float)
        player = self.player
        current = self
        for i in range(config.history_num):
            own, enemy = current.board.get_own_and_enemy_array2d(player)
            features[2 * i] = own
            features[2 * i + 1] = enemy
            if current.is_game_root:
                break
            current = current.parent
        
        if player == config.black:
            features[config.history_num * 2] = np.ones([config.N, config.N], dtype=np.float)
        return np.moveaxis(features, 0, -1)

class MCTS_Batch:
    def __init__(self, nn):
        self.nn = nn

    def select(self, nodes):
        best_nodes_batch = [None] * len(nodes)
        for i, node in enumerate(nodes):
            current = node
            while current.expanded:
                best_edge = np.argmax(current.edge_Q_plus_U)
                if best_edge not in current.child_nodes:
                    current.child_nodes[best_edge] = Node(current, best_edge, -current.player)
                if current.is_terminal:
                    break
                if best_edge == config.pass_move and current.child_nodes[best_edge].legal_moves[config.pass_move] == 1:
                    current.is_terminal = True
                    break
                current = current.child_nodes[best_edge]
            best_nodes_batch[i] = current
        return best_nodes_batch

    def expand_and_evaluate(self, nodes_batch):
        features_batch = np.zeros([len(nodes_batch), config.N, config.N, config.history_num * 2 + 1], dtype=np.float)
        for i, node in enumerate(nodes_batch):
            node.expanded = True
            features_batch[i] = node.to_features()
        
        p_batch, v_batch = self.nn.f_batch(features_batch)

        for i, node in enumerate(nodes_batch):
            node.edge_P = normalize_with_mask(p_batch[i], node.legal_moves)
        
        return v_batch

    def backup(self, nodes_batch, v_batch):
        for i, node in enumerate(nodes_batch):
            current = node
            while True:
                current.self_N += 1
                current.self_W += v_batch[i]
                if current.is_search_root:
                    break
                current = current.parent

    def search(self, nodes):
        best_nodes_batch = self.select(nodes)
        v_batch = self.expand_and_evaluate(best_nodes_batch)
        self.backup(best_nodes_batch, v_batch)
    
    def alpha(self, nodes, temperature):
        for i in range(config.simulations_num):
            self.search(nodes)

        pi_batch = np.zeros([len(nodes), config.all_moves_num], dtype=np.float)
        for i, node in enumerate(nodes):
            n_with_temperature = node.edge_N**(1 / temperature)
            sum_n_with_temperature = np.sum(n_with_temperature)
            if sum_n_with_temperature == 0:
                node.pi = np.zeros([config.all_moves_num], dtype=np.float)
                node.pi[config.pass_move] = 1
            else:
                node.pi = n_with_temperature / sum_n_with_temperature
            pi_batch[i] = node.pi
        return pi_batch
    

def normalize_with_mask(x, mask):
    x_masked = np.multiply(x, mask)
    x_normalized = x_masked / np.sum(x_masked)
    return x_normalized
