import argparse
import gc
import os
import random
import traceback
from multiprocessing import Pool, Process

import numpy as np
import tensorflow.compat.v1 as tf

import api
import board
import config
import gui
import net
import tree
from util import log, plane_2_line


class SelfPlayGame:
    def __init__(self, worker_id, batch_size=config.self_play_batch_size, echo_max=config.self_play_echo_max):
        self.version = 0
        self.echo = 0
        self.echo_max = echo_max
        self.worker_id = worker_id
        self.batch_size = batch_size
        self.fake_nodes = [None] * batch_size
        self.current_nodes = [None] * batch_size

    def start(self):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=config.self_play_woker_gpu_memory_fraction)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as session:
            saver = tf.train.Saver()
            self.restore(session, saver)
            nn = net.NN(session)
            mcts_batch = tree.MCTS_Batch(nn)
            while self.echo < self.echo_max:
                log("selfplay worker", self.worker_id, "version:", self.version, "echo:", self.echo, "session start.")
                self.play(mcts_batch)
                self.save()
                self.echo += 1
            log("selfplay worker", self.worker_id, "session end.")

    def play(self, mcts_batch):
        terminals_num = 0
        moves_num = 0
        for i in range(self.batch_size):
            self.fake_nodes[i] = tree.FakeNode()
            self.current_nodes[i] = tree.Node(self.fake_nodes[i], 0, config.black, board.Board())
            self.current_nodes[i].is_game_root = True
            self.current_nodes[i].is_search_root = True

        while terminals_num != self.batch_size:
            terminals_num = 0
            moves_num += 1
            gc.collect()
            pi_batch = mcts_batch.alpha(self.current_nodes, get_temperature(moves_num))
            for i in range(self.batch_size):
                if self.current_nodes[i].is_terminal is True:
                    terminals_num += 1
                else:
                    move = pick_move_probabilistically(pi_batch[i])
                    self.current_nodes[i] = make_move(self.current_nodes[i], move)

    def save(self):
        data = []
        for node in self.current_nodes:
            winner = 0
            black_stones_num = np.sum(node.board.black_array2d)
            white_stones_num = np.sum(node.board.white_array2d)
            if black_stones_num > white_stones_num:
                winner = 1
            elif black_stones_num < white_stones_num:
                winner = -1
            
            current = node
            while True:
                data.append(current.to_features())
                data.append(current.pi)
                data.append(winner)
                if current.is_game_root:
                    break
                current = current.parent
        np.savez_compressed(config.data_path + "{0:03d}_{1:03d}_{2:02d}{3:02d}".format(self.batch_size, self.version, self.worker_id, self.echo), data=data)

    def restore(self, session, saver):
        checkpoint_name = restore_from_last_checkpoint(session, saver)
        if checkpoint_name:
            self.version = int(checkpoint_name[1:].split('-')[0])
        
        last_echo = -1
        npz_file_names = get_npz_file_names()
        for file_name in npz_file_names:
            file_name_splited = file_name.split('_')
            if int(file_name_splited[-1][:2]) == self.worker_id:
                if int(file_name_splited[1]) < self.version:
                    os.rename(config.data_path + file_name, config.archives_path + file_name)
                else:
                    this_echo = int(file_name_splited[-1][2:4])
                    if this_echo > last_echo:
                        last_echo = this_echo
        
        self.echo = last_echo + 1


class Train:
    def __init__(self, batch_size=config.train_batch_size, echo_max=config.train_echo_max):
        self.version = 0
        self.state_data = np.zeros((0, config.N, config.N, config.history_num * 2 + 1), dtype=np.float)
        self.pi_data = np.zeros((0, config.all_moves_num), dtype=np.float)
        self.z_data = np.zeros((0, 1), dtype=np.float)
        self.batch_size = batch_size
        self.echo_max = echo_max
        self.data_len = self.load_data()
        self.batch_num = (self.data_len // self.batch_size) + 1
        self.global_step = 0
    
    def start(self):
        if self.data_len == 0:
            log("no data for training.")
            return
        with tf.Session() as session:
            saver = tf.train.Saver(max_to_keep=config.train_checkpoint_max_to_keep)
            self.restore(session, saver)
            nn = net.NN(session)
            log("training version:", self.version, "global step:", self.global_step, "session start.")
            with open(config.log_path + "loss_log.csv", "a+") as loss_log_file:
                for echo in range(self.echo_max):
                    for batch_index in range(self.batch_num):
                        self.global_step += 1
                        state_batch, pi_batch, z_batch = self.get_next_batch(batch_index, self.batch_size)
                        p_loss, v_loss = nn.train(state_batch, pi_batch, z_batch)
                        loss_log_file.write("{0},{1},{2}\n".format(self.global_step, p_loss, v_loss))
                    log("training echo:", echo, "global step:", self.global_step)
                    saver.save(session, config.checkpoint_path + "v{0:03d}".format(self.version), global_step=self.global_step)
            self.clear()
            log("training session end.")

    def load_data(self):
        npz_file_names = get_npz_file_names()
        if len(npz_file_names) == 0:
            self.data_len = 0
            return self.data_len

        self.version = int(npz_file_names[0].split('_')[1]) + 1

        for npz_file_name in npz_file_names:
            data = np.load(config.data_path + npz_file_name)['data']
            data_len = int(len(data) / 3)
            _state_data = np.zeros((data_len, config.N, config.N, config.history_num * 2 + 1), dtype=np.float)
            _pi_data = np.zeros((data_len, config.all_moves_num), dtype=np.float)
            _z_data = np.zeros((data_len, 1), dtype=np.float)
            for i in range(data_len):
                _state_data[i] = data[3 * i]
                _pi_data[i] = data[3 * i + 1]
                _z_data[i] = data[3 * i + 2]
            self.state_data = np.concatenate((self.state_data, _state_data))
            self.pi_data = np.concatenate((self.pi_data, _pi_data))
            self.z_data = np.concatenate((self.z_data, _z_data))
        
        self.data_len = len(self.state_data)
        return self.data_len
    
    def get_next_batch(self, index, size):
        start = index * size
        end = (index + 1) * size
        if start >= self.data_len:
            start = self.data_len - size
        if end > self.data_len:
            end = self.data_len
        return self.state_data[start:end], self.pi_data[start:end], self.z_data[start:end]

    def clear(self):
        npz_file_names = get_npz_file_names()
        for file_name in npz_file_names:
            os.rename(config.data_path + file_name, config.archives_path + file_name)
        log("all npz files archived.")

    def restore(self, session, saver):
        checkpoint_name = restore_from_last_checkpoint(session, saver)
        if checkpoint_name:
            self.global_step = int(checkpoint_name.split('-')[-1])


def pick_move_probabilistically(pi):
    r = random.random()
    s = 0
    for move in range(len(pi)):
        s += pi[move]
        if s >= r:
            return move
    return np.argmax(pi)


def pick_move_greedily(pi):
    return np.argmax(pi)


def get_temperature(moves_num):
    if moves_num <= 6:
        return 1
    else:
        return 0.95 ** (moves_num - 6)


def validate(move):
    if not (isinstance(move, int) or isinstance(move, np.int64)) or not (0 <= move < config.N ** 2 or move == config.pass_move):
        raise ValueError("move must be integer from [0, 63] or {}, got {}".format(config.pass_move, move))


def make_move(node, move):
    validate(move)
    if move not in node.child_nodes:
        node = tree.Node(node, move, -node.player)
    else:
        node = node.child_nodes[move]
    node.is_search_root = True
    node.parent.child_nodes.clear()
    node.parent.is_search_root = False
    return node


def print_winner(node):
    black_stones_num = np.sum(node.board.black_array2d)
    white_stones_num = np.sum(node.board.white_array2d)
    if black_stones_num > white_stones_num:
        print("black wins.")
    elif black_stones_num < white_stones_num:
        print("white wins.")
    else:
        print("draw.")


def restore_from_last_checkpoint(session, saver):
    checkpoint = tf.train.latest_checkpoint(config.checkpoint_path)
    if checkpoint:
        saver.restore(session, checkpoint)
        log("restored from last checkpoint.", checkpoint)
        return checkpoint.split('/')[-1]
    else:
        session.run(tf.global_variables_initializer())
        log("checkpoint not found.")
        return None


def get_npz_file_names():
    npz_file_names = []
    walk = os.walk(config.data_path)
    for dpath, _, fnames in walk:
        if dpath == config.data_path:
            for fname in fnames:
                if fname.split('.')[-1] == "npz":
                    npz_file_names.append(fname)
    return npz_file_names


def self_play_woker(worker_id):
    try:
        game = SelfPlayGame(worker_id)
        game.start()
    except Exception as ex:
        traceback.print_exc()


def train_woker():
    try:
        train = Train()
        train.start()
    except Exception as ex:
        traceback.print_exc()    


def learning_loop(self_play_wokers_num=config.self_play_wokers_num, echo_max=config.learning_loop_echo_max):
    for i in range(echo_max):
        pool = Pool(self_play_wokers_num)
        for i in range(self_play_wokers_num):
            pool.apply_async(self_play_woker, (i,))
        pool.close()
        pool.join()

        process = Process(target=train_woker)
        process.start()
        process.join()


def play_game(player):
    moves_num = 0
    mcts_batch = None
    current_node = tree.Node(tree.FakeNode(), 0, config.black, board.Board())
    current_node.is_game_root = True
    current_node.is_search_root = True

    def make_move_with_gui(current_node, move):
        current_node = make_move(current_node, move)
        gui.print_node(current_node)
        return current_node

    with tf.Session() as session:
        saver = tf.train.Saver()
        restore_from_last_checkpoint(session, saver)
        nn = net.NN(session)
        mcts_batch = tree.MCTS_Batch(nn)
        moves_num = 0
        while True:
            gc.collect()
            moves_num += 1

            # zero is thinking
            pi = mcts_batch.alpha([current_node], get_temperature(moves_num))[0]
            zero_move = pick_move_greedily(pi)
            current_node = make_move_with_gui(current_node, zero_move)
            if current_node.is_terminal:
                break

            # player is thinking
            mcts_batch.alpha([current_node], get_temperature(moves_num))[0]
            player_move = player.make_move(current_node)
            print("player move: {}".format(player_move))
            current_node = make_move_with_gui(current_node, player_move)
            if current_node.is_terminal:
                break
        # who is the winner
        print_winner(current_node)


def play_with_human():
    play_game(api.HumanPlayer())


parser = argparse.ArgumentParser()
parser.add_argument("-l", "--learn", help='start a learning loop from the latest model, or a new random model if there is no any model', action="store_true")
parser.add_argument("-p", "--play", help='play with you on the command line', action="store_true")
args = parser.parse_args()
if args.learn:
    learning_loop()
elif args.play:
    play_with_human()
else:
    learning_loop()
