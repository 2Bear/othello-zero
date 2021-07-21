import os

# don't touch
N = 8
board_length = N * N
pass_move = N * N
all_moves_num = N * N + 1
black = 1
white = -1

# mcts config
c_puct = 1
simulations_num = 400
noise_alpha = 0.5
noise_weight = 0.25

# nn config
history_num = 4
residual_blocks_num = 9
momentum = 0.9
l2_weight = 1e-4
learning_rate = 1e-2

# learning config
self_play_wokers_num = 8
self_play_woker_gpu_memory_fraction = 0.04
self_play_batch_size = 128
self_play_echo_max = 4
train_batch_size = 128
train_echo_max = 100
train_checkpoint_max_to_keep = 1
learning_loop_echo_max = 3

# path config
checkpoint_path = "./checkpoint/"
data_path = "./data/"
archives_path = "./data/archives/"
log_path = "./log/"
if os.path.exists(data_path) is not True:
    os.mkdir(data_path)
if os.path.exists(archives_path) is not True:
    os.mkdir(archives_path)
if os.path.exists(log_path) is not True:
    os.mkdir(log_path)
