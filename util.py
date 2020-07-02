import time
import config


def log(*messages):
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), ' '.join(str(a) for a in messages))


def plane_2_line(plane):
    if not plane or not (plane[0].isalpha() and plane[1].isdigit()):
        return None
    
    plane = plane.upper()
    col = ord(plane[0]) - 65
    row = int(plane[1]) - 1

    line = row * config.N + col

    if line > config.board_length - 1 or line < 0:
        return None
    return line


def line_2_plane(line):
    if line > config.board_length - 1 or line < 0:
        return None
    
    col = chr(line % config.N + 65)
    row = str(line // config.N + 1)

    return col + row
