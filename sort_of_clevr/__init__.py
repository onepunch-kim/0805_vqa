import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

BG_COLOR = (120, 120, 120)
COLOR = [
    (0, 0, 255),
    (0, 255, 0),
    (255, 0, 0),
    (255, 212, 0),
    (255, 0, 255),
    (0, 255, 255),
    # add more colors here if needed
]

N_GRID = 4
NUM_COLOR = len(COLOR)

NUM_Q = 5
Q_DIM = 11
NUM_ANS = 10


class Representation:
    def __init__(self, x, y, color, shape):
        self.x = x
        self.y = y
        self.color = color
        self.shape = shape

    def print_graph(self):
        for i in range(len(self.x)):
            s = 'circle' if self.shape[i] else 'rectangle'
            print('{} {} at ({}, {})'.format(color2str(self.color[i]), s, self.x[i], self.y[i]))


def generate_sample(img_size, num_shape):
    # Generate I: [img_size, img_size, 3]
    img = Image.new('RGB', (img_size, img_size), color=BG_COLOR)
    drawer = ImageDraw.Draw(img)

    block_size = int(img_size * 0.9 / N_GRID)
    shape_size = int((img_size * 0.9 / N_GRID) * 0.7 / 2)
    scale = img_size / (block_size * N_GRID)

    idx_coor = np.arange(N_GRID*N_GRID)
    np.random.shuffle(idx_coor)
    idx_color_shape = np.arange(NUM_COLOR)
    np.random.shuffle(idx_color_shape)
    coin = np.random.rand(num_shape)

    X = []
    Y = []
    for i in range(num_shape):
        x = idx_coor[i] % N_GRID
        y = (N_GRID - np.floor(idx_coor[i] / N_GRID) - 1).astype(np.uint8)
        # sqaure terms are added to remove ambiguity of distance
        d_x = np.random.uniform(0.4, 0.6)
        d_y = np.random.uniform(0.4, 0.6)
        x0 = ((x + d_x) * block_size) * scale
        y0 = ((y + d_y) * block_size) * scale
        position = (x0 - shape_size, y0 - shape_size,
                    x0 + shape_size, y0 + shape_size)
        X.append(x0)
        Y.append(y0)
        if coin[i] < 0.5:
            drawer.ellipse(position, fill=COLOR[idx_color_shape[i]])
        else:
            drawer.rectangle(position, fill=COLOR[idx_color_shape[i]])

    # Generate its representation
    color = idx_color_shape[:num_shape]
    shape = coin < 0.5
    rep = Representation(np.stack(X).astype(np.int),
                         np.stack(Y).astype(np.int), color, shape)
    return np.array(img), rep


def generate_question(rep, num_shape):
    # Generate questions: [# of shape * # of Q, # of color + # of Q]
    Q = np.zeros((num_shape*NUM_Q, NUM_COLOR+NUM_Q), dtype=np.bool)
    for i in range(num_shape):
        v = np.zeros(NUM_COLOR)
        v[rep.color[i]] = True
        Q[i*NUM_Q:(i+1)*NUM_Q, :NUM_COLOR] = np.tile(v, (NUM_Q, 1))
        Q[i*NUM_Q:(i+1)*NUM_Q, NUM_COLOR:] = np.diag(np.ones(NUM_Q))
    return Q


def generate_answer(rep, img_size, num_shape):
    # Generate answers: [# of shape * # of Q, # of color + 4]
    # # of color + 4: [color 1, color 2, ... , circle, rectangle, yes, no]
    A = np.zeros((num_shape*NUM_Q, NUM_COLOR+4), dtype=np.bool)
    for i in range(num_shape):
        # Q1: circle or rectangle?
        if rep.shape[i]:
            A[i*NUM_Q, NUM_COLOR] = True
        else:
            A[i*NUM_Q, NUM_COLOR+1] = True

        # Q2: bottom?
        if rep.y[i] > int(img_size/2):
            A[i*NUM_Q+1, NUM_COLOR+2] = True
        else:
            A[i*NUM_Q+1, NUM_COLOR+3] = True

        # Q3: left?
        if rep.x[i] < int(img_size/2):
            A[i*NUM_Q+2, NUM_COLOR+2] = True
        else:
            A[i*NUM_Q+2, NUM_COLOR+3] = True

        distance = (rep.y - rep.y[i]) ** 2 + (rep.x - rep.x[i]) ** 2
        idx = distance.argsort()
        # Q4: the color of the nearest object
        min_idx = idx[1]
        A[i*NUM_Q+3, rep.color[min_idx]] = True
        # Q5: the color of the farthest object
        max_idx = idx[-1]
        A[i*NUM_Q+4, rep.color[max_idx]] = True
    return A


def color2str(code):
    return {
        0: 'blue',
        1: 'green',
        2: 'red',
        3: 'yellow',
        4: 'magenta',
        5: 'cyan',
    }[code]


def question2str(qv):
    def q_type(q):
        return {
            0: 'is it a circle or a rectangle?',
            1: 'is it closer to the bottom of the image?',
            2: 'is it on the left of the image?',
            3: 'the color of the nearest object?',
            4: 'the color of the farthest object?',
        }[q]
    color = np.argmax(qv[:NUM_COLOR])
    q_num = np.argmax(qv[NUM_COLOR:])
    return '[Query object color: {}] [Query: {}]'.format(color2str(color), q_type(q_num))


def answer2str(av, prefix=None):
    def a_type(a):
        return {
            0: 'blue',
            1: 'green',
            2: 'red',
            3: 'yellow',
            4: 'magenta',
            5: 'cyan',
            6: 'circle',
            7: 'rectangle',
            8: 'yes',
            9: 'no',
        }[np.argmax(a)]
    if not prefix:
        return '[Answer: {}]'.format(a_type(av))
    else:
        return '[{} Answer: {}]'.format(prefix, a_type(av))


def visualize_iqa(img, q, a):
    fig = plt.figure()
    plt.imshow(img)
    plt.title(question2str(q))
    plt.xlabel(answer2str(a))
    return fig


def draw_iqa(img, q, target_a, pred_a):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img)
    ax.set_title(question2str(q))
    ax.set_xlabel(answer2str(target_a) + answer2str(pred_a, 'Predicted'))
    return fig