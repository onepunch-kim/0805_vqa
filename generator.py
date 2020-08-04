import h5py
import os
import argparse
import progressbar

from sort_of_clevr import *


def generator(config):
    img_size = config.img_size
    num_shapes = config.num_shapes
    dataset_size = config.dataset_size
    dir_name = config.dir_name

    # output files
    f = h5py.File(os.path.join(dir_name, 'data.hy'), 'w')

    # progress bar
    bar = progressbar.ProgressBar(maxval=100,
                                  widgets=[progressbar.Bar('=', '[', ']'), ' ',
                                           progressbar.Percentage()])
    bar.start()

    splits = {
        'train': int(dataset_size * 0.8),
        'val': int(dataset_size * 0.1),
        'test': int(dataset_size * 0.1)
    }
    step = 0
    for split, size in splits.items():
        image, question, answer = [], [], []
        split_g = f.create_group(split)
        count = 0
        while count < size:
            I, R = generate_sample(img_size, num_shapes)
            Q = generate_question(R, num_shapes)
            A = generate_answer(R, img_size, num_shapes)

            for j in range(num_shapes * NUM_Q):
                image.append(I)
                question.append(Q[j, :])
                answer.append(A[j, :])

                count += 1
                step += 1
                if step % (dataset_size / 100) == 0:
                    bar.update(step / (dataset_size / 100))
                if count >= size:
                    break

        image = np.stack(image)
        question = np.stack(question)
        answer = np.stack(answer)

        indices = np.random.permutation(count)
        image = image[indices]
        question = question[indices]
        answer = answer[indices]

        split_g.create_dataset('image', data=image)
        split_g.create_dataset('question', data=question)
        split_g.create_dataset('answer', data=answer)

    bar.finish()
    f.close()
    print('Dataset generated under {} with {} samples.'.format(dir_name, step))
    return


def check_path(path):
    if not os.path.exists(path):
        os.mkdir(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_shapes', type=int, default=4,
                        help="# of objects presented in each image")
    parser.add_argument('--dataset_size', type=int, default=200000)
    parser.add_argument('--img_size', type=int, default=32)
    args = parser.parse_args()

    basepath = './datasets/'
    check_path(basepath)

    # avoid a color shared by more than one objects
    assert 2 <= args.num_shapes <= 6

    path = os.path.join(basepath, f"SortOfCLEVR_{args.num_shapes}_{args.dataset_size}_{args.img_size}")
    check_path(path)
    args.dir_name = path

    generator(args)


if __name__ == '__main__':
    main()
