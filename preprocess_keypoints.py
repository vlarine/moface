import os
from glob import glob


def get_all_landmarks():
    files_path = '../data/moface/openpose_keypoints/*.tsv'
    with open('../data/moface/op_keypoints.tsv', 'w') as wf:
        for filename in sorted(glob(files_path)):
            arr = []
            with open(filename) as f:
                for i, line in enumerate(f):
                    x, y, p = line.strip().split('\t')
                    arr.append(x)
                    arr.append(y)
                    arr.append(p)
            wf.write('\t'.join(arr) + '\n')

def main():
    get_all_landmarks()


if __name__ == '__main__':
    main()
