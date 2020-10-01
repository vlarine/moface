import numpy as np
import pyopenpose as op
import cv2
from glob import glob


params = dict()

# OpenPose models folder
params["model_folder"] = '../data/models/'

# Do not use the face detector.
# Define constant face rectangles. 
params["face_detector"] = 2

params["face"] = True
params["body"] = 0

opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

# Face area in the video
face_rects = [op.Rectangle(400, 30, 512, 512),]


def op_process_file(filename):
    img = cv2.imread(filename)

    datum = op.Datum()
    datum.cvInputData = img
    datum.faceRectangles = face_rects

    opWrapper.emplaceAndPop([datum])
    keypoints = datum.faceKeypoints[0]

    return keypoints


def main():
    frames_dir = '../data/moface/frames'
    op_keypoints_dir = '../data/moface/openpose_keypoints'

    for filename in glob('{}/*.png'.format(frames_dir)):
        keypoints = op_process_file(filename)

        filename = filename.split('/')[-1].split('.')[0] + '.tsv'
        with open('{}/{}'.format(op_keypoints_dir, filename), 'w') as wf:
            for k in keypoints:
                wf.write('{}\t{}\t{}\n'.format(k[0], k[1], k[2]))



if __name__ == '__main__':
    main()
