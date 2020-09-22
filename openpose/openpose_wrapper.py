# This program is free software: you can redistribute it and/or modify
# it undER the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import sys
import bpy
import bmesh
import cv2
import numpy as np
import os

from bpy.app.handlers import persistent
from dataclasses import dataclass
from math import cos, floor, sin, sqrt, tan
from mathutils import Euler, Matrix, Quaternion, Vector
from multiprocessing.dummy import Pool as ThreadPool
from threading import Thread, Lock
from time import time
from typing import Dict, List, Optional, Tuple

# Import OpenPose
try:
    sys.path.insert(0, '/usr/local/python')  # PyOpenPose is installed in this directory by default
    import pyopenpose as op
except ImportError as e:
    print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake of OpenPose and installed it?')
    raise e

OPENPOSE_ROOT = os.environ["OPENPOSE_ROOT"]
addonName = os.path.basename(os.path.dirname(__file__))


@dataclass
class Keypoint:
    x: float
    y: float
    z: float
    accuracy: float
    angle_vector: Vector = Vector((0.0, 0.0, 0.0))


class Camera:
    """
    Utility class embedding a camera, its parameters and buffers
    """
    def __init__(self,
                 path: str) -> None:
        self._path = path
        self._camera = cv2.VideoCapture()
        self._camera.open(path)
        self._shape: Tuple[int, int, int] = (0, 0, 0)
        self._bbox = [180, 120, 270, 270]
        self._bbox_new = self._bbox

        self._rectify_map_1: Optional[np.array] = None
        self._rectify_map_2: Optional[np.array] = None
        self._rectified_matrix: Optional[np.array] = None
        self._rectified = False

        self.raw_frame: Optional[np.array] = None
        self.frame: Optional[np.array] = None

        self.pose: Optional[np.array] = None
        self.dist_coeff = np.zeros(5)
        self.matrix = np.ndarray((3, 3))

        self.r_matrix = np.ndarray((3, 3))
        self.p_matrix = np.ndarray((3, 4))

    @property
    def bbox(self) -> List[int]:
        return self._bbox

    @bbox.setter
    def bbox(self, bbox: List[int]) -> None:
        if len(bbox) == len(self._bbox):
            self._bbox = bbox

    @property
    def bbox_new(self) -> List[int]:
        return self._bbox

    @bbox_new.setter
    def bbox_new(self, bbox: List[int]) -> None:
        if len(bbox) == len(self._bbox):
            self._bbox_new = bbox

    @property
    def rectified(self) -> bool:
        return self._rectified

    @property
    def shape(self) -> Tuple[int, int, int]:
        return self._shape

    @property
    def path(self) -> str:
        return self._path

    def grab(self) -> bool:
        while True:
            start = time()
            if not self._camera.grab():
                break
            stop = time()
            if stop - start > 1e-3:
                break

        ret, self.raw_frame = self._camera.retrieve()

        if not ret:
            self.raw_frame = None
            return False

        if self._rectified:
            self.raw_frame = cv2.remap(self.raw_frame,
                                       self._rectify_map_1,
                                       self._rectify_map_2,
                                       cv2.INTER_LINEAR)

        self._shape = self.raw_frame.shape
        return True

    def init_rectify(self) -> None:
        self._rectify_map_1, self._rectify_map_2 = cv2.initUndistortRectifyMap(self.matrix,
                                                                               self.dist_coeff,
                                                                               self.r_matrix,
                                                                               self.p_matrix,
                                                                               (self._shape[1], self._shape[0]),
                                                                               cv2.CV_16SC2,
                                                                               self._rectify_map_1,
                                                                               self._rectify_map_2)
        self._rectified = True


class OpenPoseWrapper:
    def __init__(self) -> None:
        self._cameras: List[Camera] = []
        self._image_buffer: Optional[bpy.types.Image] = None
        self._camera_paths: List[str] = ['/dev/video0', '/dev/video2']
        self._is_stereo = False
        self._is_stereo_calibrated = False

        # Options
        self._name_prefix = "OpenPose_"
        self._show_in_blender = False  # If True, show the captured frames inside a Blender image buffer
        self._show_debug_objects = False  # If True, show debug objects corresponding to each face feature
        self._show_debug_mesh = True  # If True, show a debug mesh of the detected face
        self._mirror_view = False  # If True, flip the camera frames horizontally before display
        self._detection_threshold = 0.50  # Threshold for pose detection
        self._only_face = False  # if True detects only face. No body. We suggest to detect body and face as it improves accuracy
        self._mask_frame = False  # crops the frame around the bounding box to reduce the noise during detection
        self._only_body = False  # detects only the body
        self._nb_parts = 70  # number of elements detected (70 for the face and 25 for the body)
        self._calibration_time_between_shots = 0.25  # seconds between shots minimum
        self._calibration_shots_number = 16  # shots used for calibration
        self._calibration_last_shot = time()
        self._calibration_obj_points: List[List[float]] = []
        self._calibration_img_points_left: List[List[float]] = []
        self._calibration_img_points_right: List[List[float]] = []

        self._last_update: Optional[float] = None

        self._op_last_update: Optional[float] = None
        self._op: Optional[op.OpenPose] = None
        self._op_continue = False
        self._op_thread: Optional[Thread] = None
        self._op_lock = Lock()

        self.update_from_data = False

    def initial_bbox(self) -> List[int]:
        proportion = 0.5  # proportion of the frame occupied by the bounding box
        resolution = self._cameras[0].shape
        x_origin = floor(resolution[1] / 2)
        y_origin = floor(resolution[0] / 2)
        delta = min(floor(proportion * resolution[1]), floor(proportion * resolution[0]))
        return [x_origin - floor(delta / 2), y_origin - floor(delta / 2), delta, delta]

    def apply_mask(self, frame: np.array, pose_bbox: List[int], padding: float = 1.0) -> np.array:
        rows, cols, channels = frame.shape
        mask = np.zeros((rows, cols), dtype=np.uint8)
        # the padding is only applied on the left and right of the bounding box to improve detection of the shoulders
        delta_col = pose_bbox[2]
        pad_col = floor(delta_col * padding / 2)
        delta_col += 2 * pad_col
        # the min and the max are taken to prevent unexpected behavior when pose_bbox contains negative values
        min_col = max(0, pose_bbox[0] - pad_col)
        max_col = min(min_col + delta_col, cols)
        min_row = max(0, pose_bbox[1])
        mask[min_row:rows, min_col:max_col] = 1

        return cv2.bitwise_and(frame, frame, mask=mask)

    # Returns the pose closest to the bounding box if its accuracy is sufficiently high
    # If no pose are detected with sufficient accuracy, an array of zeros is returned
    def filter_poses(self, poses: Optional[np.array]) -> np.array:
        array = np.zeros((self._nb_parts, 3))

        if poses is None or not poses.shape:  # special case where no poses are detected or a malformed array is provided (sometimes OpenPose returns a scalar when it does not detect anything)
            return array

        bbox_center = (self._cameras[0].bbox[0], self._cameras[0].bbox[1])  # following openCV notation (row, col)
        poses_average = np.average(poses, axis=1)
        error_measure = np.empty((0, 2))
        for average in poses_average:
            distance = np.linalg.norm(bbox_center - average[0:2])
            error_measure = np.append(error_measure, np.array([[distance, average[2]]]), axis=0)

        mask = error_measure[:, 1] > self._detection_threshold
        if (any(mask)):
            sub_index = np.argmin(error_measure[mask, 0])
            parent_index = np.arange(error_measure.shape[0])[mask][sub_index]
            array = poses[parent_index].reshape(-1, 3)
        return array

    def initialize_cameras(self) -> bool:
        """
        Initializes cameras, and check that all resolutions are identical
        """
        for path in self._camera_paths:
            camera = Camera(path=path)
            if camera.grab():
                self._cameras.append(camera)
            else:
                self._camera_paths.remove(path)

        if not self._cameras:
            return False

        for idx, camera in enumerate(self._cameras):
            if camera.shape != self._cameras[0].shape:
                return False
            # update the cameras bounding box according to their resolution
            self._cameras[idx].bbox = self.initial_bbox()
            self._cameras[idx].bbox_new = self._cameras[idx].bbox

        self._is_stereo = True if len(self._cameras) == 2 else False

        return True

    def start(self) -> bool:
        """
        Start the detection
        """
        self._name_prefix = bpy.context.preferences.addons[addonName].preferences.name_prefix
        self._show_in_blender = bpy.context.preferences.addons[addonName].preferences.show_in_blender
        self._show_debug_mesh = bpy.context.preferences.addons[addonName].preferences.show_debug_mesh
        self._mirror_view = bpy.context.preferences.addons[addonName].preferences.mirror_cameras
        self._calibration_shots_number = bpy.context.preferences.addons[addonName].preferences.calibration_shots
        self._detection_threshold = bpy.context.preferences.addons[addonName].preferences.detection_threshold
        self._only_face = bpy.context.preferences.addons[addonName].preferences.only_face
        self._mask_frame = bpy.context.preferences.addons[addonName].preferences.mask_frame
        self._only_body = True if bpy.context.preferences.addons[addonName].preferences.pose_choice == "BODY" else False

        if self._only_body:
            # when detecting only the body, the _detection_threshold must be set lower
            # since it is compared to the mean accuracy of the body parts.
            # If the body is not completely visible, this score will be low
            self._detection_threshold = 0.25
            self._nb_parts = 25

        return self.initialize_cameras()

    def stop(self) -> None:
        """
        Stop the detection.
        Currently, it does not clean OpenPose objects due to a bug in the library
        """
        self._op_continue = False
        if self._op_thread is not None and self._op_thread.is_alive():
            self._op_thread.join()
            self._op_thread = None

    def compute_bbox(self, pose: np.array, padding: float = 0.5):
        # We need to verify the accuray, because pose contains values set to zero by default
        # the minX and minY will be set incorrectly to zero
        sufficient_accuracy = pose[np.all(pose > self._detection_threshold, axis=1)]
        if sufficient_accuracy.size != 0:
            minX = np.min(sufficient_accuracy[:, 0])
            minY = np.min(sufficient_accuracy[:, 1])
        else:
            minX = 0
            minY = 0

        maxX = np.max(pose[:, 0])
        maxY = np.max(pose[:, 1])

        width = maxX - minX
        height = maxY - minY

        padX = width * padding / 2
        padY = height * padding / 2

        minX -= padX
        minY -= padY

        width += 2 * padX
        height += 2 * padY

        score = np.mean(pose[:, 2])

        # take the max because minX and minY can have negative values
        return score, [max(int(minX), 0), max(int(minY), 0), int(width), int(height)]

    def calibrate_stereo_pair(self) -> None:
        """
        Calibrate a stereo pair
        """
        pattern_size = (9, 6)
        pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
        pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
        pattern_points *= 17.0  # square size in mm

        img_points: List[List[float]] = []
        for camera in self._cameras:
            if not camera.grab():
                return
            frame = camera.raw_frame.copy()
            bw_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            found, corners = cv2.findChessboardCorners(bw_frame, pattern_size)
            if found:
                cv2.cornerSubPix(bw_frame, corners, (5, 5), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1))
                img_points.append(corners)
            cv2.drawChessboardCorners(frame, pattern_size, corners, found)
            camera.frame = frame

        text_frame = np.zeros((self._cameras[0].frame.shape), dtype=self._cameras[0].frame.dtype)
        cv2.putText(text_frame,
                    "Calibration completion: {}%".format(round(100.0 * len(self._calibration_obj_points) / self._calibration_shots_number)),
                    (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255))

        if self._mirror_view:
            text_frame = np.flip(text_frame, axis=1)

        # add the text to the frame
        self._cameras[0].frame = cv2.add(self._cameras[0].frame, text_frame)

        if len(img_points) != 2:
            return

        current_time = time()
        if current_time - self._calibration_last_shot < self._calibration_time_between_shots:
            return

        self._calibration_last_shot = current_time
        # self._cameras[0].frame = np.full(self._cameras[0].frame.shape, 1.0)
        # self._cameras[1].frame = np.full(self._cameras[0].frame.shape, 1.0)

        self._calibration_obj_points.append(pattern_points)
        self._calibration_img_points_left.append(img_points[0])
        self._calibration_img_points_right.append(img_points[1])

        if len(self._calibration_obj_points) < self._calibration_shots_number:
            return

        #
        # Calibrate cameras individually
        result = cv2.calibrateCamera(self._calibration_obj_points,
                                     self._calibration_img_points_left,
                                     (self._cameras[0].shape[1], self._cameras[0].shape[0]),
                                     self._cameras[0].matrix,
                                     self._cameras[0].dist_coeff)
        self._cameras[0].matrix = result[1]
        self._cameras[0].dist_coeff = result[2]

        result = cv2.calibrateCamera(self._calibration_obj_points,
                                     self._calibration_img_points_right,
                                     (self._cameras[1].shape[1], self._cameras[1].shape[0]),
                                     self._cameras[1].matrix,
                                     self._cameras[1].dist_coeff)
        self._cameras[1].matrix = result[1]
        self._cameras[1].dist_coeff = result[2]

        #
        # Calibrate the stereo pair
        r_matrix = np.ndarray((3, 3))
        t_matrix = np.ndarray((3))
        e_matrix = np.ndarray((3, 3))
        f_matrix = np.ndarray((3, 3))

        cv2.stereoCalibrate(self._calibration_obj_points,
                            self._calibration_img_points_left,
                            self._calibration_img_points_right,
                            self._cameras[0].matrix,
                            self._cameras[0].dist_coeff,
                            self._cameras[1].matrix,
                            self._cameras[1].dist_coeff,
                            (self._cameras[0].shape[1], self._cameras[0].shape[0]),
                            r_matrix,
                            t_matrix,
                            e_matrix,
                            f_matrix,
                            cv2.CALIB_FIX_ASPECT_RATIO + cv2.CALIB_FIX_INTRINSIC)

        q_matrix = np.ndarray((4, 4))
        cv2.stereoRectify(self._cameras[0].matrix,
                          self._cameras[0].dist_coeff,
                          self._cameras[1].matrix,
                          self._cameras[1].dist_coeff,
                          (self._cameras[0].shape[1], self._cameras[0].shape[0]),
                          r_matrix,
                          t_matrix,
                          self._cameras[0].r_matrix,
                          self._cameras[1].r_matrix,
                          self._cameras[0].p_matrix,
                          self._cameras[1].p_matrix,
                          q_matrix,
                          cv2.CALIB_ZERO_DISPARITY)

        self._cameras[0].init_rectify()
        self._cameras[1].init_rectify()

        self._is_stereo_calibrated = True

        return

    def store_camera_calibration(self) -> None:
        if not self._is_stereo_calibrated:
            return

        camera_1_matrix = self._cameras[0].matrix.reshape([-1])
        camera_1_dist = self._cameras[0].dist_coeff.reshape([-1])
        camera_1_intrinsics = np.append(camera_1_matrix, camera_1_dist)
        bpy.context.preferences.addons[addonName].preferences.calibration_camera_1_intrinsics = camera_1_intrinsics

        camera_1_r = self._cameras[0].r_matrix.reshape([-1])
        camera_1_p = self._cameras[0].p_matrix.reshape([-1])
        camera_1_extrinsics = np.append(camera_1_r, camera_1_p)
        bpy.context.preferences.addons[addonName].preferences.calibration_camera_1_extrinsics = camera_1_extrinsics

        camera_2_matrix = self._cameras[1].matrix.reshape([-1])
        camera_2_dist = self._cameras[1].dist_coeff.reshape([-1])
        camera_2_intrinsics = np.append(camera_2_matrix, camera_2_dist)
        bpy.context.preferences.addons[addonName].preferences.calibration_camera_2_intrinsics = camera_2_intrinsics

        camera_2_r = self._cameras[1].r_matrix.reshape([-1])
        camera_2_p = self._cameras[1].p_matrix.reshape([-1])
        camera_2_extrinsics = np.append(camera_2_r, camera_2_p)
        bpy.context.preferences.addons[addonName].preferences.calibration_camera_2_extrinsics = camera_2_extrinsics

        bpy.context.preferences.addons[addonName].preferences.calibration_set = True

    def load_camera_calibration(self) -> bool:
        """
        Load camera calibration
        Returns True if it was successfull
        """
        if not self._is_stereo:
            return False

        if not bpy.context.preferences.addons[addonName].preferences.calibration_set:
            return False

        camera_1_intrinsics = np.array(bpy.context.preferences.addons[addonName].preferences.calibration_camera_1_intrinsics)
        self._cameras[0].matrix = camera_1_intrinsics[0:9].reshape((3, 3))
        self._cameras[0].dist_coeff = camera_1_intrinsics[9:14]
        camera_1_extrinsics = np.array(bpy.context.preferences.addons[addonName].preferences.calibration_camera_1_extrinsics)
        self._cameras[0].r_matrix = camera_1_extrinsics[0:9].reshape((3, 3))
        self._cameras[0].p_matrix = camera_1_extrinsics[9:21].reshape((3, 4))

        camera_2_intrinsics = np.array(bpy.context.preferences.addons[addonName].preferences.calibration_camera_2_intrinsics)
        self._cameras[1].matrix = camera_2_intrinsics[0:9].reshape((3, 3))
        self._cameras[1].dist_coeff = camera_2_intrinsics[9:14]
        camera_2_extrinsics = np.array(bpy.context.preferences.addons[addonName].preferences.calibration_camera_2_extrinsics)
        self._cameras[1].r_matrix = camera_2_extrinsics[0:9].reshape((3, 3))
        self._cameras[1].p_matrix = camera_2_extrinsics[9:21].reshape((3, 4))

        self._cameras[0].init_rectify()
        self._cameras[1].init_rectify()

        self._is_stereo_calibrated = True


    def update_pose_estimation(self) -> None:
        """
        Update the pose estimation from the readings from cameras
        """
        def grab_camera(camera: Camera) -> None:
            camera.grab()

        pool = ThreadPool(len(self._cameras))

        if not self._op:
            # for more parameters see: https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/include/openpose/flags.hpp
            params = dict()
            params["logging_level"] = 255  # no output; Important to be set at 255. Otherwise it interacts with Blender logging system
            params["model_folder"] = OPENPOSE_ROOT + os.sep + "models" + os.sep
            params["output_resolution"] = "-1x-1"  # forces the output to have the same resolution as the input image
            params["number_people_max"] = 1  # -1 for no limit
            params["model_pose"] = "BODY_25"
            params["face"] = 1
            params["body"] = 1
            if self._only_body:
                params["face"] = 0
            if self._only_face:
                params["face_detector"] = 2
                params["body"] = 0

            self._op = op.WrapperPython()
            self._op.configure(params)
            self._op.start()

        while self._op_continue:
            pool.map(grab_camera, self._cameras)

            self._op_lock.acquire()
            for camera in self._cameras:
                bbox = camera.bbox
                bbox_new = camera.bbox_new

                frame = camera.raw_frame.copy()

                if self._mask_frame:
                    detection_frame = self.apply_mask(frame, bbox)
                else:
                    detection_frame = frame

                datum = op.Datum()
                datum.cvInputData = detection_frame
                if self._only_face:
                    # if no prior body detection, we need to provide a rectangle
                    datum.faceRectangles = [op.Rectangle(bbox[0], bbox[1], bbox[2], bbox[3])]

                self._op.emplaceAndPop([datum])
                # Filters the multiple bodies or faces detected to take the one
                # closest to the bounding box
                if not self._only_body:
                    poses = datum.faceKeypoints  # shape (N, keypoints, 3) where N is the number of detected faces, it is limited by the parameter number_people_max of the variable params

                else:  # only body
                    poses = datum.poseKeypoints  # shape (N, keypoints, 3)

                camera.pose = self.filter_poses(poses)
                frame = datum.cvOutputData

                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), [50, 155, 50], 2)
                cv2.rectangle(frame, (bbox_new[0], bbox_new[1]), (bbox_new[0] + bbox_new[2], bbox_new[1] + bbox_new[3]), [250, 55, 50], 1)
                score, bbox_new = self.compute_bbox(camera.pose, padding=0.4)

                if score > self._detection_threshold:
                    bbox = bbox_new

                camera.bbox = bbox
                camera.bbox_new = bbox_new

                camera.frame = frame

            self._op_last_update = time()
            self._op_lock.release()

    def display_cameras(self) -> None:
        """
        Display the result of pose estimation
        """
        if self._is_stereo:
            camera_1_frame = self._cameras[0].frame
            camera_2_frame = self._cameras[1].frame
            frame = np.concatenate((camera_1_frame, camera_2_frame), axis=0)
        else:
            frame = self._cameras[0].frame

        # Flip the image horizontally to act as a mirror view
        if self._mirror_view:
            frame = np.flip(frame, axis=1)

        image_name = self._name_prefix + "Camera"

        if self._show_in_blender:
            # We limit the preview size to 320x240, to keep workable performances
            if frame.shape[0] > 240 or frame.shape[1] > 320:
                tmpFrame = frame
                frame = cv2.resize(tmpFrame, (320, int(320 * frame.shape[0] / frame.shape[1])))

            if not self._image_buffer:
                if image_name not in bpy.data.images.keys():
                    bpy.ops.image.new(name=image_name, width=frame.shape[1], height=frame.shape[0], alpha=True)
                self._image_buffer = bpy.data.images[image_name]

            if self._image_buffer.size[0] != frame.shape[1] or self._image_buffer.size[1] != frame.shape[0]:
                bpy.ops.images.remove(self._image_buffer)
                bpy.ops.image.new(name=image_name, width=frame.shape[1], height=frame.shape[0], alpha=True)
                self._image_buffer = bpy.data.images[image_name]

            frame = np.flip(np.flip(frame, axis=2), axis=0)
            frame = np.concatenate((frame / 255.0, np.ones((frame.shape[0], frame.shape[1], 1))), axis=2)
            frame = frame.reshape((frame.shape[0] * frame.shape[1] * frame.shape[2]))
            self._image_buffer.pixels = frame
        else:
            cv2.imshow(image_name, frame)
            cv2.waitKey(1)

    def triangulate_pose_points(self) -> np.array:
        #scale_factor = 100.0
        scale_factor = 25.0
        pose_left = np.array(self._cameras[0].pose, np.float32)[:, 0:2].transpose()
        pose_right = np.array(self._cameras[1].pose, np.float32)[:, 0:2].transpose()

        pose_3D = np.ndarray((4, pose_left.shape[0]), dtype=np.float32)

        pose_3D = cv2.triangulatePoints(self._cameras[0].p_matrix,
                                        self._cameras[1].p_matrix,
                                        pose_left,
                                        pose_right,
                                        pose_3D)

        pose_3D = pose_3D.transpose()
        pose_3D = np.array([[point[0], point[1], point[2]] / (point[3] * (1000.0 / scale_factor)) for point in pose_3D])
        rotation_matrix = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])
        pose_3D = np.inner(rotation_matrix, pose_3D).transpose()
        return pose_3D

    def display_face_3D_points(self, face: np.array) -> None:
        debug_object_name = self._name_prefix + "Debug"
        debug_point_prefix = self._name_prefix + "Point_"

        if debug_object_name not in bpy.data.objects.keys():
            bpy.ops.object.empty_add(type='PLAIN_AXES', location=(0.0, 0.0, 0.0))
            bpy.context.active_object.name = debug_object_name

        debug_object = bpy.data.objects[debug_object_name]
        debug_point_names = [child.name for child in debug_object.children]

        face_point_index = 0
        for face_point in face:
            face_point_name = debug_point_prefix + str(face_point_index)
            if face_point_name not in debug_point_names:
                bpy.ops.mesh.primitive_cube_add(radius=0.01)
                bpy.context.active_object.name = face_point_name

                bpy.ops.object.select_all(action='DESELECT')
                bpy.data.objects[face_point_name].select_set(state=True)
                bpy.data.objects[debug_object_name].select_set(state=True)
                bpy.context.view_layer.objects.active = bpy.data.objects[debug_object_name]
                bpy.ops.object.parent_set()
                bpy.ops.object.select_all(action='DESELECT')

            face_point_object = bpy.data.objects[face_point_name]
            face_point_object.location = Vector(face_point[0:3])
            face_point_index += 1

    def display_raw_mesh(self, face: np.array) -> None:
        mesh_name = self._name_prefix + "mesh"
        object_name = self._name_prefix + "object"

        if face is None or face.size == 0:
            return

        if bpy.context.active_object is not None and object_name == bpy.context.active_object.name and bpy.context.active_object.select_get() is True:
            return

        if object_name not in bpy.data.objects.keys():
            # Create mesh and object
            mesh = bpy.data.meshes.new(mesh_name)
            obj = bpy.data.objects.new(object_name, mesh)

            # Generate the vertices and faces for the mesh
            bpy.context.scene.collection.objects.link(obj)
            bpy.context.view_layer.objects.active = obj
            obj.select_set(state=True)
            mesh = bpy.data.objects[object_name].data
            bm = bmesh.new()

            for point in face:
                bm.verts.new(point)
            bm.verts.ensure_lookup_table()

            # Faces defined based on the indices from the OpenPose documentation:
            # https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/output.md
            faces = [(0, 1, 36), (1, 41, 36), (1, 40, 41), (1, 28, 40), (1, 29, 28), (1, 30, 29), (1, 2, 30),
                     (2, 31, 30), (2, 3, 31), (3, 49, 31), (3, 4, 49), (4, 48, 49), (4, 5, 48), (5, 59, 48),
                     (5, 6, 59), (6, 58, 59), (6, 7, 58), (7, 57, 58), (7, 8, 57), (8, 9, 57), (9, 56, 57),
                     (9, 10, 56), (10, 55, 56), (10, 11, 55), (11, 54, 55), (11, 12, 54), (12, 53, 54), (12, 13, 53),
                     (13, 35, 53), (13, 14, 35), (14, 30, 35), (14, 15, 30), (15, 29, 30), (15, 28, 29), (15, 47, 28),
                     (15, 46, 47), (15, 16, 46), (16, 45, 46), (16, 26, 45), (26, 44, 45), (26, 25, 44), (25, 24, 44),
                     (24, 23, 44), (23, 43, 44), (23, 22, 43), (22, 27, 43), (0, 36, 17), (17, 36, 37), (17, 37, 18),
                     (18, 37, 19), (19, 37, 20), (20, 37, 38), (20, 38, 21), (21, 38, 27), (38, 39, 27), (39, 28, 27),
                     (39, 40, 28), (43, 27, 42), (27, 28, 42), (28, 47, 42), (30, 31, 32), (30, 32, 33), (30, 33, 34),
                     (30, 34, 35), (31, 49, 50), (31, 50, 32), (32, 50, 33), (33, 50, 51), (33, 51, 52), (33, 52, 34),
                     (34, 52, 35), (35, 52, 53), (48, 60, 49), (49, 60, 61), (49, 61, 50), (50, 61, 62), (50, 62, 51),
                     (51, 62, 52), (52, 62, 63), (52, 63, 53), (53, 63, 64), (53, 64, 54), (54, 64, 55), (55, 64, 65),
                     (55, 65, 56), (56, 65, 66), (56, 66, 57), (57, 66, 58), (58, 66, 67), (58, 67, 59), (59, 67, 60),
                     (59, 60, 48)]

            for face in faces:
                bm.faces.new((bm.verts[face[0]], bm.verts[face[1]], bm.verts[face[2]]))

            bm.to_mesh(mesh)
            bm.free()
        else:
            # Update the mesh
            obj = bpy.data.objects[object_name]
            bm = bmesh.new()
            bm.from_mesh(obj.data)
            bm.verts.ensure_lookup_table()

            for index in range(len(face)):
                bm.verts[index].co = Vector(face[index])

            bm.to_mesh(obj.data)
            bm.free()

    def init_data(self, scene):
        # TODO: For testing purpose
        # move bones in function of OpenPose keypoints
        # Here we assume that we only have that one armature
        self.active_armature = next((obj for obj in scene.objects if obj.type == "ARMATURE" and obj.openpose_active), None)

    @persistent
    def update(self, scene, *args, **kwargs) -> None:
        pose_3D: Optional[np.array] = None

        if self._is_stereo and not self._is_stereo_calibrated:
            if not self.load_camera_calibration():
                self.calibrate_stereo_pair()
                self.display_cameras()
        else:
            if not self.update_from_data:
                self.store_camera_calibration()

                if self._op_thread is None:
                    self._op_continue = True
                    self._op_thread = Thread(target=self.update_pose_estimation)
                    self._op_thread.daemon = True
                    self._op_thread.start()

                self._op_lock.acquire()
                if self._op_last_update is None or self._last_update == self._op_last_update:
                    self._op_lock.release()
                    return {'FINISHED'}
                self._last_update = self._op_last_update

                if not self.update_from_data:
                    self.display_cameras()

                # verify that the pose is detected for each camera
                for camera in self._cameras:
                    if np.mean(camera.pose[:, 2]) < self._detection_threshold:
                        self._op_lock.release()
                        return {'FINISHED'}

                if self._is_stereo:
                    pose_3D = self.triangulate_pose_points()
                    self._op_lock.release()
                    if self._show_debug_objects:
                        self.display_face_3D_points(pose_3D)
                    if self._show_debug_mesh:
                        self.display_raw_mesh(pose_3D)
                else:
                    self._op_lock.release()

            active_armature = self.active_armature

            # verify that we have a pose armature
            if not active_armature:
                print("NO ACTIVE ARMATURE")
                return {'FINISHED'}

            bl_keypoints = self.get_pose(pose_3D)

            self.center_armature(active_armature)
            arm_matrix_world = active_armature.matrix_world

            keypoint = []
            free_bones = ["eye", "eyelid", "eyebrow", "chin", "lip", "upper_lip", "lower_lip", "neck", "shoulder", "nose", "nostril", "jawline", "elbow", "ear", "wrist", "hip", "knee", "ankle"]

            if self._only_body:
                if bpy.data.armatures["Armature"].display_type == "OCTAHEDRAL":  # here we assume that the name of the current armature is "Armature"
                    for bone in active_armature.pose.bones:
                        if any(x in bone.name for x in free_bones) and bone.name in bl_keypoints:
                            keypoint = bl_keypoints[bone.name]
                            # update the bone location only if the accuracy is sufficiently high
                            if keypoint.accuracy > self._detection_threshold:
                                yz_rotation = Vector((
                                    bl_keypoints["shoulder.r"].x - bl_keypoints["shoulder.l"].x,
                                    bl_keypoints["shoulder.r"].y - bl_keypoints["shoulder.l"].y,
                                    bl_keypoints["shoulder.r"].z - bl_keypoints["shoulder.l"].z
                                    )).to_track_quat("-X", "Z").to_matrix().to_4x4()
                                local_rotation = Vector((keypoint.angle_vector[0], 0, keypoint.angle_vector[2])).to_track_quat("-Z", "Z").to_matrix().to_4x4()
                                bpy.data.objects["Armature"].pose.bones[bone.name].matrix = yz_rotation @ local_rotation @ active_armature.data.bones["midhip"].matrix_local.to_4x4()
                else:
                    for bone in active_armature.pose.bones:
                        if any(x in bone.name for x in free_bones) and bone.name in bl_keypoints:
                            keypoint = bl_keypoints[bone.name]
                            # update the bone location only if the accuracy is sufficiently high
                            if keypoint.accuracy > self._detection_threshold:
                                bone_current_rotation = active_armature.data.bones[bone.name].matrix_local.to_quaternion()
                                local_rotation = (arm_matrix_world @ (active_armature.data.bones[bone.name].matrix.to_4x4()).inverted() @ Vector((keypoint.angle_vector[0], 0, keypoint.angle_vector[2]))).to_track_quat("Y", "Z")
                                bone.matrix = Matrix.Translation((keypoint.x, keypoint.y, keypoint.z)) @ (bone_current_rotation @ local_rotation).to_matrix().to_4x4()
            else:
                if bpy.data.armatures["Armature"].display_type == "STICK":  # here we assume that the name of the current armature is "Armature"
                    def move_bone(name):
                        world_co = arm_matrix_world @ active_armature.pose.bones[name].head
                        world_co.x = bl_keypoints[name].x
                        world_co.z = bl_keypoints[name].z
                        local_co = arm_matrix_world.inverted() @ world_co
                        new_mat = active_armature.data.bones[name].matrix.to_4x4().copy()
                        new_mat[0][3] = local_co.x
                        new_mat[1][3] = local_co.y
                        new_mat[2][3] = local_co.z
                        active_armature.pose.bones[name].matrix = new_mat

                    # move the control bone  -> chin if accuracy is sufficiently high
                    if self.update_from_data:
                        move_bone('nose_apex')
                    else:
                        active_armature.pose.bones["nose_apex"].matrix = arm_matrix_world.inverted() @ Matrix.Translation((bl_keypoints["nose_apex"].x, bl_keypoints["nose_apex"].y, bl_keypoints["nose_apex"].z))@ active_armature.data.bones["nose_apex"].matrix.to_quaternion().to_matrix().to_4x4()
                   # move the free bones
                    for bone in active_armature.pose.bones:
                        # check if it is a "movable bone"
                        if any(x in bone.name for x in free_bones) and bone.name in bl_keypoints and bone.name != "nose_apex":
                            keypoint = bl_keypoints[bone.name]
                            # update the bone location only if the accuracy is sufficiently high
                            if keypoint.accuracy > self._detection_threshold:
                                if self.update_from_data:
                                    move_bone(bone.name)
                                else:
                                    active_armature.pose.bones[bone.name].matrix = arm_matrix_world.inverted() @Matrix.Translation((bl_keypoints[bone.name].x, bl_keypoints[bone.name].y, bl_keypoints[bone.name].z))@ active_armature.data.bones[bone.name].matrix.to_quaternion().to_matrix().to_4x4()
                else:
                    # move the control bone  -> chin if accuracy is sufficiently high
                    if bl_keypoints["nose_apex"].accuracy > self._detection_threshold:
                        global_yz_rot = active_armature.data.bones["nose_apex"].matrix.to_4x4().inverted() @ arm_matrix_world.inverted() @ Vector((
                                        bl_keypoints["upper_lip_corner.r"].x - bl_keypoints["upper_lip_corner.l"].x,
                                        bl_keypoints["upper_lip_corner.r"].y - bl_keypoints["upper_lip_corner.l"].y,
                                        bl_keypoints["upper_lip_corner.r"].z - bl_keypoints["upper_lip_corner.l"].z
                                        ))
                        # the x rotation is not applied -> it makes the bones flip
                        # global_x_rot = active_armature.data.bones["nose_apex"].matrix.to_4x4().inverted() @ matrix_world.inverted() @ Vector((
                        #               0,
                        #               bl_keypoints["nose_upper_bridge"].y - bl_keypoints["nose_apex"].y,
                        #               bl_keypoints["nose_upper_bridge"].z - bl_keypoints["nose_apex"].z
                        #               ))
    
                        # addition of the current rotation and rotations in xz
                        global_rotation = active_armature.data.bones["nose_apex"].matrix.to_quaternion() @  global_yz_rot.to_track_quat("X", "Z")  # @ global_x_rot.to_track_quat("-Z", "Y)
                        active_armature.pose.bones["nose_apex"].matrix = Matrix.Translation((bl_keypoints["nose_apex"].x, bl_keypoints["nose_apex"].y, bl_keypoints["nose_apex"].z)) @ global_rotation.to_matrix().to_4x4()
                        # move the free bones
                        for bone in active_armature.pose.bones:
                            # check if it is a "movable bone"
                            if any(x in bone.name for x in free_bones) and bone.name in bl_keypoints:
                                keypoint = bl_keypoints[bone.name]
                                # update the bone location only if the accuracy is sufficiently high
                                if keypoint.accuracy > self._detection_threshold:
                                    bone_current_rotation = active_armature.data.bones[bone.name].matrix_local.to_quaternion()
                                    local_rotation = (arm_matrix_world @ active_armature.data.bones["nose_apex"].matrix.to_4x4() @ (active_armature.data.bones[bone.name].matrix.to_4x4()).inverted() @ Vector((keypoint.angle_vector[0], 0, keypoint.angle_vector[2]))).to_track_quat("Y", "Z")
                                    bone.matrix = Matrix.Translation((keypoint.x, keypoint.y, keypoint.z)) @ (bone_current_rotation @ local_rotation).to_matrix().to_4x4()

            return {'FINISHED'}

    # Centers the armature at the origin of the world
    def center_armature(self, armature: bpy.types.bpy_struct) -> None:
        armature.location = Vector((0, 0, 0))
        return

    def initial_position(self, armature: bpy.types.bpy_struct) -> Dict[str, Vector]:
        return {bone.name: bone.head for bone in armature.pose.bones}

    def compute_vector(self, point1: np.array, point2: np.array) -> Vector:
        return Vector((point2[0] - point1[0], point2[1] - point1[1], point2[2] - point1[2]))

    def normalize_pixels(self, array: np.array, index: int) -> np.array:
        # points are recentered from the center of the frame and normalized by the image length * scale_factor, a rotation is applied
        rows, cols, channels = self._cameras[index].shape
        rows, cols = 640, 640
        #scale_factor = 50.0
        scale_factor = 3.0
        row_origin = floor(rows / 2)
        col_origin = floor(cols / 2)
        new_array = np.empty((0, 3))
        for point in array:
            new_array = np.append(new_array, np.array([[(point[0] - row_origin) * scale_factor/ rows, (point[1] - col_origin) * scale_factor / cols, point[2]]]), axis=0)
        # apply the rotation to each point
        rotation_matrix = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])
        new_array = np.inner(rotation_matrix, new_array).transpose()
        # append the accuracy to the coordinate points
        new_array = np.insert(new_array, 3, values=array[:, 3], axis=1)
        return new_array

    def add_z_coordinate(self, array: np.array) -> np.array:
        return np.insert(array, 2, values=0.0, axis=1)

    def get_map(self, pose) -> Dict[str, Keypoint]:
        if self._only_body:
            return {
            # body keypoints
            "nose": Keypoint(x=pose[0][0], y=pose[0][1], z=pose[0][2], accuracy=pose[0][3]),
            "neck": Keypoint(x=pose[1][0], y=pose[1][1], z=pose[1][2], accuracy=pose[1][3], angle_vector=self.compute_vector(pose[0], pose[1])),
            "neck.r": Keypoint(x=pose[1][0], y=pose[1][1], z=pose[1][2], accuracy=pose[1][3]),
            "neck.l": Keypoint(x=pose[1][0], y=pose[1][1], z=pose[1][2], accuracy=pose[1][3]),
            "shoulder.r": Keypoint(x=pose[2][0], y=pose[2][1], z=pose[2][2], accuracy=pose[2][3], angle_vector=self.compute_vector(pose[2], pose[1])),
            "elbow.r": Keypoint(x=pose[3][0], y=pose[3][1], z=pose[3][2], accuracy=pose[3][3], angle_vector=self.compute_vector(pose[3], pose[2])),
            "wrist.r": Keypoint(x=pose[4][0], y=pose[4][1], z=pose[4][2], accuracy=pose[4][3], angle_vector=self.compute_vector(pose[4], pose[3])),
            "shoulder.l": Keypoint(x=pose[5][0], y=pose[5][1], z=pose[5][2], accuracy=pose[5][3], angle_vector=self.compute_vector(pose[5], pose[1])),
            "elbow.l": Keypoint(x=pose[6][0], y=pose[6][1], z=pose[6][2], accuracy=pose[6][3], angle_vector=self.compute_vector(pose[6], pose[5])),
            "wrist.l": Keypoint(x=pose[7][0], y=pose[7][1], z=pose[7][2], accuracy=pose[7][3], angle_vector=self.compute_vector(pose[7], pose[6])),
            "eye.r": Keypoint(x=pose[15][0], y=pose[15][1], z=pose[15][2], accuracy=pose[15][3]),
            "eye.l": Keypoint(x=pose[16][0], y=pose[16][1], z=pose[16][2], accuracy=pose[16][3]),
            "ear.r": Keypoint(x=pose[17][0], y=pose[17][1], z=pose[17][2], accuracy=pose[17][3]),
            "ear.l": Keypoint(x=pose[18][0], y=pose[18][1], z=pose[18][2], accuracy=pose[18][3]),
            "midhip": Keypoint(x=pose[8][0], y=pose[8][1], z=pose[8][2], accuracy=pose[8][3], angle_vector=self.compute_vector(pose[1], pose[8])),
            "hip.r": Keypoint(x=pose[9][0], y=pose[9][1], z=pose[9][2], accuracy=pose[9][3], angle_vector=self.compute_vector(pose[9], pose[8])),
            "hip.l": Keypoint(x=pose[12][0], y=pose[12][1], z=pose[12][2], accuracy=pose[12][3], angle_vector=self.compute_vector(pose[12], pose[8])),
            "knee.r": Keypoint(x=pose[10][0], y=pose[10][1], z=pose[10][2], accuracy=pose[12][3], angle_vector=self.compute_vector(pose[10], pose[9])),
            "knee.l": Keypoint(x=pose[13][0], y=pose[13][1], z=pose[13][2], accuracy=pose[13][3], angle_vector=self.compute_vector(pose[13], pose[12])),
            "ankle.r": Keypoint(x=pose[11][0], y=pose[11][1], z=pose[11][2], accuracy=pose[11][3], angle_vector=self.compute_vector(pose[11], pose[10])),
            "big_toe.r": Keypoint(x=pose[22][0], y=pose[22][1], z=pose[22][2], accuracy=pose[22][3], angle_vector=self.compute_vector(pose[22], pose[11])),
            "ankle.l": Keypoint(x=pose[14][0], y=pose[14][1], z=pose[14][2], accuracy=pose[14][3], angle_vector=self.compute_vector(pose[14], pose[13])),
            "big_toe.l": Keypoint(x=pose[19][0], y=pose[19][1], z=pose[19][2], accuracy=pose[19][3], angle_vector=self.compute_vector(pose[19], pose[14]))
            }
        else:
            return {
            # face keypoints
            "eyebrow_start.l": Keypoint(x=pose[21][0], y=pose[21][1], z=pose[21][2], accuracy=pose[21][3], angle_vector=self.compute_vector(pose[21], pose[20])),
            "eyebrow_middle.l": Keypoint(x=pose[19][0], y=pose[19][1], z=pose[19][2], accuracy=pose[19][3]),
            "eyebrow_end.l": Keypoint(x=pose[17][0], y=pose[17][1], z=pose[17][2], accuracy=pose[17][3], angle_vector=self.compute_vector(pose[17], pose[18])),
            "eyebrow_start.r": Keypoint(x=pose[22][0], y=pose[22][1], z=pose[22][2], accuracy=pose[22][[3]], angle_vector=self.compute_vector(pose[22], pose[23])),
            "eyebrow_middle.r": Keypoint(x=pose[24][0], y=pose[24][1], z=pose[24][2], accuracy=pose[24][[3]]),
            "eyebrow_end.r": Keypoint(x=pose[26][0], y=pose[26][1], z=pose[26][2], accuracy=pose[26][3], angle_vector=self.compute_vector(pose[26], pose[25])),
           "eye_pupil.l": Keypoint(x=pose[68][0], y=pose[68][1], z=pose[68][2], accuracy=pose[68][3]),
            "eye_outer_corner.l": Keypoint(x=pose[36][0], y=pose[36][1], z=pose[36][2], accuracy=pose[36][3], angle_vector=self.compute_vector(pose[36], pose[37])),
            "eye_inner_corner.l": Keypoint(x=pose[39][0], y=pose[39][1], z=pose[39][2], accuracy=pose[39][3], angle_vector=self.compute_vector(pose[39], pose[38])),
            "eye_pupil.r": Keypoint(x=pose[69][0], y=pose[69][1], z=pose[69][2], accuracy=pose[69][3]),
            "eye_outer_corner.r": Keypoint(x=pose[45][0], y=pose[45][1], z=pose[45][2], accuracy=pose[45][3]),
            "eye_inner_corner.r": Keypoint(x=pose[42][0], y=pose[42][1], z=pose[42][2], accuracy=pose[42][3]),
            "lip_corner.l": Keypoint(x=pose[48][0], y=pose[48][1], z=pose[48][2], accuracy=pose[48][3], angle_vector=self.compute_vector(pose[48], pose[59])),
            "lip_corner.r": Keypoint(x=pose[54][0], y=pose[54][1], z=pose[54][2], accuracy=pose[54][3], angle_vector=self.compute_vector(pose[54], pose[55])),
            "upper_lip_center": Keypoint(x=pose[51][0], y=pose[51][1], z=pose[51][2], accuracy=pose[51][3]),
            "lower_lip_side.l": Keypoint(x=pose[58][0], y=pose[58][1], z=pose[58][2], accuracy=pose[58][3]),
            "lower_lip_side.r": Keypoint(x=pose[56][0], y=pose[56][1], z=pose[56][2], accuracy=pose[56][3]),
            "upper_lip_side.l": Keypoint(x=pose[50][0], y=pose[50][1], z=pose[50][2], accuracy=pose[50][3]),
            "upper_lip_side.r": Keypoint(x=pose[52][0], y=pose[52][1], z=pose[52][2], accuracy=pose[52][3]),
            "eye_inner_top_side.l": Keypoint(x=pose[38][0], y=pose[38][1], z=pose[38][2], accuracy=pose[38][3]),
            "eye_inner_top_side.r": Keypoint(x=pose[43][0], y=pose[43][1], z=pose[43][2], accuracy=pose[43][3]),
            "eye_outer_top_side.l": Keypoint(x=pose[37][0], y=pose[37][1], z=pose[37][2], accuracy=pose[37][3]),
            "eye_outer_top_side.r": Keypoint(x=pose[44][0], y=pose[44][1], z=pose[44][2], accuracy=pose[44][3]),
            "eye_inner_bottom_side.l": Keypoint(x=pose[40][0], y=pose[40][1], z=pose[40][2], accuracy=pose[40][3]),
            "eye_inner_bottom_side.r": Keypoint(x=pose[47][0], y=pose[47][1], z=pose[47][2], accuracy=pose[47][3]),
            "eye_outer_bottom_side.l": Keypoint(x=pose[41][0], y=pose[41][1], z=pose[41][2], accuracy=pose[41][3]),
            "eye_outer_bottom_side.r":  Keypoint(x=pose[46][0], y=pose[46][1], z=pose[46][2], accuracy=pose[46][3]),
            "chin": Keypoint(x=pose[8][0], y=pose[8][1], z=pose[8][2], accuracy=pose[8][3]),
            "nose_apex": Keypoint(x=pose[33][0], y=pose[33][1], z=pose[33][2], accuracy=pose[33][3]),
            "nose_side.l": Keypoint(x=pose[31][0], y=pose[31][1], z=pose[31][2], accuracy=pose[31][3]),
            "nose_side.r": Keypoint(x=pose[35][0], y=pose[35][1], z=pose[35][2], accuracy=pose[35][3]),
            "nostril.l": Keypoint(x=pose[32][0], y=pose[32][1], z=pose[32][2], accuracy=pose[32][3]),
            "nostril.r": Keypoint(x=pose[34][0], y=pose[34][1], z=pose[34][2], accuracy=pose[34][3]),
            "nose_bridge_1": Keypoint(x=pose[30][0], y=pose[30][1], z=pose[30][2], accuracy=pose[30][3]),
            "nose_bridge_2": Keypoint(x=pose[29][0], y=pose[29][1], z=pose[29][2], accuracy=pose[29][3]),
            "nose_bridge_3": Keypoint(x=pose[28][0], y=pose[28][1], z=pose[28][2], accuracy=pose[28][3]),
            "nose_bridge_4": Keypoint(x=pose[27][0], y=pose[27][1], z=pose[27][2], accuracy=pose[27][3]),
            "jawline_1.r": Keypoint(x=pose[9][0], y=pose[9][1], z=pose[9][2], accuracy=pose[9][3]),
            "jawline_2.r": Keypoint(x=pose[10][0], y=pose[10][1], z=pose[10][2], accuracy=pose[10][3]),
            "jawline_3.r": Keypoint(x=pose[11][0], y=pose[11][1], z=pose[11][2], accuracy=pose[11][3]),
             "jawline_4.r": Keypoint(x=pose[12][0], y=pose[12][1], z=pose[12][2], accuracy=pose[12][3]),
            "jawline_5.r": Keypoint(x=pose[13][0], y=pose[13][1], z=pose[13][2], accuracy=pose[13][3]),
            "jawline_6.r": Keypoint(x=pose[14][0], y=pose[14][1], z=pose[14][2], accuracy=pose[14][3]),
            "jawline_7.r": Keypoint(x=pose[15][0], y=pose[15][1], z=pose[15][2], accuracy=pose[15][3]),
            "jawline_8.r": Keypoint(x=pose[16][0], y=pose[16][1], z=pose[16][2], accuracy=pose[16][3]),
            "jawline_1.l": Keypoint(x=pose[7][0], y=pose[7][1], z=pose[7][2], accuracy=pose[7][3]),
            "jawline_2.l": Keypoint(x=pose[6][0], y=pose[6][1], z=pose[6][2], accuracy=pose[6][3]),
            "jawline_3.l": Keypoint(x=pose[5][0], y=pose[5][1], z=pose[5][2], accuracy=pose[5][3]),
            "jawline_4.l": Keypoint(x=pose[4][0], y=pose[4][1], z=pose[4][2], accuracy=pose[4][3]),
            "jawline_5.l": Keypoint(x=pose[3][0], y=pose[3][1], z=pose[3][2], accuracy=pose[3][3]),
            "jawline_6.l": Keypoint(x=pose[2][0], y=pose[2][1], z=pose[2][2], accuracy=pose[2][3]),
            "jawline_7.l": Keypoint(x=pose[1][0], y=pose[1][1], z=pose[1][2], accuracy=pose[1][3]),
            "jawline_8.l": Keypoint(x=pose[0][0], y=pose[0][1], z=pose[0][2], accuracy=pose[0][3])
            }

    def get_pose(self, pose_3D: Optional[np.array], index: int = 0) -> Dict[str, Keypoint]:
        if index > len(self._cameras) - 1:
            return {}

        if self.update_from_data:
            idx = (bpy.context.scene.frame_current - 1) % self.data_len
            pose = self.data[idx]
        else:
            pose = self._cameras[index].pose

        if pose_3D is None:
            pose = self.add_z_coordinate(pose)
            pose = self.normalize_pixels(pose, index)
        else:
            pose = np.insert(pose_3D, 3, values=pose[:, 2], axis=1)

        keypoints = self.get_map(pose)
        return {key: Keypoint(x=keypoints[key].x,
                              y=keypoints[key].y,
                              z=keypoints[key].z,
                              accuracy=keypoints[key].accuracy,
                              angle_vector=keypoints[key].angle_vector) for key in keypoints}

