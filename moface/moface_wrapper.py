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

addonName = os.path.basename(os.path.dirname(__file__))

@dataclass
class Keypoint:
    x: float
    y: float
    z: float
    accuracy: float
    angle_vector: Vector = Vector((0.0, 0.0, 0.0))


class MoFaceWrapper:
    def __init__(self) -> None:

        # Options
        self._name_prefix = "MoFace_"
        self._detection_threshold = 0.50  # Threshold for pose detection


    def start(self) -> bool:
        """
        Start the detection
        """
        self._name_prefix = bpy.context.preferences.addons[addonName].preferences.name_prefix
        self._detection_threshold = bpy.context.preferences.addons[addonName].preferences.detection_threshold

    def stop(self) -> None:
        """
        Stop the detection.
        Currently, it does not clean OpenPose objects due to a bug in the library
        """
        pass


    def init_data(self, scene):
        self.active_armature = next((obj for obj in scene.objects if obj.type == "ARMATURE"), None)

    @persistent
    def update(self, scene, *args, **kwargs) -> None:
        active_armature = self.active_armature

        # verify that we have a pose armature
        if not active_armature:
            print("NO ACTIVE ARMATURE")
            return {'FINISHED'}

        bl_keypoints = self.get_pose()

        self.center_armature(active_armature)
        arm_matrix_world = active_armature.matrix_world

        keypoint = []
        free_bones = ["eye", "eyelid", "eyebrow", "chin", "lip", "upper_lip", "lower_lip", "neck", "shoulder", "nose", "nostril", "jawline", "elbow", "ear", "wrist", "hip", "knee", "ankle"]

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
        move_bone('nose_apex')

        # move the free bones
        for bone in active_armature.pose.bones:
            # check if it is a "movable bone"
            if any(x in bone.name for x in free_bones) and bone.name in bl_keypoints and bone.name != "nose_apex":
                keypoint = bl_keypoints[bone.name]
                # update the bone location only if the accuracy is sufficiently high
                if keypoint.accuracy > self._detection_threshold:
                    move_bone(bone.name)

        return {'FINISHED'}

    # Centers the armature at the origin of the world
    def center_armature(self, armature: bpy.types.bpy_struct) -> None:
        armature.location = Vector((0, 0, 0))
        return

    def initial_position(self, armature: bpy.types.bpy_struct) -> Dict[str, Vector]:
        return {bone.name: bone.head for bone in armature.pose.bones}

    def compute_vector(self, point1: np.array, point2: np.array) -> Vector:
        return Vector((point2[0] - point1[0], point2[1] - point1[1], point2[2] - point1[2]))

    def normalize_pixels(self, array: np.array) -> np.array:
        # points are recentered from the center of the frame and normalized by the image length * scale_factor, a rotation is applied
        rows, cols = self.active_armature.keypoints_height, self.active_armature.keypoints_width
        scale_factor = self.active_armature.scale_factor
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

    def get_pose(self) -> Dict[str, Keypoint]:
        idx = (bpy.context.scene.frame_current - 1) % self.data_len
        pose = self.data[idx]
        pose = self.add_z_coordinate(pose)
        pose = self.normalize_pixels(pose)

        keypoints = self.get_map(pose)
        return {key: Keypoint(x=keypoints[key].x,
                              y=keypoints[key].y,
                              z=keypoints[key].z,
                              accuracy=keypoints[key].accuracy,
                              angle_vector=keypoints[key].angle_vector) for key in keypoints}

