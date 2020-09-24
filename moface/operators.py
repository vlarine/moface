import bpy
import os
import numpy as np

from . import moface_wrapper

addonName = os.path.basename(os.path.dirname(__file__))

class LoadFile(bpy.types.Operator):
    """
    Load keypoints from file
    """
    bl_idname = "moface.load_file"
    bl_label = "Load keypoints from file"
    wrapper: moface_wrapper.MoFaceWrapper = None

    def activate_moface(context, data):
        if not LoadFile.wrapper:
            LoadFile.wrapper = moface_wrapper.MoFaceWrapper()

        LoadFile.wrapper.stop()
        LoadFile.wrapper.data = data
        LoadFile.wrapper.data_len = data.shape[0]
        LoadFile.wrapper.update_from_data = True

        context.scene.frame_end = data.shape[0]
        LoadFile.wrapper.init_data(context.scene)

        LoadFile.wrapper.start()

        if LoadFile.wrapper.update not in bpy.app.handlers.frame_change_pre:
            bpy.app.handlers.frame_change_pre.append(LoadFile.wrapper.update)

    def execute(self, context) -> set:
        keypoints_filepath = context.object.keypoints_filepath
        if os.path.exists(keypoints_filepath):
            data = []
            with open(keypoints_filepath) as f:
                for line in f:
                    arr = [float(x) for x in line.strip().split('\t')]
                    frame = []
                    for i in range(int(len(arr) / 3)):
                        frame.append([arr[i * 3], arr[i * 3 + 1], arr[i * 3 + 2]])
                    data.append(frame)
            data = np.array(data)
            bpy.ops.screen.animation_cancel()
            LoadFile.activate_moface(context, data)
            bpy.ops.screen.animation_play()

        return {'FINISHED'}

