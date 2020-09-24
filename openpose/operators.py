import bpy
import os
import numpy as np

from . import openpose_wrapper

addonName = os.path.basename(os.path.dirname(__file__))

class SwitchState(bpy.types.PropertyGroup):
    state: bpy.props.BoolProperty(
        name="switch_openpose",
        description="Enable OpenPose",
        default=False
    )


class OpenPoseStarter(bpy.types.Operator):
    """
    Activate and run OpenPose
    If a stereo pair is detected, also runs the calibration
    """
    bl_idname = "openpose.activate_and_run"
    bl_label = "Activate and run OpenPose"
    wrapper: openpose_wrapper.OpenPoseWrapper = None

    def update_state(context):
        if context.scene.openpose.state:
            context.scene.openpose.state = False
        else:
            context.scene.openpose.state = True

    def activate_openpose(context):
        if context.scene.openpose.state:
            if not OpenPoseStarter.wrapper:
                OpenPoseStarter.wrapper = openpose_wrapper.OpenPoseWrapper()
                OpenPoseStarter.wrapper.start()
            if OpenPoseStarter.wrapper.update not in bpy.app.handlers.frame_change_pre:
                bpy.app.handlers.frame_change_pre.append(OpenPoseStarter.wrapper.update)
        else:
            if OpenPoseStarter.wrapper.update in bpy.app.handlers.frame_change_pre:
                bpy.app.handlers.frame_change_pre.remove(OpenPoseStarter.wrapper.update)
            OpenPoseStarter.wrapper.stop()
            # Here the wrapper should be destroyed, once PyOpenPose provides the API.
            # OpenPoseStarter.wrapper = None

    def execute(self, context) -> set:
        OpenPoseStarter.update_state(context)
        OpenPoseStarter.activate_openpose(context)
        bpy.ops.screen.animation_play()
        return {'FINISHED'}


class OpenPoseResetCalibration(bpy.types.Operator):
    """
    Reset stereo camera calibration
    """
    bl_idname = "openpose.reset_calibration"
    bl_label = "Reset stereo camera calibration"

    def execute(self, context) -> set:
        bpy.context.preferences.addons[addonName].preferences.calibration_set = False
        return {'FINISHED'}


class OpenPoseSaveUserPrefs(bpy.types.Operator):
    """
    Shortcut operator for saving user preferences
    """
    bl_idname = "openpose.save_userpref"
    bl_label = "Save calibration along user preferences"

    def execute(self, context) -> set:
        bpy.ops.wm.save_userpref()
        return {'FINISHED'}

class LoadFile(bpy.types.Operator):
    """
    Load openpose kyepoints from file
    """
    bl_idname = "openpose.load_file"
    bl_label = "Load openpose kyepoints from file"
    wrapper: openpose_wrapper.OpenPoseWrapper = None

    def activate_openpose(context, data):
        if not LoadFile.wrapper:
            LoadFile.wrapper = openpose_wrapper.OpenPoseWrapper()

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
        openpose_filepath = context.object.openpose_filepath
        if os.path.exists(openpose_filepath):
            data = []
            with open(openpose_filepath) as f:
                for line in f:
                    arr = [float(x) for x in line.strip().split('\t')]
                    frame = []
                    for i in range(int(len(arr) / 3)):
                        frame.append([arr[i * 3], arr[i * 3 + 1], arr[i * 3 + 2]])
                    data.append(frame)
            data = np.array(data)
            bpy.ops.screen.animation_cancel()
            LoadFile.activate_openpose(context, data)
            bpy.ops.screen.animation_play()

        return {'FINISHED'}

