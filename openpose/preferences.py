import bpy
import os

addonName = os.path.basename(os.path.dirname(__file__))


class AddonPreferences(bpy.types.AddonPreferences):
    bl_idname = addonName

    name_prefix: bpy.props.StringProperty(
        name="Prefix used for generated objects name",
        description="Prefix used for generated objects name",
        default="OpenPose_"
    )

    show_debug_mesh: bpy.props.BoolProperty(
        name="Show debug mesh",
        description="Show a debug mesh generated from the detected face",
        default=False
    )

    show_in_blender: bpy.props.BoolProperty(
        name="Show camera in the UV/Image Editor",
        description="Show input cameras as an image in the UV/Image Editor",
        default=False
    )

    mirror_cameras: bpy.props.BoolProperty(
        name="Flip camera view",
        description="Apply a flip transformation to camera view before display",
        default=False
    )

    calibration_shots: bpy.props.IntProperty(
        name="Calibration shots count",
        description="Number of shots taken for calibration",
        default=20,
        min=6
    )

    detection_threshold: bpy.props.FloatProperty(
        name="Detection threshold",
        description="Threshold to consider that the detection is good enough",
        default=0.75,
        min=0.0,
        max=1.0
    )

    calibration_set: bpy.props.BoolProperty(
        name="Calibration flag",
        description="Calibration flag, True if the stereo camera calibration is done",
        default=False
    )

    calibration_camera_1_intrinsics: bpy.props.FloatVectorProperty(
        name="Intrinsic parameters for camera 1",
        description="Intrinsic parameters for camera 1",
        size=14
    )

    calibration_camera_1_extrinsics: bpy.props.FloatVectorProperty(
        name="Extrinsics parameters for camera 1",
        description="Extrinsics parameters for camera 1",
        size=21
    )

    calibration_camera_2_intrinsics: bpy.props.FloatVectorProperty(
        name="Intrinsic parameters for camera 2",
        description="Intrinsic parameters for camera 2",
        size=14
    )

    calibration_camera_2_extrinsics: bpy.props.FloatVectorProperty(
        name="Extrinsics parameters for camera 2",
        description="Extrinsics parameters for camera 2",
        size=21
    )

    body_and_face: bpy.props.BoolProperty(
        name="Body pose and face detection",
        description="Detect body poses first. It is more accurate than just face detection, but it is more computationally expensive",
        default=False
    )

    only_face: bpy.props.BoolProperty(
        name="Face detection only",
        description="Detects the face only",
        default=True
    )
    mask_frame: bpy.props.BoolProperty(
        name="Mask frame",
        description="Mask the frame around the bounding box to reduce the noise during detection",
        default=False
    )

    pose_choice: bpy.props.EnumProperty(
        name="my enum",
        description="my enum description",
        items=[
               ("FACE", "Face detection", "Detects the face only"),
               ("BODY", "Body detection", "Detects the body only")
              ]
    )

    def draw(self, context) -> None:
        layout = self.layout

        main_col = layout.column(align=True)
        main_col.prop(self, "name_prefix")
        main_col.prop(self, "calibration_shots")

        main_col.prop(self, "detection_threshold")
        main_col.prop(self, "show_in_blender")
        main_col.prop(self, "mirror_cameras")
        main_col.prop(self, "mask_frame")

        face_or_body_panel = layout.row()
        face_or_body_panel.prop(self, "pose_choice", expand=True)

        if self.pose_choice == "FACE":
            box = layout.box()
            col = box.column(align=True)
            col.prop(self, "show_debug_mesh")
            self.only_face = False
#           Temporarily disabled; At the moment, detect face from a rectangle is not working
#            col.prop(self, "body_and_face")
#            if self.body_and_face:
#                self.only_face = False
#            else:
#                self.only_face = True
        else:
            # Making sure that these settings are inactive if body_detection is on
            self.show_debug_mesh = False
            self.body_and_face = False
            self.only_face = False
