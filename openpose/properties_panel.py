# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
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

import bpy
import sys
from typing import Optional
from . import operators

# Import OpenPose
try:
    sys.path.insert(0, '/usr/local/python')  # PyOpenPose is installed in this directory by default
    import pyopenpose as op
    USE_OPENPOSE = True
except ImportError as e:
    print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake of OpenPose and installed it?')
    USE_OPENPOSE = False
    #raise e

class OpenPosePropertiesPanel(bpy.types.Panel):
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_context = "object"
    bl_label = "OpenPose Properties"
    bl_idname = "OPENPOSE_PT_properties_panel"

    bpy.types.Object.openpose_active = bpy.props.BoolProperty(
        name="Use OpenPose",
        description="Control the armature through OpenPose",
        default=False
    )

    bpy.types.Object.openpose_filepath = bpy.props.StringProperty(
        name=".tsv file",
        description="Load OpenPose keypoints from file",
        default="./assets/op_keypoints.tsv",
        maxlen= 1024,
        subtype='FILE_PATH'
    )


    @classmethod
    def poll(cls, context) -> bool:
        obj = context.active_object
        return (obj and obj.type == 'ARMATURE')

    def draw_header(self, context):
        self.layout.prop(context.object, "openpose_active", text="")

    def draw(self, context) -> None:
        layout = self.layout
        main_col = layout.column(align=True)

        if USE_OPENPOSE:
            row = main_col.row(align=True)
            row.operator("openpose.activate_and_run", text="Stop OpenPose" if context.scene.openpose.state else "Start OpenPose")
            row = main_col.row(align=True)
            row.operator("openpose.reset_calibration", text="Reset calibration")
            row.operator("openpose.save_userpref", text="Save calibration")

        main_col.label(text='Animate from file')
        row = main_col.row(align=True)
        row.prop(context.object, 'openpose_filepath')
        row = main_col.row(align=True)
        row.operator('openpose.load_file', text='Load file')




