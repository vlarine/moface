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

class MoFacePropertiesPanel(bpy.types.Panel):
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_context = "object"
    bl_label = "MoFace Properties"
    bl_idname = "MOFACE_PT_properties_panel"

    bpy.types.Object.keypoints_filepath = bpy.props.StringProperty(
        name=".tsv file",
        description="Load keypoints from file",
        default="./assets/op_keypoints.tsv",
        maxlen= 1024,
        subtype='FILE_PATH'
    )


    @classmethod
    def poll(cls, context) -> bool:
        obj = context.active_object
        return (obj and obj.type == 'ARMATURE')

    def draw(self, context) -> None:
        layout = self.layout
        main_col = layout.column(align=True)

        main_col.label(text='Animate from file')
        row = main_col.row(align=True)
        row.prop(context.object, 'keypoints_filepath')
        row = main_col.row(align=True)
        row.operator('moface.load_file', text='Load file')




