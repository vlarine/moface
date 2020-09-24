import bpy
import os

addonName = os.path.basename(os.path.dirname(__file__))

class AddonPreferences(bpy.types.AddonPreferences):
    bl_idname = addonName

    name_prefix: bpy.props.StringProperty(
        name="Prefix used for generated objects name",
        description="Prefix used for generated objects name",
        default="MoFace_"
    )

    detection_threshold: bpy.props.FloatProperty(
        name="Detection threshold",
        description="Threshold to consider that the detection is good enough",
        default=0.75,
        min=0.0,
        max=1.0
    )


    def draw(self, context) -> None:
        layout = self.layout

        main_col = layout.column(align=True)
        main_col.prop(self, "name_prefix")

        main_col.prop(self, "detection_threshold")
