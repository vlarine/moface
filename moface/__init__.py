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

if "bpy" in locals():
    import imp
    imp.reload('operators')
    imp.reload('properties_panel')
    imp.reload('preferences')
else:
    import bpy
    from . import operators
    from . import properties_panel
    from . import preferences

__copyright__ = """
This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public
License as published by the Free Software Foundation; either version 3 of the license, or (at your option) any later
version (http://www.gnu.org/licenses/).
"""
__license__ = "GPLv3"

bl_info = {
    "name": "MoFace",
    "author": "Vladimir Larin",
    "version": (0, 0, 1),
    "blender": (2, 80, 0),
    "description": "Allows for controlling an armature through the keypoints",
    "category": "Animation"
}

def register() -> None:
    """
    Register MoFace addon classes
    """
    bpy.utils.register_class(properties_panel.MoFacePropertiesPanel)
    bpy.utils.register_class(preferences.AddonPreferences)
    bpy.utils.register_class(operators.LoadFile)


def unregister() -> None:
    """
    Unregister MoFace addon classes
    """
    bpy.utils.unregister_class(properties_panel.MoFacePropertiesPanel)
    bpy.utils.unregister_class(preferences.AddonPreferences)
    bpy.utils.unregister_class(operators.LoadFile)


if __name__ == "__main__":
    register()
