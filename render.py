import sys
sys.path.append('./')
import bpy
import argparse


class Options:
    def __init__(self, argv):

        if "--" not in argv:
            self.argv = []  # as if no args are passed
        else:
            self.argv = argv[argv.index("--") + 1:]

        usage_text = (
                "Run blender in background mode with this script:"
                "  blender --background --python [main_python_file] -- [options]"
        )

        self.parser = argparse.ArgumentParser(description=usage_text)
        self.initialize()
        self.args = self.parser.parse_args(self.argv)

    def initialize(self):
        self.parser.add_argument('--save_path', type=str, default='./results/', help='path of output video file')
        self.parser.add_argument('--keypoints_filepath', type=str, default='./assets/op_keypoints.tsv', help='Keypoints filepath')
        self.parser.add_argument('--render_engine', type=str, default='eevee',
                                 help='name of preferable render engine: cycles, eevee')
        self.parser.add_argument('--render', action='store_true', default=False, help='render an output video')

    def parse(self):
            return self.args


def add_rendering_parameters(scene, args):
    scene.render.filepath = args.save_path

    if args.render_engine == 'cycles':
        scene.render.engine = 'CYCLES'
        scene.cycles.device = 'GPU'
    elif args.render_engine == 'eevee':
        scene.render.engine = 'BLENDER_EEVEE'

    scene.render.image_settings.file_format = 'PNG'

    return scene


def main(args):
    add_rendering_parameters(bpy.context.scene, args)

    if args.render:
        bpy.context.object.keypoints_filepath = args.keypoints_filepath
        bpy.ops.moface.load_file()
        bpy.ops.render.render(animation=True, use_viewport=True)


if __name__ == '__main__':
    args = Options(sys.argv).parse()
    main(args)






