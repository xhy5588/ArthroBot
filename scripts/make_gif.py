"""Convert an MP4 into a palette-optimized GIF for the README (uses imageio's bundled ffmpeg).

    python scripts/make_gif.py build/humanoid_training/getup_videos/update_010000_front_handover_plain.mp4 docs/media/humanoid_getup.gif
    python scripts/make_gif.py clip.mp4 clip.gif --width 640 --fps 12 --start 1 --duration 6 --crop 900:506:200:120
"""
import argparse
from pathlib import Path
import subprocess

import imageio_ffmpeg


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--width', type=int, default=800, help='Output width in pixels (height keeps the aspect ratio).')
    parser.add_argument('--fps', type=int, default=15)
    parser.add_argument('--start', type=float, default=0., help='Seconds to skip at the start.')
    parser.add_argument('--duration', type=float, help='Seconds to keep (default: to the end).')
    parser.add_argument('--crop', help='Crop before scaling, as ffmpeg width:height:x:y in source pixels.')
    parser.add_argument('--colors', type=int, default=256, help='Palette size; grey scenes look fine with 64.')
    args = parser.parse_args()
    trim = ['-ss', str(args.start)] + (['-t', str(args.duration)] if args.duration else [])
    # One palette for the whole clip; only changed rectangles are re-encoded, which keeps static scenes small.
    crop = f'crop={args.crop},' if args.crop else ''
    filters = (f'{crop}fps={args.fps},scale={args.width}:-2:flags=lanczos,split[frames][copy];'
               f'[copy]palettegen=max_colors={args.colors}:stats_mode=diff[palette];'
               '[frames][palette]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-loglevel', 'error', '-y', *trim, '-i', str(args.input),
                    '-vf', filters, '-loop', '0', str(args.output)], check=True)
    print(f'Wrote {args.output} ({args.output.stat().st_size / 1e6:.1f} MB)')


if __name__ == '__main__':
    main()
