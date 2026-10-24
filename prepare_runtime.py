"""Stage the bundled FFmpeg executable for the standalone Windows build."""
from pathlib import Path
import shutil

import imageio_ffmpeg

root = Path(__file__).resolve().parent
target = root / "bin" / "ffmpeg.exe"
target.parent.mkdir(exist_ok=True)
shutil.copy2(imageio_ffmpeg.get_ffmpeg_exe(), target)
print(f"Bundled FFmpeg: {target}")
