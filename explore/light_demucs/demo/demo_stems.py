import sys
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parents[1]
THIRD_PARTY_DIR = MODULE_ROOT / "third_party"
if str(THIRD_PARTY_DIR) not in sys.path:
    sys.path.insert(0, str(THIRD_PARTY_DIR))

from demucs.api import Separator, save_audio

MODEL_NAME = "htdemucs"
DEVICE = "cpu"
SHIFTS = 1
OVERLAP = 0.25
SEGMENT = 7
MODEL_REPO = None  # Hugging Face；本地仓改为 Path("/Users/boom/Model/MSS")
INPUT_AUDIO = Path("/Users/boom/workspace/LightClear/assets/audio/music/next_station_heaven.mp3")
OUTPUT_DIR = MODULE_ROOT / "outputs" / MODEL_NAME / INPUT_AUDIO.stem

separator = Separator(
    model=MODEL_NAME,
    repo=MODEL_REPO,
    device=DEVICE,
    shifts=SHIFTS,
    overlap=OVERLAP,
    segment=SEGMENT,
    progress=True,
)
_, stems = separator.separate_audio_file(INPUT_AUDIO)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
for stem_name, stem_wav in stems.items():
    save_audio(stem_wav, str(OUTPUT_DIR / f"{stem_name}.wav"), samplerate=separator.samplerate)
