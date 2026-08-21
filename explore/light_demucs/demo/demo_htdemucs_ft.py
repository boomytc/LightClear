import sys
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parents[1]
THIRD_PARTY_DIR = MODULE_ROOT / "third_party"
if str(THIRD_PARTY_DIR) not in sys.path:
    sys.path.insert(0, str(THIRD_PARTY_DIR))

from demucs.api import Separator, save_audio

MODEL_NAME = "htdemucs_ft"
STEM_NAME = "vocals"
STEM_NAMES = ("drums", "bass", "other", "vocals")
DEVICE = "cpu"
SHIFTS = 1
OVERLAP = 0.25
SEGMENT = 7
MODEL_REPO = MODULE_ROOT / "models" / MODEL_NAME
INPUT_AUDIO = Path("/Users/boom/workspace/LightClear/assets/audio/music/next_station_heaven.mp3")
OUTPUT_DIR = MODULE_ROOT / "outputs" / MODEL_NAME / INPUT_AUDIO.stem
OUTPUT_VOCALS = MODULE_ROOT / "outputs" / MODEL_NAME / f"{INPUT_AUDIO.stem}_{STEM_NAME}.wav"
OUTPUT_ACCOMPANIMENT = MODULE_ROOT / "outputs" / MODEL_NAME / f"{INPUT_AUDIO.stem}_no_{STEM_NAME}.wav"

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
vocals = stems[STEM_NAME]
accompaniment = stems["drums"] + stems["bass"] + stems["other"]

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
for stem_name in STEM_NAMES:
    save_audio(stems[stem_name], str(OUTPUT_DIR / f"{stem_name}.wav"), samplerate=separator.samplerate)
save_audio(vocals, str(OUTPUT_VOCALS), samplerate=separator.samplerate)
save_audio(accompaniment, str(OUTPUT_ACCOMPANIMENT), samplerate=separator.samplerate)
