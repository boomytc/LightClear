import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

MODULE_ROOT = Path(__file__).resolve().parents[1]
THIRD_PARTY_DIR = MODULE_ROOT / "third_party"
if str(THIRD_PARTY_DIR) not in sys.path:
    sys.path.insert(0, str(THIRD_PARTY_DIR))

from clearvoice import ClearVoice

TASK_NAME = "speech_enhancement"
MODEL_NAME = "MossFormer2_SE_48K"
SAMPLE_RATE = 48000
INPUT_AUDIO = "/Users/boom/workspace/LightClear/assets/clearvoice_samples/input.wav"
OUTPUT_AUDIO = str(MODULE_ROOT / "outputs" / "numpy2numpy" / "MossFormer2_SE_48K_batch.wav")

myClearVoice = ClearVoice(task=TASK_NAME, model_names=[MODEL_NAME])

audio, sample_rate = sf.read(INPUT_AUDIO)
if sample_rate != SAMPLE_RATE:
    audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=SAMPLE_RATE)
if len(audio.shape) < 2:
    audio = np.reshape(audio, [1, audio.shape[0]])
audio = audio.astype(np.float32)

Path(OUTPUT_AUDIO).parent.mkdir(parents=True, exist_ok=True)
output_wav = myClearVoice(audio, False)
sf.write(OUTPUT_AUDIO, output_wav[0, :], SAMPLE_RATE)
