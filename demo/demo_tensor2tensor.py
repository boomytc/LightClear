import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
THIRD_PARTY_DIR = PROJECT_ROOT / "third_party"
if str(THIRD_PARTY_DIR) not in sys.path:
    sys.path.insert(0, str(THIRD_PARTY_DIR))

from clearvoice import ClearVoice

TASK_NAME = "speech_enhancement"
MODEL_NAME = "FRCRN_SE_16K"
SAMPLE_RATE = 16000
INPUT_AUDIO = "assets/clearvoice_samples/input.wav"
OUTPUT_NUMPY_AUDIO = "outputs/tensor2tensor/FRCRN_SE_16K_numpy.wav"
OUTPUT_TORCH_AUDIO = "outputs/tensor2tensor/FRCRN_SE_16K_torch.wav"

myClearVoice = ClearVoice(task=TASK_NAME, model_names=[MODEL_NAME])

audio, sample_rate = sf.read(INPUT_AUDIO)
if sample_rate != SAMPLE_RATE:
    audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=SAMPLE_RATE)
if len(audio.shape) > 1:
    audio = audio[:, 0]
audio = np.reshape(audio, [1, audio.shape[0]]).astype(np.float32)

Path(OUTPUT_NUMPY_AUDIO).parent.mkdir(parents=True, exist_ok=True)

output_numpy = myClearVoice(audio, False)
sf.write(OUTPUT_NUMPY_AUDIO, output_numpy[0, :], SAMPLE_RATE)

torch_audio = torch.from_numpy(audio.copy())
output_torch = myClearVoice(torch_audio, False)
if isinstance(output_torch, torch.Tensor):
    output_torch = output_torch.detach().cpu().numpy()
sf.write(OUTPUT_TORCH_AUDIO, output_torch[0, :], SAMPLE_RATE)
