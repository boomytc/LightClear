import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
THIRD_PARTY_DIR = PROJECT_ROOT / "third_party"
if str(THIRD_PARTY_DIR) not in sys.path:
    sys.path.insert(0, str(THIRD_PARTY_DIR))

from clearvoice import ClearVoice

TASK_NAME = "speech_enhancement"
MODEL_NAME = "FRCRN_SE_16K"
INPUT_AUDIO = "assets/clearvoice_samples/input.wav"
INPUT_DIR = "assets/clearvoice_samples/path_to_input_wavs"
INPUT_SCP = "assets/clearvoice_samples/scp/audio_samples.scp"
OUTPUT_FILE = "outputs/FRCRN_SE_16K/input.wav"
OUTPUT_DIR = "outputs/FRCRN_SE_16K/path_to_output_wavs"
OUTPUT_SCP_DIR = "outputs/FRCRN_SE_16K/path_to_output_wavs_scp"

myClearVoice = ClearVoice(task=TASK_NAME, model_names=[MODEL_NAME])

output_wav = myClearVoice(input_path=INPUT_AUDIO, online_write=False)
myClearVoice.write(output_wav, output_path=OUTPUT_FILE)

myClearVoice(input_path=INPUT_DIR, online_write=True, output_path=OUTPUT_DIR)

myClearVoice(input_path=INPUT_SCP, online_write=True, output_path=OUTPUT_SCP_DIR)
