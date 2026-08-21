import sys
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parents[1]
THIRD_PARTY_DIR = MODULE_ROOT / "third_party"
if str(THIRD_PARTY_DIR) not in sys.path:
    sys.path.insert(0, str(THIRD_PARTY_DIR))

from clearvoice import ClearVoice

TASK_NAME = "speech_enhancement"
MODEL_NAME = "MossFormerGAN_SE_16K"
INPUT_AUDIO = "/Users/boom/workspace/LightClear/assets/audio/noisy/test.wav"
INPUT_DIR = "/Users/boom/workspace/LightClear/assets/audio/noisy"
INPUT_SCP = "/Users/boom/workspace/LightClear/assets/manifests/se.scp"
OUTPUT_FILE = str(MODULE_ROOT / "outputs" / "MossFormerGAN_SE_16K" / "test.wav")
OUTPUT_DIR = str(MODULE_ROOT / "outputs" / "MossFormerGAN_SE_16K" / "path_to_output_wavs")
OUTPUT_SCP_DIR = str(MODULE_ROOT / "outputs" / "MossFormerGAN_SE_16K" / "path_to_output_wavs_scp")

myClearVoice = ClearVoice(task=TASK_NAME, model_names=[MODEL_NAME])

output_wav = myClearVoice(input_path=INPUT_AUDIO, online_write=False)
myClearVoice.write(output_wav, output_path=OUTPUT_FILE)

myClearVoice(input_path=INPUT_DIR, online_write=True, output_path=OUTPUT_DIR)

myClearVoice(input_path=INPUT_SCP, online_write=True, output_path=OUTPUT_SCP_DIR)
