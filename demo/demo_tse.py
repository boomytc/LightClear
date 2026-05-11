import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
THIRD_PARTY_DIR = PROJECT_ROOT / "third_party"
if str(THIRD_PARTY_DIR) not in sys.path:
    sys.path.insert(0, str(THIRD_PARTY_DIR))

from clearvoice import ClearVoice

TASK_NAME = "target_speaker_extraction"
MODEL_NAME = "AV_MossFormer2_TSE_16K"
INPUT_VIDEO = "assets/clearvoice_samples/path_to_input_videos_tse/001.avi"
INPUT_VIDEO_DIR = "assets/clearvoice_samples/path_to_input_videos_tse"
INPUT_VIDEO_SCP = "assets/clearvoice_samples/scp/video_samples.scp"
OUTPUT_VIDEO = "outputs/AV_MossFormer2_TSE_16K/single_video"
OUTPUT_VIDEO_DIR = "outputs/AV_MossFormer2_TSE_16K/path_to_output_videos"
OUTPUT_VIDEO_SCP_DIR = "outputs/AV_MossFormer2_TSE_16K/path_to_output_videos_scp"

myClearVoice = ClearVoice(task=TASK_NAME, model_names=[MODEL_NAME])

myClearVoice(input_path=INPUT_VIDEO, online_write=True, output_path=OUTPUT_VIDEO)

myClearVoice(input_path=INPUT_VIDEO_DIR, online_write=True, output_path=OUTPUT_VIDEO_DIR)

myClearVoice(input_path=INPUT_VIDEO_SCP, online_write=True, output_path=OUTPUT_VIDEO_SCP_DIR)
