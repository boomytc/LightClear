import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
THIRD_PARTY_DIR = PROJECT_ROOT / "third_party"
if str(THIRD_PARTY_DIR) not in sys.path:
    sys.path.insert(0, str(THIRD_PARTY_DIR))

from clearvoice import ClearVoice

myClearVoice = ClearVoice(task='speech_super_resolution', model_names=['MossFormer2_SR_48K'])

##第一种调用方法：处理来自输入文件的波形并返回输出波形，然后以相同的音频格式写入 output_MossFormer2_SR_48K_xxx
output_wav = myClearVoice(input_path='assets/clearvoice_samples/input_sr.wav', online_write=False)
myClearVoice.write(output_wav, output_path='outputs/MossFormer2_SR_48K_input_sr_8k.wav')

##第二种调用方法：处理 'path_to_input_wavs_sr/' 中的所有 wav 文件，并将输出写入 'path_to_output_wavs'
myClearVoice(input_path='assets/clearvoice_samples/path_to_input_wavs_sr', online_write=True, output_path='outputs/path_to_output_wavs')

##第三种调用方法：处理 .scp 文件中列出的 wav 文件，并将输出写入 'path_to_output_wavs_scp/'
myClearVoice(input_path='assets/clearvoice_samples/scp/audio_samples_sr.scp', online_write=True, output_path='outputs/path_to_output_wavs_scp')
