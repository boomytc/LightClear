import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
THIRD_PARTY_DIR = PROJECT_ROOT / "third_party"
if str(THIRD_PARTY_DIR) not in sys.path:
    sys.path.insert(0, str(THIRD_PARTY_DIR))

from clearvoice import ClearVoice

myClearVoice = ClearVoice(task='speech_separation', model_names=['MossFormer2_SS_16K'])

##第一种调用方法：处理输入波形并返回输出波形，然后写入 output_MossFormer2_SS_16K_s1.wav 和 output_MossFormer2_SS_16K_s2.wav
output_wav = myClearVoice(input_path='assets/clearvoice_samples/input_ss.wav', online_write=False)
myClearVoice.write(output_wav, output_path='outputs/MossFormer2_SS_16K/input_ss.wav')

#第二种调用方法：处理 'path_to_input_wavs/' 中的所有 wav 文件，并将输出写入 'path_to_output_wavs'
myClearVoice(input_path='assets/clearvoice_samples/path_to_input_wavs_ss', online_write=True, output_path='outputs/path_to_output_wavs')

##第三种调用方法：处理 .scp 文件中列出的 wav 文件，并将输出写入 'path_to_output_wavs_scp/'
myClearVoice(input_path='assets/clearvoice_samples/scp/audio_samples_mix.scp', online_write=True, output_path='outputs/path_to_output_wavs_scp')
