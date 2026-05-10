import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
THIRD_PARTY_DIR = PROJECT_ROOT / "third_party"
if str(THIRD_PARTY_DIR) not in sys.path:
    sys.path.insert(0, str(THIRD_PARTY_DIR))

from clearvoice import ClearVoice

##-----演示一：使用 MossFormer2_SR_48K 模型进行语音超分辨率 -----------------
if False:
    myClearVoice = ClearVoice(task='speech_super_resolution', model_names=['MossFormer2_SR_48K'])

    ##第一种调用方法：处理来自输入文件的波形并返回输出波形，然后以相同的音频格式写入 output_MossFormer2_SR_48K_xxx
    output_wav = myClearVoice(input_path='assets/clearvoice_samples/input_sr.wav', online_write=False)
    myClearVoice.write(output_wav, output_path='outputs/output_MossFormer2_SR_48K_input_sr_8k.wav')
    
    ##第二种调用方法：处理 'path_to_input_wavs_sr/' 中的所有 wav 文件，并将输出写入 'path_to_output_wavs'
    myClearVoice(input_path='assets/clearvoice_samples/path_to_input_wavs_sr', online_write=True, output_path='outputs/path_to_output_wavs')

    ##第三种调用方法：处理 .scp 文件中列出的 wav 文件，并将输出写入 'path_to_output_wavs_scp/'
    myClearVoice(input_path='assets/clearvoice_samples/scp/audio_samples_sr.scp', online_write=True, output_path='outputs/path_to_output_wavs_scp')
    
##-----演示二：在带噪语音数据上使用 MossFormer2_SR_48K 模型进行语音超分辨率 -----------------
if False:
    # 假设你有带噪语音音频并希望进行语音超分辨率
    # 分别为语音增强和超分辨率构建两个对象。
    myClearVoice_SE = ClearVoice(task='speech_enhancement', model_names=['MossFormer2_SE_48K'])
    myClearVoice_SR = ClearVoice(task='speech_super_resolution', model_names=['MossFormer2_SR_48K'])
    
    # 执行语音增强
    output_wav = myClearVoice_SE(input_path='assets/clearvoice_samples/input.wav', online_write=False)
    myClearVoice_SE.write(output_wav, output_path='outputs/output_MossFormer2_SE_48K_input.wav')
    # 执行语音超分辨率
    output_wav = myClearVoice_SR(input_path='outputs/output_MossFormer2_SE_48K_input.wav', online_write=False)
    myClearVoice_SR.write(output_wav, output_path='outputs/output_MossFormer2_SR_48K_input.wav')
    
    # 执行语音增强
    output_wav = myClearVoice_SE(input_path='assets/clearvoice_samples/speech2.wav', online_write=False)
    myClearVoice_SE.write(output_wav, output_path='outputs/output_MossFormer2_SE_48K_speech2.wav')
    # 执行语音超分辨率
    output_wav = myClearVoice_SR(input_path='outputs/output_MossFormer2_SE_48K_speech2.wav', online_write=False)
    myClearVoice_SR.write(output_wav, output_path='outputs/output_MossFormer2_SR_48K_speech2.wav')

##-----演示三：使用 MossFormer2_SE_48K 模型进行语音增强 -----------------
if False:
    myClearVoice = ClearVoice(task='speech_enhancement', model_names=['MossFormer2_SE_48K'])

    ##第一种调用方法：处理来自输入文件的波形并返回输出波形，然后以相同的音频格式写入 output_MossFormer2_SE_48K_xxx
    output_wav = myClearVoice(input_path='assets/clearvoice_samples/input.wav', online_write=False)
    myClearVoice.write(output_wav, output_path='outputs/output_MossFormer2_SE_48K_input.wav')
    output_wav = myClearVoice(input_path='assets/clearvoice_samples/speech2.wav', online_write=False)
    myClearVoice.write(output_wav, output_path='outputs/output_MossFormer2_SE_48K_speech2.wav')
    output_wav = myClearVoice(input_path='assets/clearvoice_samples/speech1.mp3', online_write=False)
    myClearVoice.write(output_wav, output_path='outputs/output_MossFormer2_SE_48K_speech1.mp3')
    output_wav = myClearVoice(input_path='assets/clearvoice_samples/speech1.flac', online_write=False)
    myClearVoice.write(output_wav, output_path='outputs/output_MossFormer2_SE_48K_speech1.flac')
    output_wav = myClearVoice(input_path='assets/clearvoice_samples/speech1.ogg', online_write=False)
    myClearVoice.write(output_wav, output_path='outputs/output_MossFormer2_SE_48K_speech1.ogg')
    output_wav = myClearVoice(input_path='assets/clearvoice_samples/speech1.wav', online_write=False)
    myClearVoice.write(output_wav, output_path='outputs/output_MossFormer2_SE_48K_speech1.wav')
    output_wav = myClearVoice(input_path='assets/clearvoice_samples/speech2.aac', online_write=False)
    myClearVoice.write(output_wav, output_path='outputs/output_MossFormer2_SE_48K_speech2.aac')
    output_wav = myClearVoice(input_path='assets/clearvoice_samples/speech2.aiff', online_write=False)
    myClearVoice.write(output_wav, output_path='outputs/output_MossFormer2_SE_48K_speech2.aiff')
    
    ##第二种调用方法：处理 'path_to_input_wavs/' 中的所有 wav 文件，并将输出写入 'path_to_output_wavs'
    myClearVoice(input_path='assets/clearvoice_samples/path_to_input_wavs', online_write=True, output_path='outputs/path_to_output_wavs')

    ##第三种调用方法：处理 .scp 文件中列出的 wav 文件，并将输出写入 'path_to_output_wavs_scp/'
    myClearVoice(input_path='assets/clearvoice_samples/scp/audio_samples.scp', online_write=True, output_path='outputs/path_to_output_wavs_scp')
    
##-----演示四：使用 FRCRN_SE_16K 模型进行语音增强 -----------------
if False:
    myClearVoice = ClearVoice(task='speech_enhancement', model_names=['FRCRN_SE_16K'])

    ##第一种调用方法：处理输入波形并返回输出波形，然后写入 output_FRCRN_SE_16K.wav
    output_wav = myClearVoice(input_path='assets/clearvoice_samples/input.wav', online_write=False)
    myClearVoice.write(output_wav, output_path='outputs/output_FRCRN_SE_16K.wav')

    ##第二种调用方法：处理 'path_to_input_wavs/' 中的所有 wav 文件，并将输出写入 'path_to_output_wavs'
    myClearVoice(input_path='assets/clearvoice_samples/path_to_input_wavs', online_write=True, output_path='outputs/path_to_output_wavs')

    ##第三种调用方法：处理 .scp 文件中列出的 wav 文件，并将输出写入 'path_to_output_wavs_scp/'
    myClearVoice(input_path='assets/clearvoice_samples/scp/audio_samples.scp', online_write=True, output_path='outputs/path_to_output_wavs_scp')

##-----演示五：使用 MossFormerGAN_SE_16K 模型进行语音增强 -----------------
if False:
    myClearVoice = ClearVoice(task='speech_enhancement', model_names=['MossFormerGAN_SE_16K'])

    ##第一种调用方法：处理来自 input.wav 的波形并返回输出波形，然后写入 output_MossFormerGAN_SE_16K.wav
    output_wav = myClearVoice(input_path='assets/clearvoice_samples/test.wav', online_write=False)
    myClearVoice.write(output_wav, output_path='outputs/output_MossFormerGAN_SE_16K.wav')

    ##第二种调用方法：处理 'path_to_input_wavs/' 中的所有 wav 文件，并将输出写入 'path_to_output_wavs'
    myClearVoice(input_path='assets/clearvoice_samples/path_to_input_wavs', online_write=True, output_path='outputs/path_to_output_wavs')

    ##第三种调用方法：处理 .scp 文件中列出的 wav 文件，并将输出写入 'path_to_output_wavs_scp/'
    myClearVoice(input_path='assets/clearvoice_samples/scp/audio_samples.scp', online_write=True, output_path='outputs/path_to_output_wavs_scp')

##-----演示六：使用 MossFormer2_SS_16K 模型进行语音分离 -----------------
if True:
    myClearVoice = ClearVoice(task='speech_separation', model_names=['MossFormer2_SS_16K'])

    ##第一种调用方法：处理输入波形并返回输出波形，然后写入 output_MossFormer2_SS_16K_s1.wav 和 output_MossFormer2_SS_16K_s2.wav
    output_wav = myClearVoice(input_path='assets/clearvoice_samples/input_ss.wav', online_write=False)
    myClearVoice.write(output_wav, output_path='outputs/output_MossFormer2_SS_16K.wav')

    #第二种调用方法：处理 'path_to_input_wavs/' 中的所有 wav 文件，并将输出写入 'path_to_output_wavs'
    myClearVoice(input_path='assets/clearvoice_samples/path_to_input_wavs_ss', online_write=True, output_path='outputs/path_to_output_wavs')

    ##第三种调用方法：处理 .scp 文件中列出的 wav 文件，并将输出写入 'path_to_output_wavs_scp/'
    myClearVoice(input_path='assets/clearvoice_samples/scp/audio_samples_mix.scp', online_write=True, output_path='outputs/path_to_output_wavs_scp')

##-----演示七：使用 AV_MossFormer2_TSE_16K 模型进行音视频目标说话人提取 ------
if False:
    myClearVoice = ClearVoice(task='target_speaker_extraction', model_names=['AV_MossFormer2_TSE_16K'])

    #第一种调用方法：处理 'path_to_input_videos/' 中的所有视频文件，并将输出写入 'path_to_output_videos_tse'
    myClearVoice(input_path='assets/clearvoice_samples/path_to_input_videos_tse', online_write=True, output_path='outputs/path_to_output_videos_tse')

    #第二种调用方法：处理 .scp 文件中列出的视频文件，并将输出写入 'path_to_output_videos_tse_scp/'
    myClearVoice(input_path='assets/clearvoice_samples/scp/video_samples.scp', online_write=True, output_path='outputs/path_to_output_videos_tse_scp')
