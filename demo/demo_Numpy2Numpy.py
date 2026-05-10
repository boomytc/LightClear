import sys
from pathlib import Path

import soundfile as sf
import numpy as np
import librosa

PROJECT_ROOT = Path(__file__).resolve().parents[1]
THIRD_PARTY_DIR = PROJECT_ROOT / "third_party"
if str(THIRD_PARTY_DIR) not in sys.path:
    sys.path.insert(0, str(THIRD_PARTY_DIR))

from clearvoice import ClearVoice
 
""" 
此演示展示了 ClearVoice 提供从 numpy 输入到 numpy 输出的批量处理功能
"""
use_case_list = ['MossFormer2_SR_48K', 'MossFormer2_SE_48K', 'FRCRN_SE_16K', 'MossFormerGAN_SE_16K', 'MossFormer2_SS_16K']

use_case = 'MossFormer2_SR_48K'

if use_case not in use_case_list:
    print(f'错误：不支持的 use_case: {use_case}')
    use_case_enabled = False
else:
    use_case_enabled = True

##-----演示一：使用 MossFormer2_SR_48K 模型进行语音超分辨率 -----------------
if use_case_enabled and use_case == 'MossFormer2_SR_48K':
    print('testing MossFormer2_SR_48K ...')
    myClearVoice = ClearVoice(task='speech_super_resolution', model_names=['MossFormer2_SR_48K'])

    audio, sr = sf.read('assets/clearvoice_samples/input_sr.wav')
    ## 输入音频必须为 48000 Hz
    if sr != 48000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=48000)
    if len(audio.shape) < 2:
        audio = np.reshape(audio, [1, audio.shape[0]])
    ## 模拟批量输入
    #audio = np.concatenate((audio, audio), axis=0)
    audio = audio.astype(np.float32)    
    ## audio: [批次, 长度]
    ##output_wav: [批次, 长度]
    output_wav = myClearVoice(audio, False)
    sf.write('outputs/output_MossFormer2_SR_48K_input_sr.wav', output_wav[0,:], 48000) 

##-----演示二：使用 MossFormer2_SE_48K 模型进行语音增强 -----------------
if use_case_enabled and use_case == 'MossFormer2_SE_48K':
    print(f'testing MossFormer2_SE_48K ...')
    myClearVoice = ClearVoice(task='speech_enhancement', model_names=['MossFormer2_SE_48K'])

    audio, sr = sf.read('assets/clearvoice_samples/input.wav')
    ## 输入音频必须为 48000 Hz
    if sr != 48000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=48000)
    if len(audio.shape) < 2:
        audio = np.reshape(audio, [1, audio.shape[0]])
    audio = audio.astype(np.float32)
    ## audio: [批次, 长度]
    ##output_wav: [批次, 长度]
    output_wav = myClearVoice(audio, False)
    sf.write('outputs/output_MossFormer2_SE_48K_batch.wav', output_wav[0,:], 48000)    
      
##-----演示三：使用 FRCRN_SE_16K 模型进行语音增强 -----------------
if use_case_enabled and use_case == 'FRCRN_SE_16K':
    print('testing FRCRN_SE_16K ...')
    myClearVoice = ClearVoice(task='speech_enhancement', model_names=['FRCRN_SE_16K'])

    ##第一种调用方法：处理输入波形并返回输出波形，然后写入 output_FRCRN_SE_16K.wav
    audio, sr = sf.read('assets/clearvoice_samples/input.wav')
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    if len(audio.shape) < 2:
        audio = np.reshape(audio, [1, audio.shape[0]])
    #audio = np.concatenate((audio, audio), axis=0)
    audio = audio.astype(np.float32)
    ## audio: [批次, 长度]
    ##output_wav: [批次, 长度]
    output_wav = myClearVoice(audio, False)
    sf.write('outputs/output_FRCRN_SE_16K_batch.wav', output_wav[0, :], 16000)

##-----演示四：使用 MossFormerGAN_SE_16K 模型进行语音增强 -----------------
if use_case_enabled and use_case == 'MossFormerGAN_SE_16K':
    print(f'testing MossFormerGAN_SE_16K ...')
    myClearVoice = ClearVoice(task='speech_enhancement', model_names=['MossFormerGAN_SE_16K'])

    ##第一种调用方法：处理来自 input.wav 的波形并返回输出波形，然后写入 output_MossFormerGAN_SE_16K.wav
    audio, sr = sf.read('assets/clearvoice_samples/input.wav')
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    if len(audio.shape) < 2:
        audio = np.reshape(audio, [1, audio.shape[0]])
    #audio = np.concatenate((audio, audio), axis=0)
    audio = audio.astype(np.float32)
    ## audio: [批次, 长度]
    ##output_wav: [批次, 长度]
    output_wav = myClearVoice(audio, False)
    sf.write('outputs/output_MossFormerGAN_SE_16K_batch.wav', output_wav[0, :], sr)

##-----演示五：使用 MossFormer2_SS_16K 模型进行语音分离 -----------------
if use_case_enabled and use_case == 'MossFormer2_SS_16K':
    print(f'testing MossFormer2_SS_16K ...')
    myClearVoice = ClearVoice(task='speech_separation', model_names=['MossFormer2_SS_16K'])

    ##第一种调用方法：处理输入波形并返回输出波形，然后写入 output_MossFormer2_SS_16K_s1.wav 和 output_MossFormer2_SS_16K_s2.wav
    num_spks = 2
    audio, sr = sf.read('assets/clearvoice_samples/input_ss.wav')
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    if len(audio.shape) < 2:
        audio = np.reshape(audio, [1, audio.shape[0]])
    audio = audio.astype(np.float32)
    ## audio: [批次, 长度]
    ## output_wav: [说话人, 批次, 长度]
    output_wav = myClearVoice(audio, False)
    for spk in range(num_spks):
        output_file = f'outputs/output_MossFormer2_SS_16K_batch_spk{spk+1}.wav'
        sf.write(output_file, output_wav[spk, 0, :], 16000)
