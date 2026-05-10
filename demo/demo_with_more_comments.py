import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
THIRD_PARTY_DIR = PROJECT_ROOT / "third_party"
if str(THIRD_PARTY_DIR) not in sys.path:
    sys.path.insert(0, str(THIRD_PARTY_DIR))

from clearvoice import ClearVoice  # 导入用于语音处理任务的 ClearVoice 类

if __name__ == '__main__':
    ## ----------------- 演示一：使用单个模型 ----------------------
    if True:  # 此代码块演示如何使用单个模型进行语音增强
        # 初始化 ClearVoice，使用 MossFormer2_SE_48K 模型进行语音增强任务
        myClearVoice = ClearVoice(task='speech_enhancement', model_names=['MossFormer2_SE_48K'])

        # 第一种调用方法： 
        #   处理输入波形并返回增强后的输出波形
        # - input_path (str): 输入带噪音频文件的路径 (input.wav)
        # - output_wav (dict 或 ndarray) : 增强后的输出波形
        output_wav = myClearVoice(input_path='assets/clearvoice_samples/input.wav')
        # 将处理后的波形写入输出文件
        # - output_path (str): 保存增强后音频文件的路径 (output_MossFormer2_SE_48K.wav)
        myClearVoice.write(output_wav, output_path='outputs/output_MossFormer2_SE_48K.wav')

        # 第二种调用方法： 
        #   直接处理并写入音频文件
        # - input_path (str): 输入带噪音频文件目录的路径
        # - online_write (bool): 设置为 True 以在处理过程中直接将增强后的音频保存到文件
        # - output_path (str): 保存增强后输出文件的目录路径
        myClearVoice(input_path='assets/clearvoice_samples/path_to_input_wavs', online_write=True, output_path='outputs/path_to_output_wavs')

        # 第三种调用方法： 
        #   使用 .scp 文件指定输入音频路径
        # - input_path (str): 列出多个音频文件路径的 .scp 文件路径
        # - online_write (bool): 设置为 True 以在处理过程中直接将增强后的音频保存到文件
        # - output_path (str): 保存增强后输出文件的目录路径
        myClearVoice(input_path='assets/clearvoice_samples/scp/audio_samples.scp', online_write=True, output_path='outputs/path_to_output_wavs_scp')


    ## ---------------- 演示二：使用多个模型 -----------------------
    if False:  # 此代码块演示如何使用多个模型进行语音增强
        # 初始化 ClearVoice，使用两个模型进行语音增强任务：MossFormer2_SE_48K 和 FRCRN_SE_16K
        myClearVoice = ClearVoice(task='speech_enhancement', model_names=['MossFormer2_SE_48K', 'FRCRN_SE_16K'])

        # 第一种调用方法：
        #   使用多个模型处理输入波形并返回增强后的输出波形
        # - input_path (str): 输入带噪音频文件的路径 (input.wav)
        # - output_wav (dict 或 ndarray) : 经模型处理后返回的输出波形
        output_wav = myClearVoice(input_path='assets/clearvoice_samples/input.wav')
        # 将处理后的波形写入输出文件
        # - output_path (str): 保存增强后音频文件的目录路径，使用与输入相同的文件名 (input.wav)
        myClearVoice.write(output_wav, output_path='outputs/path_to_output_wavs')

        # 第二种调用方法：
        #   使用多个模型直接处理并写入音频文件
        # - input_path (str): 输入带噪音频文件目录的路径
        # - online_write (bool): 设置为 True 以在处理过程中直接将增强后的音频保存到文件
        # - output_path (str): 保存增强后输出文件的目录路径
        myClearVoice(input_path='assets/clearvoice_samples/path_to_input_wavs', online_write=True, output_path='outputs/path_to_output_wavs')

        # 第三种调用方法：
        #   使用 .scp 文件为多个模型指定输入音频路径
        # - input_path (str): 列出多个音频文件路径的 .scp 文件路径
        # - online_write (bool): 设置为 True 以在处理过程中直接将增强后的音频保存到文件
        # - output_path (str): 保存增强后输出文件的目录路径
        myClearVoice(input_path='assets/clearvoice_samples/scp/audio_samples.scp', online_write=True, output_path='outputs/path_to_output_wavs_scp')
