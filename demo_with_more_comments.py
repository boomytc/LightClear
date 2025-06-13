from clearvoice import ClearVoice  # 导入ClearVoice类用于语音处理任务

if __name__ == '__main__':
    ## ----------------- 示例一：使用单个模型 ----------------------
    if True:  # 此代码块演示如何使用单个模型进行语音增强
        # 初始化ClearVoice用于语音增强任务，使用MossFormer2_SE_48K模型
        myClearVoice = ClearVoice(task='speech_enhancement', model_names=['MossFormer2_SE_48K'])

        # 第一种调用方法：
        #   处理输入波形并返回增强后的输出波形
        # - input_path (str): 输入含噪音频文件的路径 (input.wav)
        # - output_wav (dict or ndarray): 增强后的输出波形
        output_wav = myClearVoice(input_path='samples/input.wav')
        # 将处理后的波形写入输出文件
        # - output_path (str): 保存增强音频文件的路径 (output_MossFormer2_SE_48K.wav)
        myClearVoice.write(output_wav, output_path='samples/output_MossFormer2_SE_48K.wav')

        # 第二种调用方法：
        #   直接处理并写入音频文件
        # - input_path (str): 输入含噪音频文件目录的路径
        # - online_write (bool): 设置为True以在处理过程中直接保存增强的音频到文件
        # - output_path (str): 保存增强输出文件的目录路径
        myClearVoice(input_path='samples/path_to_input_wavs', online_write=True, output_path='samples/path_to_output_wavs')

        # 第三种调用方法：
        #   使用.scp文件指定输入音频路径
        # - input_path (str): 包含多个音频文件路径的.scp文件路径
        # - online_write (bool): 设置为True以在处理过程中直接保存增强的音频到文件
        # - output_path (str): 保存增强输出文件的目录路径
        myClearVoice(input_path='samples/scp/audio_samples.scp', online_write=True, output_path='samples/path_to_output_wavs_scp')


    ## ---------------- 示例二：使用多个模型 -----------------------
    if False:  # 此代码块演示如何使用多个模型进行语音增强
        # 初始化ClearVoice用于语音增强任务，使用两个模型：MossFormer2_SE_48K和FRCRN_SE_16K
        myClearVoice = ClearVoice(task='speech_enhancement', model_names=['MossFormer2_SE_48K', 'FRCRN_SE_16K'])

        # 第一种调用方法：
        #   使用多个模型处理输入波形并返回增强后的输出波形
        # - input_path (str): 输入含噪音频文件的路径 (input.wav)
        # - output_wav (dict or ndarray): 经过模型处理后返回的输出波形
        output_wav = myClearVoice(input_path='samples/input.wav')
        # 将处理后的波形写入输出文件
        # - output_path (str): 保存增强音频文件的目录路径，使用与输入相同的文件名 (input.wav)
        myClearVoice.write(output_wav, output_path='samples/path_to_output_wavs')

        # 第二种调用方法：
        #   使用多个模型直接处理并写入音频文件
        # - input_path (str): 输入含噪音频文件目录的路径
        # - online_write (bool): 设置为True以在处理过程中直接保存增强的音频到文件
        # - output_path (str): 保存增强输出文件的目录路径
        myClearVoice(input_path='samples/path_to_input_wavs', online_write=True, output_path='samples/path_to_output_wavs')

        # 第三种调用方法：
        #   使用.scp文件为多个模型指定输入音频路径
        # - input_path (str): 包含多个音频文件路径的.scp文件路径
        # - online_write (bool): 设置为True以在处理过程中直接保存增强的输出
        # - output_path (str): 保存增强输出文件的目录路径
        myClearVoice(input_path='samples/scp/audio_samples.scp', online_write=True, output_path='samples/path_to_output_wavs_scp')
