from clearvoice import ClearVoice
import os
import sys

# 初始化语音增强模型
cv_se = ClearVoice(
    task='speech_enhancement',
    model_names=['MossFormer2_SE_48K']
)

# 检查命令行参数
if len(sys.argv) != 2:
    print("使用方法: python demo_se.py <输入音频文件路径>")
    sys.exit(1)

# 处理单个音频文件
input_path = sys.argv[1]
output_wav = cv_se(
    input_path=input_path,
    online_write=False
)

# 保存增强后的音频到输入文件相同目录
input_dir = os.path.dirname(input_path)
input_filename = os.path.basename(input_path)
output_filename = os.path.splitext(input_filename)[0] + '_enhanced' + os.path.splitext(input_filename)[1]
output_path = os.path.join(input_dir, output_filename)
cv_se.write(output_wav, output_path=output_path)