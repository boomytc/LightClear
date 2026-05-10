import time
from pathlib import Path
import os

from PySide6.QtCore import QThread, Signal

from clearvoice import ClearVoice


class AudioEnhancementWorker(QThread):
    """音频增强处理工作线程"""
    progress_updated = Signal(int)
    status_updated = Signal(str)
    finished_processing = Signal(str, dict)  # 输出路径和时间统计
    error_occurred = Signal(str)
    
    def __init__(self, input_path, output_dir=None):
        super().__init__()
        self.input_path = input_path
        self.output_dir = output_dir
        
    def run(self):
        try:
            total_start_time = time.time()
            
            # 模型加载阶段
            self.status_updated.emit("正在加载语音增强模型...")
            self.progress_updated.emit(20)
            model_load_start_time = time.time()
            
            cv_se = ClearVoice(
                task='speech_enhancement',
                model_names=['MossFormer2_SE_48K']
            )
            
            model_load_time = time.time() - model_load_start_time
            self.status_updated.emit("模型加载完成，开始处理音频...")
            self.progress_updated.emit(50)
            
            # 音频处理阶段
            process_start_time = time.time()
            output_wav = cv_se(
                input_path=self.input_path,
                online_write=False
            )
            process_time = time.time() - process_start_time
            
            self.status_updated.emit("音频处理完成，正在保存文件...")
            self.progress_updated.emit(80)
            
            # 保存文件
            input_path_obj = Path(self.input_path)
            if self.output_dir:
                output_filename = input_path_obj.stem + '_enhanced' + input_path_obj.suffix
                output_path = os.path.join(self.output_dir, output_filename)
            else:
                output_filename = input_path_obj.stem + '_enhanced' + input_path_obj.suffix
                output_path = input_path_obj.parent / output_filename
                
            cv_se.write(output_wav, output_path=str(output_path))
            
            total_time = time.time() - total_start_time
            
            # 时间统计
            time_stats = {
                'model_load_time': model_load_time,
                'process_time': process_time,
                'total_time': total_time
            }
            
            self.progress_updated.emit(100)
            self.status_updated.emit("处理完成！")
            self.finished_processing.emit(str(output_path), time_stats)
            
        except Exception as e:
            self.error_occurred.emit(f"处理过程中发生错误: {str(e)}")
