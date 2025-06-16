import sys
import os
import time
import threading
from pathlib import Path

# 兼容性修复：为Python < 3.11添加Self支持
try:
    from typing import Self
except ImportError:
    try:
        from typing_extensions import Self
    except ImportError:
        # 如果typing_extensions也不可用，则创建一个占位符
        from typing import TypeVar
        Self = TypeVar('Self')

from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QLabel, QPushButton, QFileDialog, 
                               QProgressBar, QTextEdit, QGroupBox, QGridLayout,
                               QFrame, QMessageBox, QSplitter)
from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtGui import QFont, QIcon, QPalette, QColor

# 延迟导入ClearVoice，避免在主线程中导入时出现Self类型错误
try:
    from clearvoice import ClearVoice
except Exception as e:
    print(f"警告：ClearVoice导入时出现问题: {e}")
    ClearVoice = None


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
        self.cv_se = None
        
    def run(self):
        try:
            # 在工作线程中导入ClearVoice，避免主线程的类型错误
            global ClearVoice
            if ClearVoice is None:
                try:
                    from clearvoice import ClearVoice
                except Exception as import_error:
                    self.error_occurred.emit(f"无法导入ClearVoice模块: {str(import_error)}")
                    return
            
            total_start_time = time.time()
            
            # 模型加载阶段
            self.status_updated.emit("正在加载语音增强模型...")
            self.progress_updated.emit(20)
            model_load_start_time = time.time()
            
            self.cv_se = ClearVoice(
                task='speech_enhancement',
                model_names=['MossFormer2_SE_48K']
            )
            
            model_load_time = time.time() - model_load_start_time
            self.status_updated.emit("模型加载完成，开始处理音频...")
            self.progress_updated.emit(50)
            
            # 音频处理阶段
            process_start_time = time.time()
            output_wav = self.cv_se(
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
                
            self.cv_se.write(output_wav, output_path=str(output_path))
            
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


class SpeechEnhanceGUI(QMainWindow):
    """语音增强GUI主窗口"""
    
    def __init__(self):
        super().__init__()
        self.worker_thread = None
        self.input_file_path = ""
        self.output_directory = ""
        self.setup_ui()
        self.setup_style()
        
    def setup_ui(self):
        """设置用户界面"""
        self.setWindowTitle("LightClear - 语音增强工具")
        self.setGeometry(100, 100, 900, 400)
        
        # 主widget和布局
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(30, 30, 30, 30)
        
        # 标题
        title_label = QLabel("🎵 LightClear 语音增强工具")
        title_label.setAlignment(Qt.AlignCenter)
        title_font = QFont()
        title_font.setPointSize(24)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setStyleSheet("color: #2c3e50; margin-bottom: 10px;")
        main_layout.addWidget(title_label)
        
        # 分割线
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        line.setStyleSheet("color: #bdc3c7;")
        main_layout.addWidget(line)
        
        # 创建分割器
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
        
        # 左侧面板
        left_panel = self.create_left_panel()
        splitter.addWidget(left_panel)
        
        # 右侧面板
        right_panel = self.create_right_panel()
        splitter.addWidget(right_panel)
        
        # 设置分割器比例
        splitter.setSizes([400, 500])
        
    def create_left_panel(self):
        """创建左侧控制面板"""
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        
        # 文件选择组
        file_group = QGroupBox("📁 文件选择")
        file_layout = QVBoxLayout(file_group)
        
        # 输入文件选择
        input_layout = QHBoxLayout()
        self.input_file_label = QLabel("请选择音频文件")
        self.input_file_label.setStyleSheet("padding: 8px; background-color: #ecf0f1; border-radius: 4px; color: #7f8c8d;")
        self.select_input_btn = QPushButton("选择输入文件")
        self.select_input_btn.clicked.connect(self.select_input_file)
        input_layout.addWidget(self.input_file_label, 2)
        input_layout.addWidget(self.select_input_btn, 1)
        file_layout.addLayout(input_layout)
        
        # 输出目录选择
        output_layout = QHBoxLayout()
        self.output_dir_label = QLabel("默认保存到输入文件目录")
        self.output_dir_label.setStyleSheet("padding: 8px; background-color: #ecf0f1; border-radius: 4px; color: #7f8c8d;")
        self.select_output_btn = QPushButton("选择输出目录")
        self.select_output_btn.clicked.connect(self.select_output_directory)
        output_layout.addWidget(self.output_dir_label, 2)
        output_layout.addWidget(self.select_output_btn, 1)
        file_layout.addLayout(output_layout)
        
        left_layout.addWidget(file_group)
        
        # 处理控制组
        control_group = QGroupBox("🎛️ 处理控制")
        control_layout = QVBoxLayout(control_group)
        
        # 处理按钮
        self.process_btn = QPushButton("🚀 开始语音增强")
        self.process_btn.setEnabled(False)
        self.process_btn.clicked.connect(self.start_processing)
        self.process_btn.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                padding: 12px;
                font-size: 14px;
                font-weight: bold;
                border-radius: 6px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton:disabled {
                background-color: #bdc3c7;
                color: #7f8c8d;
            }
        """)
        control_layout.addWidget(self.process_btn)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        control_layout.addWidget(self.progress_bar)
        
        # 状态标签
        self.status_label = QLabel("准备就绪")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("padding: 8px; background-color: #d5dbdb; border-radius: 4px; color: #2c3e50;")
        control_layout.addWidget(self.status_label)
        
        left_layout.addWidget(control_group)
        
        # 时间统计组
        self.stats_group = QGroupBox("⏱️ 时间统计")
        self.stats_layout = QGridLayout(self.stats_group)
        
        self.model_load_label = QLabel("模型加载时间: --")
        self.process_time_label = QLabel("音频处理时间: --")
        self.total_time_label = QLabel("总执行时间: --")
        
        self.stats_layout.addWidget(self.model_load_label, 0, 0)
        self.stats_layout.addWidget(self.process_time_label, 1, 0)
        self.stats_layout.addWidget(self.total_time_label, 2, 0)
        
        self.stats_group.setVisible(False)
        left_layout.addWidget(self.stats_group)
        
        left_layout.addStretch()
        return left_widget
        
    def create_right_panel(self):
        """创建右侧日志面板"""
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        
        # 日志组
        log_group = QGroupBox("📋 处理日志")
        log_layout = QVBoxLayout(log_group)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("""
            QTextEdit {
                background-color: #2c3e50;
                color: #ecf0f1;
                font-family: 'Courier New', monospace;
                font-size: 12px;
                border: 1px solid #34495e;
                border-radius: 4px;
                padding: 10px;
            }
        """)
        log_layout.addWidget(self.log_text)
        
        # 清空日志按钮
        clear_log_btn = QPushButton("清空日志")
        clear_log_btn.clicked.connect(self.clear_log)
        clear_log_btn.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                color: white;
                border: none;
                padding: 8px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
        """)
        log_layout.addWidget(clear_log_btn)
        
        right_layout.addWidget(log_group)
        return right_widget
        
    def setup_style(self):
        """设置应用程序样式"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f8f9fa;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #bdc3c7;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
                background-color: white;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 8px 0 8px;
                color: #2c3e50;
            }
            QPushButton {
                background-color: #34495e;
                color: white;
                border: none;
                padding: 10px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2c3e50;
            }
            QProgressBar {
                border: 2px solid #bdc3c7;
                border-radius: 5px;
                text-align: center;
                font-weight: bold;
            }
            QProgressBar::chunk {
                background-color: #27ae60;
                border-radius: 3px;
            }
        """)
        
    def log_message(self, message):
        """添加日志消息"""
        current_time = time.strftime("%H:%M:%S")
        formatted_message = f"[{current_time}] {message}"
        self.log_text.append(formatted_message)
        self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())
        
    def clear_log(self):
        """清空日志"""
        self.log_text.clear()
        
    def select_input_file(self):
        """选择输入音频文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "选择音频文件", 
            "", 
            "音频文件 (*.wav *.mp3 *.flac *.m4a *.ogg);;所有文件 (*)"
        )
        
        if file_path:
            self.input_file_path = file_path
            self.input_file_label.setText(os.path.basename(file_path))
            self.input_file_label.setStyleSheet("padding: 8px; background-color: #d5f4e6; border-radius: 4px; color: #27ae60;")
            self.process_btn.setEnabled(True)
            self.log_message(f"已选择输入文件: {file_path}")
            
    def select_output_directory(self):
        """选择输出目录"""
        dir_path = QFileDialog.getExistingDirectory(self, "选择输出目录")
        
        if dir_path:
            self.output_directory = dir_path
            self.output_dir_label.setText(f"输出到: {os.path.basename(dir_path)}")
            self.output_dir_label.setStyleSheet("padding: 8px; background-color: #d5f4e6; border-radius: 4px; color: #27ae60;")
            self.log_message(f"已选择输出目录: {dir_path}")
        else:
            self.output_directory = ""
            self.output_dir_label.setText("默认保存到输入文件目录")
            self.output_dir_label.setStyleSheet("padding: 8px; background-color: #ecf0f1; border-radius: 4px; color: #7f8c8d;")
            
    def start_processing(self):
        """开始处理音频"""
        if not self.input_file_path:
            QMessageBox.warning(self, "警告", "请先选择输入音频文件！")
            return
            
        self.process_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.stats_group.setVisible(False)
        
        self.log_message("开始语音增强处理...")
        self.log_message(f"输入文件: {self.input_file_path}")
        
        # 创建工作线程
        output_dir = self.output_directory if self.output_directory else None
        self.worker_thread = AudioEnhancementWorker(self.input_file_path, output_dir)
        
        # 连接信号
        self.worker_thread.progress_updated.connect(self.update_progress)
        self.worker_thread.status_updated.connect(self.update_status)
        self.worker_thread.finished_processing.connect(self.on_processing_finished)
        self.worker_thread.error_occurred.connect(self.on_error_occurred)
        
        # 启动线程
        self.worker_thread.start()
        
    def update_progress(self, value):
        """更新进度条"""
        self.progress_bar.setValue(value)
        
    def update_status(self, status):
        """更新状态标签"""
        self.status_label.setText(status)
        self.log_message(status)
        
    def on_processing_finished(self, output_path, time_stats):
        """处理完成回调"""
        self.progress_bar.setVisible(False)
        self.process_btn.setEnabled(True)
        self.status_label.setText("处理完成！")
        
        # 显示时间统计
        self.model_load_label.setText(f"模型加载时间: {time_stats['model_load_time']:.2f} 秒")
        self.process_time_label.setText(f"音频处理时间: {time_stats['process_time']:.2f} 秒")
        self.total_time_label.setText(f"总执行时间: {time_stats['total_time']:.2f} 秒")
        self.stats_group.setVisible(True)
        
        self.log_message(f"输出文件已保存: {output_path}")
        self.log_message(f"模型加载时间: {time_stats['model_load_time']:.2f} 秒")
        self.log_message(f"音频处理时间: {time_stats['process_time']:.2f} 秒")
        self.log_message(f"总执行时间: {time_stats['total_time']:.2f} 秒")
        self.log_message("=" * 50)
        
        # 显示完成对话框
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setWindowTitle("处理完成")
        msg.setText("语音增强处理已完成！")
        msg.setInformativeText(f"输出文件: {output_path}")
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec()
        
    def on_error_occurred(self, error_message):
        """错误处理回调"""
        self.progress_bar.setVisible(False)
        self.process_btn.setEnabled(True)
        self.status_label.setText("处理失败")
        
        self.log_message(f"错误: {error_message}")
        
        # 显示错误对话框
        QMessageBox.critical(self, "错误", error_message)


def main():
    """主函数"""
    # 检查并安装必要的依赖
    try:
        import typing_extensions
    except ImportError:
        print("正在安装typing_extensions以支持Python < 3.11...")
        import subprocess
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "typing_extensions"])
            print("typing_extensions安装成功！")
        except subprocess.CalledProcessError:
            print("警告：无法自动安装typing_extensions，可能会遇到类型错误")
    
    app = QApplication(sys.argv)
    
    # 设置应用程序图标和信息
    app.setApplicationName("LightClear")
    app.setApplicationVersion("1.0")
    app.setOrganizationName("LightClear Team")
    
    # 创建主窗口
    window = SpeechEnhanceGUI()
    window.show()
    
    # 运行应用程序
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
