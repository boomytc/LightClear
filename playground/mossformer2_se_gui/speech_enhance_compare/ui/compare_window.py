import os
import time

from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QLabel, QPushButton, QFileDialog, 
                               QProgressBar, QTextEdit, QGroupBox, QFrame, 
                               QMessageBox, QSplitter)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont

from common.worker import AudioEnhancementWorker
from .visualization import AudioVisualizationWidget


class CompareWindow(QMainWindow):
    """语音增强对比分析GUI主窗口"""
    
    def __init__(self):
        super().__init__()
        self.worker_thread = None
        self.input_file_path = ""
        self.output_directory = ""
        self.output_file_path = ""
        self.audio_visualization = None
        self.setup_ui()
        self.setup_style()
        
    def setup_ui(self):
        """设置用户界面"""
        self.setWindowTitle("LightClear - 语音增强对比分析工具")
        self.setGeometry(100, 100, 1200, 800)
        
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(15, 15, 15, 15)
        
        # 标题
        title_label = QLabel("🎵 LightClear 语音增强对比分析工具")
        title_label.setAlignment(Qt.AlignCenter)
        title_font = QFont()
        title_font.setPointSize(18)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setStyleSheet("color: #2c3e50; margin-bottom: 10px;")
        main_layout.addWidget(title_label)
        
        # 分割线
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        main_layout.addWidget(line)
        
        # 主分割器
        main_splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(main_splitter)
        
        # 左侧控制面板
        left_panel = self.create_left_panel()
        main_splitter.addWidget(left_panel)
        
        # 右侧面板
        right_panel = self.create_right_panel()
        main_splitter.addWidget(right_panel)
        
        # 设置分割器比例
        main_splitter.setSizes([300, 900])
        
    def create_left_panel(self):
        """创建左侧控制面板"""
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        
        # 文件选择组
        file_group = QGroupBox("📁 文件选择")
        file_layout = QVBoxLayout(file_group)
        
        # 输入文件选择
        self.input_file_label = QLabel("请选择音频文件")
        self.input_file_label.setStyleSheet("padding: 8px; background-color: #ecf0f1; border-radius: 4px; color: #7f8c8d;")
        self.select_input_btn = QPushButton("选择输入文件")
        self.select_input_btn.clicked.connect(self.select_input_file)
        
        file_layout.addWidget(self.input_file_label)
        file_layout.addWidget(self.select_input_btn)
        
        # 输出目录选择
        self.output_dir_label = QLabel("默认保存到输入文件目录")
        self.output_dir_label.setStyleSheet("padding: 8px; background-color: #ecf0f1; border-radius: 4px; color: #7f8c8d;")
        self.select_output_btn = QPushButton("选择输出目录")
        self.select_output_btn.clicked.connect(self.select_output_directory)
        
        file_layout.addWidget(self.output_dir_label)
        file_layout.addWidget(self.select_output_btn)
        
        left_layout.addWidget(file_group)
        
        # 处理控制组
        control_group = QGroupBox("🎛️ 处理控制")
        control_layout = QVBoxLayout(control_group)
        
        # 处理按钮
        self.process_btn = QPushButton("🚀 开始语音增强")
        self.process_btn.setEnabled(False)
        self.process_btn.clicked.connect(self.start_processing)
        control_layout.addWidget(self.process_btn)
        
        # 对比分析按钮
        self.compare_btn = QPushButton("📊 对比分析")
        self.compare_btn.setEnabled(False)
        self.compare_btn.clicked.connect(self.start_comparison)
        control_layout.addWidget(self.compare_btn)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        control_layout.addWidget(self.progress_bar)
        
        # 状态标签
        self.status_label = QLabel("准备就绪")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("padding: 8px; background-color: #d5dbdb; border-radius: 4px;")
        control_layout.addWidget(self.status_label)
        
        left_layout.addWidget(control_group)
        
        # 时间统计组
        self.stats_group = QGroupBox("⏱️ 时间统计")
        stats_layout = QVBoxLayout(self.stats_group)
        
        self.model_load_label = QLabel("模型加载时间: --")
        self.process_time_label = QLabel("音频处理时间: --")
        self.total_time_label = QLabel("总执行时间: --")
        
        stats_layout.addWidget(self.model_load_label)
        stats_layout.addWidget(self.process_time_label)
        stats_layout.addWidget(self.total_time_label)
        
        self.stats_group.setVisible(False)
        left_layout.addWidget(self.stats_group)
        
        left_layout.addStretch()
        return left_widget
        
    def create_right_panel(self):
        """创建右侧面板"""
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        
        # 日志区域
        log_group = QGroupBox("📋 处理日志")
        log_layout = QVBoxLayout(log_group)
        log_group.setMaximumHeight(150)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(100)
        self.log_text.setStyleSheet("""
            QTextEdit {
                background-color: #2c3e50;
                color: #ecf0f1;
                font-family: 'Courier New', monospace;
                font-size: 10px;
                border: 1px solid #34495e;
                border-radius: 4px;
                padding: 5px;
            }
        """)
        log_layout.addWidget(self.log_text)
        right_layout.addWidget(log_group)
        
        # 音频对比面板
        self.audio_visualization = AudioVisualizationWidget()
        compare_group = QGroupBox("📊 音频对比分析")
        compare_layout = QVBoxLayout(compare_group)
        compare_layout.addWidget(self.audio_visualization)
        right_layout.addWidget(compare_group)
        
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
            QPushButton:disabled {
                background-color: #bdc3c7;
                color: #7f8c8d;
            }
        """)
        
    def log_message(self, message):
        """添加日志消息"""
        current_time = time.strftime("%H:%M:%S")
        formatted_message = f"[{current_time}] {message}"
        self.log_text.append(formatted_message)
        
    def select_input_file(self):
        """选择输入音频文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "选择音频文件", 
            "", 
            "所有文件 (*)"
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
        self.output_file_path = output_path
        self.progress_bar.setVisible(False)
        self.process_btn.setEnabled(True)
        self.compare_btn.setEnabled(True)
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
        
    def on_error_occurred(self, error_message):
        """错误处理回调"""
        self.progress_bar.setVisible(False)
        self.process_btn.setEnabled(True)
        self.status_label.setText("处理失败")
        
        self.log_message(f"错误: {error_message}")
        QMessageBox.critical(self, "错误", error_message)
        
    def start_comparison(self):
        """开始对比分析"""
        if not self.input_file_path or not self.output_file_path:
            QMessageBox.warning(self, "警告", "请先完成语音增强处理！")
            return
            
        self.log_message("开始对比分析...")
        self.audio_visualization.set_audio_data(self.input_file_path, self.output_file_path)
        self.log_message("对比分析完成")

    def closeEvent(self, event):
        """关闭窗口时清理资源"""
        if self.worker_thread and self.worker_thread.isRunning():
            self.worker_thread.terminate()
            self.worker_thread.wait()
        event.accept()
