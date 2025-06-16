import sys
import os
import time
import threading
from pathlib import Path
import numpy as np
import librosa
import soundfile as sf
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from scipy import signal
import io

# 兼容性修复：为Python < 3.11添加Self支持
try:
    from typing import Self
except ImportError:
    try:
        from typing_extensions import Self
    except ImportError:
        from typing import TypeVar
        Self = TypeVar('Self')

from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QLabel, QPushButton, QFileDialog, 
                               QProgressBar, QTextEdit, QGroupBox, QGridLayout,
                               QFrame, QMessageBox, QSplitter, QTabWidget,
                               QScrollArea, QComboBox, QSlider, QCheckBox)
from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtGui import QFont, QIcon, QPalette, QColor, QPixmap

# 延迟导入ClearVoice
try:
    from clearvoice import ClearVoice
except Exception as e:
    print(f"警告：ClearVoice导入时出现问题: {e}")
    ClearVoice = None


class AudioAnalyzer:
    """音频分析工具类"""
    
    @staticmethod
    def load_audio(file_path, sr=None):
        """加载音频文件"""
        try:
            audio, sample_rate = librosa.load(file_path, sr=sr)
            return audio, sample_rate
        except Exception as e:
            raise Exception(f"无法加载音频文件 {file_path}: {str(e)}")
    
    @staticmethod
    def calculate_snr(clean_audio, noisy_audio):
        """计算信噪比"""
        try:
            min_len = min(len(clean_audio), len(noisy_audio))
            clean_audio = clean_audio[:min_len]
            noisy_audio = noisy_audio[:min_len]
            
            noise = noisy_audio - clean_audio
            signal_power = np.mean(clean_audio ** 2)
            noise_power = np.mean(noise ** 2)
            
            if noise_power == 0:
                return float('inf')
            
            snr = 10 * np.log10(signal_power / noise_power)
            return snr
        except Exception:
            return 0
    
    @staticmethod
    def calculate_audio_metrics(original_audio, enhanced_audio, sr):
        """计算音频质量指标"""
        metrics = {}
        
        try:
            # 基本信息
            metrics['duration'] = len(original_audio) / sr
            
            # RMS能量
            metrics['original_rms'] = np.sqrt(np.mean(original_audio ** 2))
            metrics['enhanced_rms'] = np.sqrt(np.mean(enhanced_audio ** 2))
            
            # 峰值
            metrics['original_peak'] = np.max(np.abs(original_audio))
            metrics['enhanced_peak'] = np.max(np.abs(enhanced_audio))
            
            # 动态范围
            metrics['original_dynamic_range'] = 20 * np.log10(np.max(np.abs(original_audio)) / (np.mean(np.abs(original_audio)) + 1e-10))
            metrics['enhanced_dynamic_range'] = 20 * np.log10(np.max(np.abs(enhanced_audio)) / (np.mean(np.abs(enhanced_audio)) + 1e-10))
            
            # 信噪比改善
            snr_improvement = AudioAnalyzer.calculate_snr(enhanced_audio, original_audio)
            metrics['snr_improvement'] = snr_improvement
            
            # 零交叉率
            metrics['original_zcr'] = np.mean(librosa.feature.zero_crossing_rate(original_audio))
            metrics['enhanced_zcr'] = np.mean(librosa.feature.zero_crossing_rate(enhanced_audio))
            
            # 谱质心
            metrics['original_spectral_centroid'] = np.mean(librosa.feature.spectral_centroid(y=original_audio, sr=sr))
            metrics['enhanced_spectral_centroid'] = np.mean(librosa.feature.spectral_centroid(y=enhanced_audio, sr=sr))
            
        except Exception as e:
            print(f"计算音频指标时出错: {e}")
        
        return metrics


class SimplePlotWidget(QLabel):
    """简单的图表显示Widget"""
    def __init__(self):
        super().__init__()
        self.setMinimumSize(600, 400)
        self.setMaximumSize(800, 600)
        self.setStyleSheet("border: 1px solid #ccc; background-color: white;")
        self.setAlignment(Qt.AlignCenter)
        self.setText("等待数据加载...")
        self.setScaledContents(True)
        
    def display_figure(self, figure):
        """显示matplotlib图形"""
        try:
            # 保存图形到内存
            buffer = io.BytesIO()
            figure.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            buffer.seek(0)
            
            # 创建QPixmap并显示
            pixmap = QPixmap()
            pixmap.loadFromData(buffer.getvalue())
            self.setPixmap(pixmap)
            
            buffer.close()
            plt.close(figure)
            
        except Exception as e:
            self.setText(f"图表显示失败: {str(e)}")
    
    def clear_display(self):
        """清空显示"""
        self.clear()
        self.setText("等待数据加载...")


class AudioVisualizationWidget(QWidget):
    """音频可视化Widget"""
    
    def __init__(self):
        super().__init__()
        self.setup_ui()
        self.original_audio = None
        self.enhanced_audio = None
        self.sample_rate = None
        
    def setup_ui(self):
        """设置UI"""
        layout = QVBoxLayout(self)
        
        # 创建标签页
        self.tab_widget = QTabWidget()
        
        # 波形对比标签页
        self.waveform_tab = QWidget()
        self.setup_waveform_tab()
        self.tab_widget.addTab(self.waveform_tab, "波形对比")
        
        # 频谱对比标签页
        self.spectrum_tab = QWidget()
        self.setup_spectrum_tab()
        self.tab_widget.addTab(self.spectrum_tab, "频谱对比")
        
        # 质量指标标签页
        self.metrics_tab = QWidget()
        self.setup_metrics_tab()
        self.tab_widget.addTab(self.metrics_tab, "质量指标")
        
        layout.addWidget(self.tab_widget)
        
    def setup_waveform_tab(self):
        """设置波形对比标签页"""
        layout = QVBoxLayout(self.waveform_tab)
        
        # 创建图表显示区域
        self.waveform_canvas = SimplePlotWidget()
        
        # 添加控制面板
        control_layout = QHBoxLayout()
        
        # 显示选项
        self.show_original_cb = QCheckBox("显示原始音频")
        self.show_original_cb.setChecked(True)
        self.show_original_cb.stateChanged.connect(self.update_waveform_display)
        
        self.show_enhanced_cb = QCheckBox("显示增强音频")
        self.show_enhanced_cb.setChecked(True)
        self.show_enhanced_cb.stateChanged.connect(self.update_waveform_display)
        
        refresh_btn = QPushButton("刷新图表")
        refresh_btn.clicked.connect(self.update_waveform_display)
        
        control_layout.addWidget(self.show_original_cb)
        control_layout.addWidget(self.show_enhanced_cb)
        control_layout.addWidget(refresh_btn)
        control_layout.addStretch()
        
        layout.addLayout(control_layout)
        layout.addWidget(self.waveform_canvas)
        
    def setup_spectrum_tab(self):
        """设置频谱对比标签页"""
        layout = QVBoxLayout(self.spectrum_tab)
        
        # 创建图表显示区域
        self.spectrum_canvas = SimplePlotWidget()
        
        # 添加控制面板
        control_layout = QHBoxLayout()
        
        # 频谱类型选择
        spectrum_label = QLabel("频谱类型:")
        self.spectrum_type = QComboBox()
        self.spectrum_type.addItems(["功率谱密度", "mel频谱"])
        self.spectrum_type.currentTextChanged.connect(self.update_spectrum_display)
        
        refresh_btn = QPushButton("刷新图表")
        refresh_btn.clicked.connect(self.update_spectrum_display)
        
        control_layout.addWidget(spectrum_label)
        control_layout.addWidget(self.spectrum_type)
        control_layout.addWidget(refresh_btn)
        control_layout.addStretch()
        
        layout.addLayout(control_layout)
        layout.addWidget(self.spectrum_canvas)
        
    def setup_metrics_tab(self):
        """设置质量指标标签页"""
        layout = QVBoxLayout(self.metrics_tab)
        
        # 指标显示区域
        self.metrics_text = QTextEdit()
        self.metrics_text.setReadOnly(True)
        self.metrics_text.setStyleSheet("""
            QTextEdit {
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 4px;
                padding: 10px;
                font-family: 'Courier New', monospace;
                font-size: 12px;
            }
        """)
        
        layout.addWidget(self.metrics_text)
        
    def set_audio_data(self, original_path, enhanced_path):
        """设置音频数据"""
        try:
            # 加载音频
            self.original_audio, self.sample_rate = AudioAnalyzer.load_audio(original_path)
            self.enhanced_audio, _ = AudioAnalyzer.load_audio(enhanced_path, sr=self.sample_rate)
            
            # 更新显示
            self.update_waveform_display()
            self.update_spectrum_display()
            self.update_metrics_display()
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载音频数据失败: {str(e)}")
    
    def update_waveform_display(self):
        """更新波形显示"""
        if self.original_audio is None or self.enhanced_audio is None:
            return
            
        try:
            fig = plt.figure(figsize=(12, 8))
            
            # 时间轴
            max_samples = min(len(self.original_audio), len(self.enhanced_audio), 48000)  # 限制显示1秒
            time_axis = np.linspace(0, max_samples / self.sample_rate, max_samples)
            
            # 创建子图
            if self.show_original_cb.isChecked() and self.show_enhanced_cb.isChecked():
                ax1 = fig.add_subplot(211)
                ax2 = fig.add_subplot(212)
                
                # 原始音频
                ax1.plot(time_axis, self.original_audio[:max_samples], 'b-', alpha=0.7, linewidth=0.5)
                ax1.set_title('原始音频波形', fontsize=12, fontweight='bold')
                ax1.set_ylabel('振幅')
                ax1.grid(True, alpha=0.3)
                
                # 增强音频
                ax2.plot(time_axis, self.enhanced_audio[:max_samples], 'r-', alpha=0.7, linewidth=0.5)
                ax2.set_title('增强音频波形', fontsize=12, fontweight='bold')
                ax2.set_xlabel('时间 (秒)')
                ax2.set_ylabel('振幅')
                ax2.grid(True, alpha=0.3)
                
            else:
                ax = fig.add_subplot(111)
                
                if self.show_original_cb.isChecked():
                    ax.plot(time_axis, self.original_audio[:max_samples], 'b-', alpha=0.7, linewidth=0.5, label='原始音频')
                
                if self.show_enhanced_cb.isChecked():
                    ax.plot(time_axis, self.enhanced_audio[:max_samples], 'r-', alpha=0.7, linewidth=0.5, label='增强音频')
                
                ax.set_title('音频波形对比', fontsize=12, fontweight='bold')
                ax.set_xlabel('时间 (秒)')
                ax.set_ylabel('振幅')
                ax.grid(True, alpha=0.3)
                ax.legend()
            
            fig.tight_layout()
            self.waveform_canvas.display_figure(fig)
            
        except Exception as e:
            self.waveform_canvas.setText(f"波形图生成失败: {str(e)}")
    
    def update_spectrum_display(self):
        """更新频谱显示"""
        if self.original_audio is None or self.enhanced_audio is None:
            return
            
        spectrum_type = self.spectrum_type.currentText()
        
        try:
            fig = plt.figure(figsize=(12, 8))
            
            if spectrum_type == "功率谱密度":
                self.plot_power_spectrum(fig)
            elif spectrum_type == "mel频谱":
                self.plot_mel_spectrum(fig)
            
            fig.tight_layout()
            self.spectrum_canvas.display_figure(fig)
            
        except Exception as e:
            self.spectrum_canvas.setText(f"频谱图生成失败: {str(e)}")
    
    def plot_power_spectrum(self, fig):
        """绘制功率谱密度"""
        # 计算功率谱密度
        f1, psd1 = signal.welch(self.original_audio, self.sample_rate, nperseg=1024)
        f2, psd2 = signal.welch(self.enhanced_audio, self.sample_rate, nperseg=1024)
        
        ax = fig.add_subplot(111)
        ax.semilogx(f1, 10 * np.log10(psd1 + 1e-10), 'b-', alpha=0.7, label='原始音频')
        ax.semilogx(f2, 10 * np.log10(psd2 + 1e-10), 'r-', alpha=0.7, label='增强音频')
        ax.set_title('功率谱密度对比', fontsize=12, fontweight='bold')
        ax.set_xlabel('频率 (Hz)')
        ax.set_ylabel('功率谱密度 (dB/Hz)')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    def plot_mel_spectrum(self, fig):
        """绘制mel频谱"""
        # 计算mel频谱
        mel1 = librosa.feature.melspectrogram(y=self.original_audio, sr=self.sample_rate)
        mel2 = librosa.feature.melspectrogram(y=self.enhanced_audio, sr=self.sample_rate)
        
        # 转换为dB
        mel1_db = librosa.power_to_db(mel1, ref=np.max)
        mel2_db = librosa.power_to_db(mel2, ref=np.max)
        
        # 绘制
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        
        im1 = ax1.imshow(mel1_db, aspect='auto', origin='lower', 
                       extent=[0, len(self.original_audio)/self.sample_rate, 0, self.sample_rate/2])
        ax1.set_title('原始音频 Mel频谱图', fontsize=12, fontweight='bold')
        ax1.set_ylabel('频率 (Hz)')
        fig.colorbar(im1, ax=ax1, format='%+2.0f dB')
        
        im2 = ax2.imshow(mel2_db, aspect='auto', origin='lower',
                       extent=[0, len(self.enhanced_audio)/self.sample_rate, 0, self.sample_rate/2])
        ax2.set_title('增强音频 Mel频谱图', fontsize=12, fontweight='bold')
        ax2.set_xlabel('时间 (秒)')
        ax2.set_ylabel('频率 (Hz)')
        fig.colorbar(im2, ax=ax2, format='%+2.0f dB')
    
    def update_metrics_display(self):
        """更新质量指标显示"""
        if self.original_audio is None or self.enhanced_audio is None:
            return
        
        # 计算音频指标
        metrics = AudioAnalyzer.calculate_audio_metrics(
            self.original_audio, self.enhanced_audio, self.sample_rate
        )
        
        # 格式化显示
        metrics_text = "=" * 60 + "\n"
        metrics_text += "                    音频质量分析报告\n"
        metrics_text += "=" * 60 + "\n\n"
        
        # 基本信息
        metrics_text += "📊 基本信息:\n"
        metrics_text += f"   音频时长: {metrics.get('duration', 0):.2f} 秒\n"
        metrics_text += f"   采样率: {self.sample_rate} Hz\n\n"
        
        # 能量分析
        metrics_text += "⚡ 能量分析:\n"
        metrics_text += f"   原始RMS能量: {metrics.get('original_rms', 0):.4f}\n"
        metrics_text += f"   增强RMS能量: {metrics.get('enhanced_rms', 0):.4f}\n"
        energy_change = ((metrics.get('enhanced_rms', 0) / (metrics.get('original_rms', 1) + 1e-10)) - 1) * 100
        metrics_text += f"   能量变化: {energy_change:.2f}%\n\n"
        
        # 峰值分析
        metrics_text += "📈 峰值分析:\n"
        metrics_text += f"   原始峰值: {metrics.get('original_peak', 0):.4f}\n"
        metrics_text += f"   增强峰值: {metrics.get('enhanced_peak', 0):.4f}\n"
        peak_change = ((metrics.get('enhanced_peak', 0) / (metrics.get('original_peak', 1) + 1e-10)) - 1) * 100
        metrics_text += f"   峰值变化: {peak_change:.2f}%\n\n"
        
        # 动态范围
        metrics_text += "📊 动态范围:\n"
        metrics_text += f"   原始动态范围: {metrics.get('original_dynamic_range', 0):.2f} dB\n"
        metrics_text += f"   增强动态范围: {metrics.get('enhanced_dynamic_range', 0):.2f} dB\n"
        dr_improvement = metrics.get('enhanced_dynamic_range', 0) - metrics.get('original_dynamic_range', 0)
        metrics_text += f"   动态范围改善: {dr_improvement:.2f} dB\n\n"
        
        # 信噪比分析
        metrics_text += "🔊 信噪比分析:\n"
        metrics_text += f"   信噪比改善: {metrics.get('snr_improvement', 0):.2f} dB\n\n"
        
        # 频谱分析
        metrics_text += "🎵 频谱分析:\n"
        metrics_text += f"   原始谱质心: {metrics.get('original_spectral_centroid', 0):.2f} Hz\n"
        metrics_text += f"   增强谱质心: {metrics.get('enhanced_spectral_centroid', 0):.2f} Hz\n\n"
        
        # 其他特征
        metrics_text += "🔍 其他特征:\n"
        metrics_text += f"   原始零交叉率: {metrics.get('original_zcr', 0):.4f}\n"
        metrics_text += f"   增强零交叉率: {metrics.get('enhanced_zcr', 0):.4f}\n\n"
        
        # 质量评估
        metrics_text += "🎯 质量评估:\n"
        snr_improvement = metrics.get('snr_improvement', 0)
        if snr_improvement > 10:
            quality_rating = "优秀"
        elif snr_improvement > 5:
            quality_rating = "良好"
        elif snr_improvement > 0:
            quality_rating = "一般"
        else:
            quality_rating = "需要改进"
        
        metrics_text += f"   整体质量: {quality_rating}\n"
        stars = min(5, max(1, int(abs(snr_improvement) / 2)))
        metrics_text += f"   推荐度: {'★' * stars}{'☆' * (5 - stars)}\n"
        
        self.metrics_text.setText(metrics_text)


class AudioEnhancementWorker(QThread):
    """音频增强处理工作线程"""
    progress_updated = Signal(int)
    status_updated = Signal(str)
    finished_processing = Signal(str, dict)
    error_occurred = Signal(str)
    
    def __init__(self, input_path, output_dir=None):
        super().__init__()
        self.input_path = input_path
        self.output_dir = output_dir
        self.cv_se = None
        
    def run(self):
        try:
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
        self.progress_bar.setVisible(False)
        self.process_btn.setEnabled(True)
        self.status_label.setText("处理完成！")
        
        # 保存输出文件路径
        self.output_file_path = output_path
        self.compare_btn.setEnabled(True)
        
        # 显示时间统计
        self.model_load_label.setText(f"模型加载时间: {time_stats['model_load_time']:.2f} 秒")
        self.process_time_label.setText(f"音频处理时间: {time_stats['process_time']:.2f} 秒")
        self.total_time_label.setText(f"总执行时间: {time_stats['total_time']:.2f} 秒")
        self.stats_group.setVisible(True)
        
        self.log_message(f"输出文件已保存: {output_path}")
        self.log_message("处理完成！点击 '对比分析' 按钮查看增强效果")
        
        QMessageBox.information(
            self, 
            "处理完成", 
            f"语音增强处理已完成！\n\n输出文件: {output_path}\n\n点击 '对比分析' 按钮查看增强效果对比"
        )
        
    def start_comparison(self):
        """开始对比分析"""
        if not self.input_file_path or not self.output_file_path:
            QMessageBox.warning(self, "警告", "请先完成音频增强处理！")
            return
        
        try:
            self.log_message("开始加载音频数据进行对比分析...")
            self.status_label.setText("正在分析音频...")
            
            # 设置音频数据到可视化组件
            self.audio_visualization.set_audio_data(self.input_file_path, self.output_file_path)
            
            self.log_message("对比分析完成！")
            self.status_label.setText("对比分析完成")
            
        except Exception as e:
            self.log_message(f"对比分析失败: {str(e)}")
            QMessageBox.critical(self, "错误", f"对比分析失败: {str(e)}")
            
    def on_error_occurred(self, error_message):
        """错误处理回调"""
        self.progress_bar.setVisible(False)
        self.process_btn.setEnabled(True)
        self.status_label.setText("处理失败")
        self.log_message(f"错误: {error_message}")
        QMessageBox.critical(self, "错误", error_message)


def main():
    """主函数"""
    # 检查依赖
    required_packages = ['numpy', 'librosa', 'soundfile', 'matplotlib', 'scipy']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"缺少依赖包: {', '.join(missing_packages)}")
        print("请运行: pip install " + " ".join(missing_packages))
        return
    
    app = QApplication(sys.argv)
    
    # 设置应用程序信息
    app.setApplicationName("LightClear")
    app.setApplicationVersion("2.0")
    app.setOrganizationName("LightClear Team")
    
    # 创建主窗口
    window = SpeechEnhanceGUI()
    window.show()
    
    # 运行应用程序
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
