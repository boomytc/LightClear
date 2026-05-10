import os
import io
import numpy as np
import librosa
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import signal
import matplotlib.font_manager as fm

from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                               QPushButton, QTabWidget, QComboBox, QCheckBox,
                               QTableWidget, QTableWidgetItem, QMessageBox)
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap

from common.utils import AudioAnalyzer

# 配置matplotlib字体
ASSETS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', 'assets')
font_path = os.path.join(ASSETS_DIR, 'fonts', 'SimHei.ttf')
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    font_prop = fm.FontProperties(fname=font_path)
    plt.rcParams['font.sans-serif'] = [font_prop.get_name()]
    plt.rcParams['axes.unicode_minus'] = False


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
            buffer = io.BytesIO()
            figure.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            buffer.seek(0)
            
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
        
        self.waveform_canvas = SimplePlotWidget()
        
        control_layout = QHBoxLayout()
        
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
        
        self.spectrum_canvas = SimplePlotWidget()
        
        control_layout = QHBoxLayout()
        
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
        
        self.metrics_table = QTableWidget()
        self.metrics_table.setColumnCount(5)
        self.metrics_table.setHorizontalHeaderLabels(['指标类别', '原始音频', '增强音频', '变化量', '变化说明'])
        
        self.metrics_table.setStyleSheet("""
            QTableWidget {
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 4px;
                gridline-color: #dee2e6;
            }
            QTableWidget::item {
                padding: 8px;
                border-bottom: 1px solid #dee2e6;
            }
            QHeaderView::section {
                background-color: #e9ecef;
                border: 1px solid #dee2e6;
                padding: 8px;
                font-weight: bold;
            }
        """)
        
        self.metrics_table.setAlternatingRowColors(True)
        self.metrics_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.metrics_table.verticalHeader().setVisible(False)
        self.metrics_table.horizontalHeader().setStretchLastSection(True)
        
        layout.addWidget(self.metrics_table)
        
    def set_audio_data(self, original_path, enhanced_path):
        """设置音频数据"""
        try:
            self.original_audio, self.sample_rate = AudioAnalyzer.load_audio(original_path)
            self.enhanced_audio, _ = AudioAnalyzer.load_audio(enhanced_path, sr=self.sample_rate)
            
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
            
            max_samples = min(len(self.original_audio), len(self.enhanced_audio), 48000)
            time_axis = np.linspace(0, max_samples / self.sample_rate, max_samples)
            
            if self.show_original_cb.isChecked() and self.show_enhanced_cb.isChecked():
                ax1 = fig.add_subplot(211)
                ax2 = fig.add_subplot(212)
                
                ax1.plot(time_axis, self.original_audio[:max_samples], 'b-', alpha=0.7, linewidth=0.5)
                ax1.set_title('原始音频波形', fontsize=12, fontweight='bold')
                ax1.set_ylabel('振幅')
                ax1.grid(True, alpha=0.3)
                
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
        mel1 = librosa.feature.melspectrogram(y=self.original_audio, sr=self.sample_rate)
        mel2 = librosa.feature.melspectrogram(y=self.enhanced_audio, sr=self.sample_rate)
        
        mel1_db = librosa.power_to_db(mel1, ref=np.max)
        mel2_db = librosa.power_to_db(mel2, ref=np.max)
        
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
        
        metrics = AudioAnalyzer.calculate_audio_metrics(
            self.original_audio, self.enhanced_audio, self.sample_rate
        )
        
        table_data = []
        
        # 基本信息
        table_data.append(('音频时长 (秒)', f"{metrics.get('duration', 0):.2f}", f"{metrics.get('duration', 0):.2f}", "无变化", "时长保持不变"))
        table_data.append(('采样率 (Hz)', f"{self.sample_rate}", f"{self.sample_rate}", "无变化", "采样率保持不变"))
        
        # 能量分析
        original_rms = metrics.get('original_rms', 0)
        enhanced_rms = metrics.get('enhanced_rms', 0)
        energy_change = ((enhanced_rms / (original_rms + 1e-10)) - 1) * 100 if original_rms > 0 else 0
        energy_desc = "能量适度降低，噪声抑制良好" if energy_change < -30 else "能量略有降低，噪声得到控制" if energy_change < 0 else "能量增加，可能增强了信号强度"
        table_data.append(('RMS能量', f"{original_rms:.4f}", f"{enhanced_rms:.4f}", f"{energy_change:+.2f}%", energy_desc))
        
        # 峰值分析
        original_peak = metrics.get('original_peak', 0)
        enhanced_peak = metrics.get('enhanced_peak', 0)
        peak_change = ((enhanced_peak / (original_peak + 1e-10)) - 1) * 100 if original_peak > 0 else 0
        peak_desc = "峰值显著降低，噪声峰值被有效抑制" if peak_change < -30 else "峰值适度降低，信号更加平滑" if peak_change < 0 else "峰值增加，信号得到增强"
        table_data.append(('峰值', f"{original_peak:.4f}", f"{enhanced_peak:.4f}", f"{peak_change:+.2f}%", peak_desc))
        
        # 动态范围
        original_dr = metrics.get('original_dynamic_range', 0)
        enhanced_dr = metrics.get('enhanced_dynamic_range', 0)
        dr_improvement = enhanced_dr - original_dr
        dr_desc = "动态范围显著改善，音频层次更丰富" if dr_improvement > 2 else "动态范围略有改善，音质有所提升" if dr_improvement > 0 else "动态范围基本保持，处理较为保守"
        table_data.append(('动态范围 (dB)', f"{original_dr:.2f}", f"{enhanced_dr:.2f}", f"{dr_improvement:+.2f} dB", dr_desc))
        
        # 信噪比分析
        snr_improvement = metrics.get('snr_improvement', 0)
        snr_desc = "信噪比显著改善，噪声抑制效果优秀" if snr_improvement > 5 else "信噪比适度改善，噪声得到良好控制" if snr_improvement > 0 else "信噪比略有下降，可能过度处理" if snr_improvement < -2 else "信噪比基本保持"
        table_data.append(('信噪比改善 (dB)', '-', f"{snr_improvement:.2f}", f"{snr_improvement:+.2f} dB", snr_desc))
        
        # 频谱分析
        original_centroid = metrics.get('original_spectral_centroid', 0)
        enhanced_centroid = metrics.get('enhanced_spectral_centroid', 0)
        centroid_change = enhanced_centroid - original_centroid
        centroid_desc = "频谱质心上移，高频成分增强，清晰度提升" if centroid_change > 100 else "频谱质心略有上移，音质略有改善" if centroid_change > 0 else "频谱质心下移，高频成分减少，可能过度滤波"
        table_data.append(('谱质心 (Hz)', f"{original_centroid:.2f}", f"{enhanced_centroid:.2f}", f"{centroid_change:+.2f} Hz", centroid_desc))
        
        # 其他特征
        original_zcr = metrics.get('original_zcr', 0)
        enhanced_zcr = metrics.get('enhanced_zcr', 0)
        zcr_change = ((enhanced_zcr / (original_zcr + 1e-10)) - 1) * 100 if original_zcr > 0 else 0
        zcr_desc = "零交叉率显著降低，噪声和毛刺明显减少" if zcr_change < -30 else "零交叉率适度降低，信号更加平滑" if zcr_change < 0 else "零交叉率基本保持，信号特征稳定"
        table_data.append(('零交叉率', f"{original_zcr:.4f}", f"{enhanced_zcr:.4f}", f"{zcr_change:+.2f}%", zcr_desc))
        
        self.metrics_table.setRowCount(len(table_data))
        
        for row, (metric_name, original_value, enhanced_value, change_value, description) in enumerate(table_data):
            self.metrics_table.setItem(row, 0, QTableWidgetItem(metric_name))
            self.metrics_table.setItem(row, 1, QTableWidgetItem(original_value))
            self.metrics_table.setItem(row, 2, QTableWidgetItem(enhanced_value))
            self.metrics_table.setItem(row, 3, QTableWidgetItem(change_value))
            self.metrics_table.setItem(row, 4, QTableWidgetItem(description))
        
        self.metrics_table.resizeColumnsToContents()
        self.metrics_table.horizontalHeader().setStretchLastSection(True)
