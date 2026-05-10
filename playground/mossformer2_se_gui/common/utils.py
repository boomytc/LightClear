import numpy as np
import librosa


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
