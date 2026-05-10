import sys
from pathlib import Path

# 添加项目根目录到Python路径
current_dir = Path(__file__).absolute().parent
app_root = current_dir.parent
project_root = current_dir.parents[2]
third_party_dir = project_root / "third_party"

for path in (app_root, project_root, third_party_dir):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from PySide6.QtWidgets import QApplication

from ui.compare_window import CompareWindow


def main():
    """主函数"""
    app = QApplication(sys.argv)
    
    # 设置应用程序信息
    app.setApplicationName("LightClear Compare")
    
    # 创建主窗口
    window = CompareWindow()
    window.show()
    
    # 运行应用程序
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
