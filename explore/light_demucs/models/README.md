# models/

本模块本地 Demucs 权重，按模型分子目录，不进版本库。

```bash
.venv/bin/python scripts/install_model.py htdemucs
.venv/bin/python scripts/install_model.py htdemucs_ft
.venv/bin/python scripts/install_model.py htdemucs_6s
```

分别写入 `htdemucs/`、`htdemucs_ft/`、`htdemucs_6s/`。产品不读取本目录；产品需要时把模型名和目标目录一并传给本脚本再装一份。
