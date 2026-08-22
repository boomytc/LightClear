# models/

本产品本地 Demucs 权重，不进版本库，不读取 explore。按模型分子目录：`htdemucs`、`htdemucs_ft`、`htdemucs_6s`。

```bash
cd ../../explore/light_demucs
.venv/bin/python scripts/install_model.py htdemucs ../../products/vocal_isolate_web/models/htdemucs
.venv/bin/python scripts/install_model.py htdemucs_ft ../../products/vocal_isolate_web/models/htdemucs_ft
.venv/bin/python scripts/install_model.py htdemucs_6s ../../products/vocal_isolate_web/models/htdemucs_6s
```
