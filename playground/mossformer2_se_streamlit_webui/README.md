# MossFormer2 SE Streamlit WebUI

This Streamlit variant mirrors the local PySide6 MossFormer2 speech enhancement playground with a browser workflow.

## Run

```bash
uv venv --python 3.12
uv pip install -e ".[streamlit]"
.venv/bin/streamlit run playground/mossformer2_se_streamlit_webui/app.py
```

The app reads local assets and models from the project root. Outputs are written under `outputs/mossformer2_se_streamlit_webui/`.
