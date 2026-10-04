from pathlib import Path

import streamlit.components.v1 as components

_FRONTEND_DIR = Path(__file__).resolve().parent / "pixel_canvas_frontend"

_pixel_canvas = components.declare_component("pixel_canvas", path=str(_FRONTEND_DIR))


def pixel_canvas(key, size=28, cell_size=14):
    return _pixel_canvas(size=size, cell_size=cell_size, key=key, default=None)
