"""Headless Streamlit smoke test:  python tests/test_app.py"""
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from streamlit.testing.v1 import AppTest

at = AppTest.from_file(str(ROOT / "app" / "streamlit_app.py"), default_timeout=120).run()
assert not at.exception, at.exception
print("tabs:", len(at.tabs), "| metrics on page:", [m.label for m in at.metric][:6])
at.selectbox[0].select(50).run()
assert not at.exception, at.exception
print("engine 50 ->", [(m.label, m.value) for m in at.metric][:2])
print("APP OK")
