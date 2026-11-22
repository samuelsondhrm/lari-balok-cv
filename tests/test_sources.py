"""Validation of sources typed into the dashboard's 'Tambah' box. Run: python tests/test_sources.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline import canonical  # noqa: E402


def rejected(s):
    try:
        canonical(s)
    except ValueError:
        return True
    return False


def test_sources():
    assert canonical(" 1 ") == ("1", "kamera") and canonical("01") == ("1", "kamera")
    assert canonical("http://192.168.1.5:8080/video") == ("http://192.168.1.5:8080/video", "stream")
    assert canonical("RTSP://10.0.0.2/live") == ("RTSP://10.0.0.2/live", "stream")
    me = Path(__file__).resolve()
    assert canonical(f'"{me}"') == (str(me), "file")  # Windows "Copy as path" adds quotes
    for bad in ("", "   ", "99", "²", "file:///C:/Windows/win.ini", "ftp://x/y", "concat:a|b", "tidak/ada.mp4", "x" * 600):
        assert rejected(bad), bad


if __name__ == "__main__":
    test_sources()
    print("ok  test_sources")
