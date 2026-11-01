"""JIT cache paths must remain usable after native libraries change locale."""

import locale

from faceAppearance.vendor.wildCap.backend import writeUtf8IfChanged


def testJitCacheWithChinesePathsUnderCLocale(tmp_path):
    path = tmp_path / "构建.ninja"
    content = "source = /home/用户/项目/render.cpp\n"
    previousLocale = locale.setlocale(locale.LC_CTYPE)
    try:
        locale.setlocale(locale.LC_CTYPE, "C")
        writeUtf8IfChanged(path, content)
        previousMtime = path.stat().st_mtime_ns
        writeUtf8IfChanged(path, content)
        assert path.stat().st_mtime_ns == previousMtime
        writeUtf8IfChanged(path, content + "changed = yes\n")
    finally:
        locale.setlocale(locale.LC_CTYPE, previousLocale)
    assert path.read_text(encoding="utf-8") == content + "changed = yes\n"
