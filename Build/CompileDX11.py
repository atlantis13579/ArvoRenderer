"""Invoke the build-only DX11 compiler module; no application helper is emitted."""
import ctypes
from pathlib import Path
import sys

if len(sys.argv) != 4 or sys.argv[3] not in ("0", "1"):
    raise SystemExit("Usage: CompileDX11.py <compiler-module> <output-header> <debug:0|1>")

compiler = ctypes.CDLL(str(Path(sys.argv[1]).resolve()))
compile_shader = compiler.CompileDX11
compile_shader.argtypes = [ctypes.c_wchar_p, ctypes.c_int]
compile_shader.restype = ctypes.c_int
raise SystemExit(compile_shader(str(Path(sys.argv[2]).resolve()), int(sys.argv[3])))
