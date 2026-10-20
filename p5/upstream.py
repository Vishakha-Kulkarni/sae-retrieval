"""Load the pinned upstream CL-SR functions and constants without running their CLIs."""
import ast
from pathlib import Path

CL_SR = Path(__file__).resolve().parents[1] / 'upstream/Decoding-Dense-Embeddings/cl-sr'


def upstream_functions(filename):
    """Imports, module constants (k1, k2, b, TOP_K) and functions of cl-sr/<filename>; skips __main__."""
    path = CL_SR / filename
    tree = ast.parse(path.read_text())
    keep = (ast.Import, ast.ImportFrom, ast.Assign, ast.FunctionDef)
    module = ast.Module(body=[n for n in tree.body if isinstance(n, keep)], type_ignores=[])
    namespace = {}
    exec(compile(module, str(path), 'exec'), namespace)
    return namespace
