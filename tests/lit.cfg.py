import lit.formats
import os

config.name = "TensorCompiler"
config.test_format = lit.formats.ShTest()
config.suffixes = [".mlir"]

config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = os.path.join(os.path.dirname(__file__), "..", "build", "tests")

tool_dirs = [
    os.path.join(os.path.dirname(__file__), "..", "build", "tools", "tensor-opt"),
    "/usr/lib/llvm-20/bin",
]
config.environment["PATH"] = ":".join(tool_dirs) + ":" + os.environ.get("PATH", "")

config.substitutions.append(("%tensor-opt", "tensor-opt"))
config.substitutions.append(("%FileCheck", "FileCheck"))
