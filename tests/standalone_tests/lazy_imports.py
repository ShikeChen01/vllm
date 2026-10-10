# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Description: Test the lazy import module
# The utility function cannot be placed in `vllm.utils`
# this needs to be a standalone script
import subprocess
import sys

# The engine core, workers and `vllm.config` are imported by every process
# that `vllm serve` spawns, so they must not depend on the HTTP layer.
# Each entry point is imported in a fresh interpreter so that the checks
# cannot hide (or cause) leaks in one another.
engine_side_modules = ["vllm.config", "vllm.v1.engine.core"]
http_layer_modules = ["fastapi", "openai", "openai_harmony", "vllm.entrypoints"]

CHECK_SCRIPT = """
import importlib, sys
importlib.import_module(sys.argv[1])
forbidden = sys.argv[2:]
leaked = sorted(
    name for name in sys.modules
    if any(name == m or name.startswith(m + ".") for m in forbidden)
)
print("LEAKED:" + ",".join(leaked))
"""

failures = []
for engine_module in engine_side_modules:
    result = subprocess.run(
        [sys.executable, "-c", CHECK_SCRIPT, engine_module, *http_layer_modules],
        capture_output=True,
        text=True,
    )
    # Only the last line is ours: vLLM may log to stdout while importing.
    last_line = (result.stdout.strip().splitlines() or [""])[-1]
    if result.returncode != 0 or not last_line.startswith("LEAKED:"):
        failures.append(f"`import {engine_module}` failed:\n{result.stderr}")
    elif leaked := last_line.removeprefix("LEAKED:"):
        failures.append(
            f"`import {engine_module}` imported the HTTP layer: {leaked}. "
            f"Run `python -X importtime -c 'import {engine_module}'` "
            "to find the offending import."
        )
if failures:
    raise AssertionError("\n\n".join(failures))

# List of modules that should not be imported too early.
# Lazy import `torch._inductor.async_compile` to avoid creating
# too many processes before we set the number of compiler threads.
# Lazy import `cv2` to avoid bothering users who only use text models.
# `cv2` can easily mess up the environment.
module_names = ["torch._inductor.async_compile", "cv2"]

# set all modules in `module_names` to be None.
# if we import any modules during `import vllm`, there would be a
# hard error and nice stacktrace on the first import.
for module_name in module_names:
    sys.modules[module_name] = None

import vllm  # noqa
