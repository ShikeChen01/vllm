# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the autouse compile-cache isolation fixtures in tests/conftest.py.

Some tests below are order-dependent on purpose: they check that state left
behind by the previous test does not reach the next one.
"""

import os
import subprocess
import sys

import pytest

from vllm import envs

pytestmark = pytest.mark.skip_global_cleanup

_seen_cache_roots: list[str] = []


@pytest.mark.parametrize("run", range(2))
def test_cache_root_is_fresh_per_test(run, isolated_compile_cache):
    cache_root = isolated_compile_cache
    assert os.environ["VLLM_CACHE_ROOT"] == cache_root
    assert cache_root == envs.VLLM_CACHE_ROOT
    assert os.environ["TORCHINDUCTOR_CACHE_DIR"].startswith(cache_root)
    assert not os.path.exists(os.path.join(cache_root, "torch_compile_cache"))
    assert cache_root not in _seen_cache_roots
    for previous in _seen_cache_roots:
        assert not os.path.exists(previous)
    _seen_cache_roots.append(cache_root)
    os.makedirs(os.path.join(cache_root, "torch_compile_cache"))


def test_subprocess_inherits_cache_root(isolated_compile_cache):
    out = subprocess.check_output(
        [
            sys.executable,
            "-c",
            "import os; print(os.environ['VLLM_CACHE_ROOT'])",
        ],
        text=True,
    )
    assert out.strip() == isolated_compile_cache


def test_non_compile_caches_stay_shared(isolated_compile_cache, _session_compile_cache):
    shared_cache_root, shared_env = _session_compile_cache
    assert os.environ["TRITON_CACHE_DIR"] == shared_env["TRITON_CACHE_DIR"]
    assert os.environ["DG_JIT_CACHE_DIR"] == shared_env["DG_JIT_CACHE_DIR"]
    assert not os.environ["TRITON_CACHE_DIR"].startswith(isolated_compile_cache)
    modelinfos = os.path.join(isolated_compile_cache, "modelinfos")
    if not os.path.islink(modelinfos):
        pytest.skip("symlinks are not supported here")
    assert os.path.realpath(modelinfos) == os.path.realpath(
        os.path.join(shared_cache_root, "modelinfos")
    )


def test_leak_env_from_previous_test(isolated_compile_cache):
    # What compilation does via os.environ (decorators.py, compiler_interface.py)
    # and what an in-process EngineCore does via enable_envs_cache().
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = "/leaked/inductor_cache"
    os.environ["TRITON_CACHE_DIR"] = "/leaked/triton_cache"
    os.environ["DG_JIT_CACHE_DIR"] = "/leaked/deep_gemm"
    envs.enable_envs_cache()


def test_env_is_restored_after_previous_test(
    isolated_compile_cache, _session_compile_cache
):
    _, shared_env = _session_compile_cache
    assert not os.environ["TORCHINDUCTOR_CACHE_DIR"].startswith("/leaked")
    for key, value in shared_env.items():
        assert os.environ[key] == value
    assert isolated_compile_cache == envs.VLLM_CACHE_ROOT
