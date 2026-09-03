"""Which GPUs are present, and what that is allowed to change.

The machine this was written on reports two adapters from two vendors, so the
mixed case is the normal one here rather than an edge case.
"""
from __future__ import annotations

import pytest

from fluid_motion.core import gpu


@pytest.mark.parametrize(
    "name, expected",
    [
        ("NVIDIA GeForce RTX 5070 Ti", gpu.NVIDIA),
        ("NVIDIA GeForce GTX 1660 SUPER", gpu.NVIDIA),
        ("Quadro P2000", gpu.NVIDIA),
        ("AMD Radeon RX 7900 XTX", gpu.AMD),
        ("AMD Radeon(TM) Graphics", gpu.AMD),
        ("Radeon(TM) Vega 8 Graphics", gpu.AMD),
        ("Intel(R) Arc(TM) A770 Graphics", gpu.INTEL),
        ("Intel(R) UHD Graphics 770", gpu.INTEL),
        ("Microsoft Basic Display Adapter", gpu.UNKNOWN),
        ("", gpu.UNKNOWN),
    ],
)
def test_vendor_is_read_off_the_adapter_name(name, expected):
    assert gpu.vendor_of(name) == expected


def test_nvidia_smi_answering_counts_as_an_nvidia_gpu(monkeypatch):
    """The registry walk is the primary source, but it is not the only proof.
    If it ever comes back empty on a machine where nvidia-smi answers, the
    NVIDIA GPU is still there and must not be reasoned away."""
    monkeypatch.setattr(gpu, "detect_adapters", lambda **_kw: [])
    monkeypatch.setattr(gpu, "snapshot", lambda: gpu.GpuSnapshot(name="X", available=True))

    assert gpu.NVIDIA in gpu.available_vendors()


def test_an_unreadable_registry_reports_nothing_rather_than_guessing(monkeypatch):
    """Empty means "could not tell". resolve_backend treats that as "keep
    TensorRT", which is only safe because this returns empty instead of
    inventing a vendor."""
    monkeypatch.setattr(gpu, "detect_adapters", lambda **_kw: [])
    monkeypatch.setattr(gpu, "snapshot", lambda: gpu.GpuSnapshot())

    assert gpu.available_vendors() == set()


def test_unknown_adapters_do_not_become_a_vendor(monkeypatch):
    monkeypatch.setattr(
        gpu, "detect_adapters", lambda **_kw: ["Microsoft Basic Display Adapter"]
    )
    monkeypatch.setattr(gpu, "snapshot", lambda: gpu.GpuSnapshot())

    assert gpu.available_vendors() == set()


def test_detect_adapters_reads_the_real_machine():
    """Not pinned to this box's hardware -- just that the registry walk returns
    plausible strings rather than raising or handing back junk. Every name it
    finds has to be a non-empty string, because vendor_of is fed straight from
    here.
    """
    adapters = gpu.detect_adapters(refresh=True)

    assert isinstance(adapters, list)
    assert all(isinstance(a, str) and a.strip() for a in adapters)
