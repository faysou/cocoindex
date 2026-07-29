from __future__ import annotations

from pathlib import Path

import cocoindex as coco
import numpy as np
from cocoindex.connectors import turboquant
from cocoindex.connectors.turboquant._target import (
    _vector_fingerprint,
)

from tests import common


def test_turboquant_target_add_update_delete(tmp_path: Path) -> None:
    index_path = tmp_path / "index.tvim"
    state = {
        1: np.eye(8, dtype=np.float32)[0],
        2: np.eye(8, dtype=np.float32)[1],
    }

    async def flow() -> None:
        target = await turboquant.mount_index_target(index_path, bit_width=4)
        for id_, vector in state.items():
            target.declare_vector(id=id_, vector=vector)

    app = coco.App(
        coco.AppConfig(
            name="test_turboquant_target_add_update_delete",
            environment=common.create_test_env(__file__),
        ),
        flow,
    )

    app.update_blocking()
    assert index_path.read_bytes()[:5] == b"TVIM\x06"
    assert [result.id for result in turboquant.search(index_path, state[1], 2)] == [
        1,
        2,
    ]

    del state[2]
    state[3] = np.eye(8, dtype=np.float32)[2]

    app.update_blocking()
    assert index_path.read_bytes()[:5] == b"TVIM\x06"
    assert [result.id for result in turboquant.search(index_path, state[3], 2)] == [
        3,
        1,
    ]
    assert [
        result.id
        for result in turboquant.search(index_path, state[3], 2, allowlist=[3, 999])
    ] == [3]
    assert turboquant.search(index_path, state[3], 2, allowlist=[2, 999]) == []


def test_turboquant_vector_fingerprint_is_compact() -> None:
    vector = np.eye(1536, dtype=np.float32)[0]

    fingerprint = _vector_fingerprint(vector)

    assert len(fingerprint) == 16


def test_turboquant_target_removes_index(tmp_path: Path) -> None:
    index_path = tmp_path / "index.tvim"
    enabled = True

    async def flow() -> None:
        if enabled:
            target = await turboquant.mount_index_target(index_path, bit_width=4)
            target.declare_vector(id=1, vector=np.eye(8, dtype=np.float32)[0])

    app = coco.App(
        coco.AppConfig(
            name="test_turboquant_target_removes_index",
            environment=common.create_test_env(__file__),
        ),
        flow,
    )

    app.update_blocking()
    assert index_path.exists()

    enabled = False
    app.update_blocking()
    assert not index_path.exists()
