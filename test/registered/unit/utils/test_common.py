import unittest
from array import array
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.utils.common import (
    _get_fastapi_request_path,
    flatten_arrays_to_int64_tensor,
)
from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=5, stage="base-b", runner_config="1-gpu-small")
register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _has_iter_route_contexts() -> bool:
    try:
        from fastapi.routing import iter_route_contexts  # noqa: F401

        return True
    except ImportError:
        return False


@unittest.skipUnless(_has_iter_route_contexts(), "requires fastapi>=0.137.2")
class TestGetFastapiRequestPath(CustomTestCase):
    """`_get_fastapi_request_path` resolves templated route paths for the
    Prometheus HTTP middleware. FastAPI >=0.137 stores included routers as
    `_IncludedRouter` wrappers without `.path`; iter_route_contexts() is
    required to recover the templated label.
    """

    @staticmethod
    def _request(app, path: str, method: str = "GET"):
        scope = {
            "type": "http",
            "method": method,
            "path": path,
            "headers": [],
            "query_string": b"",
            "root_path": "",
        }
        return SimpleNamespace(app=app, scope=scope, url=SimpleNamespace(path=path))

    def test_included_router_without_prefix(self):
        from fastapi import APIRouter, FastAPI

        app = FastAPI()
        router = APIRouter()

        @router.get("/v1/loads")
        def _loads():
            return {}

        app.include_router(router)
        path, handled = _get_fastapi_request_path(self._request(app, "/v1/loads"))
        self.assertEqual(path, "/v1/loads")
        self.assertTrue(handled)

    def test_included_router_with_prefix(self):
        from fastapi import APIRouter, FastAPI

        app = FastAPI()
        router = APIRouter()

        @router.get("/chat/completions")
        def _chat():
            return {}

        app.include_router(router, prefix="/v1")
        path, handled = _get_fastapi_request_path(
            self._request(app, "/v1/chat/completions")
        )
        self.assertEqual(path, "/v1/chat/completions")
        self.assertTrue(handled)

    def test_nested_included_router(self):
        from fastapi import APIRouter, FastAPI

        app = FastAPI()
        sub_router = APIRouter()

        @sub_router.get("/items")
        def _items():
            return {}

        router = APIRouter()
        router.include_router(sub_router, prefix="/sub")
        app.include_router(router, prefix="/v1")

        path, handled = _get_fastapi_request_path(self._request(app, "/v1/sub/items"))
        self.assertEqual(path, "/v1/sub/items")
        self.assertTrue(handled)

    def test_path_params_return_templated_path(self):
        from fastapi import APIRouter, FastAPI

        app = FastAPI()
        router = APIRouter()

        @router.get("/models/{model_id}")
        def _models(model_id: str):
            return {}

        app.include_router(router, prefix="/v1")
        path, handled = _get_fastapi_request_path(self._request(app, "/v1/models/foo"))
        self.assertEqual(path, "/v1/models/{model_id}")
        self.assertTrue(handled)

    def test_builtin_route_is_unchanged(self):
        from fastapi import FastAPI

        app = FastAPI()
        path, handled = _get_fastapi_request_path(self._request(app, "/docs"))
        self.assertEqual(path, "/docs")
        self.assertTrue(handled)

    def test_unmatched_path_falls_back(self):
        from fastapi import FastAPI

        app = FastAPI()
        path, handled = _get_fastapi_request_path(self._request(app, "/nope"))
        self.assertEqual(path, "/nope")
        self.assertFalse(handled)

    @mock.patch("sglang.srt.utils.common.print_warning_once")
    def test_warns_and_falls_back_without_iter_route_contexts(self, mock_warn):
        import sys
        import types

        from fastapi import FastAPI

        fake_routing = types.ModuleType("fastapi.routing")
        app = FastAPI()
        with mock.patch.dict(sys.modules, {"fastapi.routing": fake_routing}):
            path, handled = _get_fastapi_request_path(self._request(app, "/anything"))

        self.assertEqual(path, "/anything")
        self.assertFalse(handled)
        mock_warn.assert_called_once()


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestFlattenArraysToInt64Tensor(CustomTestCase):
    """`flatten_arrays_to_int64_tensor` is invoked by `prepare_for_extend`
    to build the per-batch input_ids tensor (pinned, async H2D) from a
    list of array.array('q') per-req get_fill_ids() slices. Tests the
    full matrix of (device, pin) the production code paths through.
    """

    DEVICES = ("cpu", "cuda")
    PIN_OPTIONS = (False, True)

    def _check(self, parts: list, expected: list[int]) -> None:
        for device in self.DEVICES:
            for pin in self.PIN_OPTIONS:
                with self.subTest(device=device, pin=pin):
                    out = flatten_arrays_to_int64_tensor(parts, device, pin)
                    if device == "cuda":
                        torch.cuda.synchronize()
                    self.assertEqual(out.dtype, torch.int64)
                    self.assertEqual(out.device.type, device)
                    self.assertEqual(out.shape, (len(expected),))
                    self.assertEqual(out.cpu().tolist(), expected)

    def test_single_part(self):
        parts = [array("q", [1, 2, 3, 4, 5])]
        self._check(parts, [1, 2, 3, 4, 5])

    def test_multiple_parts(self):
        parts = [
            array("q", [10, 20, 30]),
            array("q", [100, 200]),
            array("q", [1000]),
        ]
        self._check(parts, [10, 20, 30, 100, 200, 1000])


if __name__ == "__main__":
    unittest.main()
