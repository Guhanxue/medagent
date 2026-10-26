"""Eager package trust/hash checks and optional CPU graph roundtrip."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from harness.seg3d_delivery import (
    ARTIFACT_FORMAT, ARTIFACT_NAME, INFERENCE_SETTINGS,
    load_eager_package, save_eager_package, _safe_preprocessing,
)


class EagerBoundaryTests(unittest.TestCase):
    def test_trust_is_required_before_file_access(self):
        with self.assertRaises(PermissionError):
            load_eager_package("does-not-exist")

    def test_bound_task_preprocessing_is_rejected(self):
        class Task:
            def to_tensor(self, image, size, device):
                return image
        with self.assertRaises(TypeError):
            _safe_preprocessing(Task().to_tensor)

    def test_data_bearing_closure_is_rejected(self):
        task = object()
        def preprocess(image, size, device):
            return task
        with self.assertRaises(ValueError):
            _safe_preprocessing(preprocess)

    def test_keyword_defaults_and_function_attributes_cannot_capture_data(self):
        def keyword(image, size, device, *, capture=b"private fixture"):
            return image
        with self.assertRaises(ValueError):
            _safe_preprocessing(keyword)
        def attribute(image, size, device):
            return image
        attribute.private_data = b"private fixture"
        with self.assertRaises(ValueError):
            _safe_preprocessing(attribute)

    def test_referenced_global_data_is_rejected(self):
        namespace = {"PRIVATE_FIXTURE": b"private fixture"}
        exec("def preprocess(image, size, device): return PRIVATE_FIXTURE", namespace)
        with self.assertRaises(ValueError):
            _safe_preprocessing(namespace["preprocess"])
        exec("def nested(image, size, device): return (lambda: PRIVATE_FIXTURE)()", namespace)
        with self.assertRaises(ValueError):
            _safe_preprocessing(namespace["nested"])

    def test_known_public_model_preprocessing_is_accepted(self):
        from harness.seg3d_adapter import _make_window_to_tensor
        from scripts.run_public_model_repair import wholebody_to_tensor
        _safe_preprocessing(_make_window_to_tensor(-57, 164, [1.5, 1.5, 2.0]))
        _safe_preprocessing(wholebody_to_tensor([1.5, 1.5, 1.5]))

    def test_unknown_metadata_is_rejected_before_serialization(self):
        def preprocess(image, size, device):
            return image
        with self.assertRaises(ValueError):
            save_eager_package(None, preprocess, "unused", {"credentials": "dummy"})

    def test_tampering_rejected_before_unpickling(self):
        fake_pickle = Mock()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            manifest = {
                "schema_version": 1, "artifact_format": ARTIFACT_FORMAT,
                "artifact": ARTIFACT_NAME, "inference": INFERENCE_SETTINGS,
                "preprocessing_size": 96,
                "artifact_sha256": hashlib.sha256(b"original").hexdigest(),
            }
            (path / "manifest.json").write_text(json.dumps(manifest))
            (path / ARTIFACT_NAME).write_bytes(b"changed")
            with patch.dict("sys.modules", {"cloudpickle": fake_pickle}):
                with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                    load_eager_package(path, trusted=True)
            fake_pickle.loads.assert_not_called()

    def test_manifest_cannot_redirect_artifact_path(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "manifest.json").write_text(json.dumps({
                "schema_version": 1, "artifact_format": ARTIFACT_FORMAT,
                "artifact": "../external.pkl",
            }))
            with self.assertRaisesRegex(ValueError, "filename"):
                load_eager_package(path, trusted=True)


HAS_RUNTIME = all(importlib.util.find_spec(name) is not None
                  for name in ("cloudpickle", "torch", "monai"))


@unittest.skipUnless(HAS_RUNTIME, "requires cloudpickle, torch, and MONAI")
class EagerRoundtripTests(unittest.TestCase):
    def test_local_model_class_and_preprocessing_roundtrip_on_cpu(self):
        import numpy as np
        import torch
        from torch import nn

        class LocalAdapter(nn.Module):
            def __init__(self):
                super().__init__()
                self.head = nn.Conv3d(1, 2, 1)
            def forward(self, x):
                return self.head(x)

        class LocalExtractor(nn.Module):
            def __init__(self):
                super().__init__()
                self.base = LocalAdapter()
            def forward(self, x):
                return self.base(x)[:, [0, 1]]

        scale = 2.0
        def preprocess(image, size, device):
            import numpy as np
            import torch
            return torch.from_numpy(np.asarray(image).astype(np.float32) * scale)[None, None].to(device)

        model = LocalExtractor().train()
        before = {key: value.clone() for key, value in model.state_dict().items()}
        with tempfile.TemporaryDirectory() as directory:
            manifest = save_eager_package(model, preprocess, directory, {"model_id": "test"})
            loaded, task = load_eager_package(directory, "cpu", trusted=True)
            self.assertEqual(manifest["artifact_format"], ARTIFACT_FORMAT)
            self.assertTrue(model.training)
            self.assertFalse(loaded.training)
            self.assertIsNot(loaded, model)
            image = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
            x = preprocess(image, 96, "cpu")
            self.assertTrue(torch.equal(x, task.to_tensor(image, "cpu")))
            self.assertTrue(torch.equal(model(x), loaded(x)))
            for key, value in model.state_dict().items():
                self.assertTrue(torch.equal(value, before[key]))
            second, _ = load_eager_package(directory, "cpu", trusted=True)
            self.assertIsNot(second, loaded)
            with self.assertRaises(FileExistsError):
                save_eager_package(model, preprocess, directory, {})


if __name__ == "__main__":
    unittest.main()
