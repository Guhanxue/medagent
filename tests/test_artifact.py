"""Actual model and fitted preprocessing roundtrips using synthetic images."""
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np
import torch
from harness.delivery import package_artifact
from harness.registry_bridge import apply_registry_tool, CtxCandidate, run_inference, ctx_auroc
from scripts.run_baseline_tta import _to_tensor

class ArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_fitted_preprocessing_survives_roundtrip(self):
        from medadaptagent.models.unet import UNet
        torch.manual_seed(42)
        model=UNet(3,2,base_ch=64).eval()
        image=np.random.default_rng(1).uniform(0,1,(32,32,3)).astype('float32')
        ctx=apply_registry_tool('gamma_correct',model,[image],2,'cpu')
        expected=run_inference(ctx,image,_to_tensor)
        with tempfile.TemporaryDirectory() as dest:
            outcome=SimpleNamespace(delivered=CtxCandidate(ctx,_to_tensor), decision='ship', report={})
            artifact=package_artifact(out=outcome,task=SimpleNamespace(type='segmentation',dims='2D',metric_name='Dice'),
                cfg={'name':'test','arch':'unet','nc':2,'ckpt':'deliberately_missing.pt'},dest_dir=dest,project_root='.')
            spec=importlib.util.spec_from_file_location('reloaded',Path(artifact.dir)/'predict.py')
            module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            restored=module._replay_ctx(module.load_model('cpu'),'cpu')
            np.testing.assert_allclose(restored.preprocess_fn(image),ctx.preprocess_fn(image),atol=0,rtol=0)
            np.testing.assert_array_equal(module.predict([image],device='cpu')[0],expected)
            self.assertEqual(artifact.manifest['tools_applied'],['gamma_correct'])

    def test_classification_roundtrip_and_tensor_convention(self):
        from harness.cls_adapter import build_cls_model, _cls_tensor
        model=build_cls_model('resnet18',2).eval()
        image=np.random.default_rng(2).uniform(0,1,(32,32,3)).astype('float32')
        with torch.no_grad():expected=model(_cls_tensor(image,'cpu')).softmax(1)[0].numpy()
        with tempfile.TemporaryDirectory() as dest:
            artifact=package_artifact(out=SimpleNamespace(delivered=model,decision='keep_source',report={}),
                task=SimpleNamespace(type='classification',dims='2D',metric_name='AUROC'),
                cfg={'name':'test','arch':'resnet18','nc':2,'ckpt':'deliberately_missing.pt'},dest_dir=dest,project_root='.')
            spec=importlib.util.spec_from_file_location('reloaded_cls',Path(artifact.dir)/'predict.py')
            module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            np.testing.assert_allclose(module.predict([image],device='cpu')[0],expected,atol=0,rtol=0)

    def test_undefined_auroc_does_not_become_accuracy(self):
        from unittest.mock import patch
        with patch('harness.registry_bridge._ctx_probs',return_value=np.array([[.2,.8],[.3,.7]])):
            with self.assertRaises(ValueError):ctx_auroc(None,[0,1],[1,1],2,None)
