"""Failure-path regressions for the bounded public-model experiment runner."""
import tempfile
import unittest
from unittest.mock import Mock, patch
import numpy as np
import torch
from torch import nn
from scripts.run_public_model_repair import require_target_labels
from scripts.public_repair_support import export_and_verify_eager
from medadaptagent.tasks.segmentation_3d import SegmentationTask3D


def preprocess(image, size, device):
    import numpy as np
    import torch
    return torch.from_numpy(np.asarray(image).astype(np.float32))[None, None].to(device)


class RepairFailureTests(unittest.TestCase):
    def test_error_escalation_does_not_request_labels(self):
        action = {'tool':'escalate', 'args':{'report':{'summary':'provider failed'}}}
        policy=require_target_labels(Mock(return_value=action),5)
        self.assertEqual(policy({'k_labels_used':0}),action)

    def test_delivery_and_training_obtain_labels_first(self):
        for name in ['ship','keep_source','finetune_partial','probe_head']:
            action={'tool':name,'args':{}}
            policy=require_target_labels(Mock(return_value=action),5)
            self.assertEqual(policy({'k_labels_used':0}),{'tool':'request_labels','args':{'K':5}})
            self.assertEqual(policy({'k_labels_used':5}),action)

    def test_failed_torchscript_reload_keeps_explicit_eager_verification(self):
        task=Mock()
        task.device='cpu';task._to_tensor=preprocess
        task.timgs=[np.ones((4,4,4),dtype=np.float32)]
        task.tlbls=[np.ones((4,4,4),dtype=np.int64)]
        inference=SegmentationTask3D();inference.set_source_preprocessing('fixture',preprocess,96)
        task._t3d.return_value=inference
        task._center_patch.side_effect=lambda x:x
        task._fg_dice.return_value=1.0
        model=nn.Conv3d(1,2,1).eval()
        with tempfile.TemporaryDirectory() as out, patch('torch.jit.load',side_effect=RuntimeError('fixture reload failure')):
            verification,rows=export_and_verify_eager(task,model,[0],out,{'model_id':'fixture'})
            import json
            from pathlib import Path
            diagnostics=json.loads((Path(out)/'export_diagnostics.json').read_text())
            self.assertEqual(diagnostics['torchscript_status'],'conversion_failed')
            self.assertTrue(verification['verified'])
            self.assertEqual(verification['exact_matching_prediction_masks'],1)
            self.assertEqual(verification['verification_set'],'all 1 hidden-test cases')

if __name__=='__main__':unittest.main()
