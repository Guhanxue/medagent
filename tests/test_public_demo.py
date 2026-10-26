"""Real bundled public data through the classifier, without training or API calls."""
import unittest
from unittest.mock import patch
import numpy as np
from medadaptagent.demo.public_example import load_samples
from medadaptagent.demo.workflow import run

class PublicExampleTests(unittest.TestCase):
    def test_bundled_inputs_have_provenance_and_expected_shape(self):
        arrays, metadata = load_samples()
        self.assertEqual(arrays['source_images'].shape,(6,128,128))
        self.assertEqual(arrays['target_images'].shape,(3,128,128))
        self.assertEqual(arrays['target_images'].dtype,np.uint8)
        self.assertEqual(metadata['license'],'CC BY 4.0')
        self.assertEqual(metadata['example_indices'],[0,1,2])
        self.assertEqual(arrays['target_labels'].reshape(-1).tolist(),[1,0,1])

    def test_public_workflow_keeps_reference_labels_out_of_adaptation(self):
        from harness.deploy import deploy_request
        def checked(*args,**kwargs):
            # The scripted request resolves target.npz in this temporary working directory.
            from pathlib import Path
            with np.load(Path(kwargs['cwd'])/'target.npz',allow_pickle=False) as data:
                self.assertEqual(data.files,['images'])
            return deploy_request(*args,**kwargs)
        with patch('harness.deploy.deploy_request',side_effect=checked):
            result=run(dataset='pneumoniamnist',sample=1,shift=0.0)
        self.assertEqual(result['decision'],'keep_source')
        demo=result['demo']
        self.assertEqual(demo['mode'],'public_data_scripted')
        self.assertEqual(demo['dataset_label'],'normal')
        self.assertEqual(demo['source'],demo['target'])
        self.assertTrue(demo['reload_identical'])
        self.assertAlmostEqual(sum(demo['probabilities'].values()),1,places=6)
        self.assertIsNone(demo['prediction']) # classification probabilities, never a fake segmentation

    def test_out_of_range_example_is_rejected(self):
        with self.assertRaises(ValueError):run(dataset='pneumoniamnist',sample=3)
