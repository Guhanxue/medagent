"""Attributed public X-ray inputs through the real classification harness.

This is an integration walkthrough with an untrained model and scripted policy,
not a medical evaluation. Target labels are shown only as dataset metadata.
"""
import base64
import hashlib
import importlib.util
from importlib.resources import files
import io
import json
from pathlib import Path
import tempfile
import time
import numpy as np
import torch


def load_samples():
    resources = files('medadaptagent.demo').joinpath('samples')
    metadata = json.loads(resources.joinpath('provenance.json').read_text())
    raw = resources.joinpath('pneumoniamnist_examples.npz').read_bytes()
    if hashlib.sha256(raw).hexdigest() != metadata['subset_sha256']:
        raise ValueError('Public sample checksum mismatch; restore the bundled dataset files')
    with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    return arrays, metadata


def run_public(seed=7, shift=.25, sample=0):
    from harness.cls_adapter import build_cls_model, _cls_tensor
    from harness.deploy import deploy_request
    started = time.perf_counter()
    arrays, metadata = load_samples()
    if type(sample) is not int or not 0 <= sample < len(arrays['target_images']):
        raise ValueError('Choose one of the bundled public examples')
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    originals = arrays['target_images'].astype('float32') / 255
    shifted = np.clip(originals * (1-shift) + shift, 0, 1)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        np.savez(root/'source.npz', images=arrays['source_images'].astype('float32')/255)
        # Never pass held-out dataset labels into this zero-label walkthrough.
        np.savez(root/'target.npz', images=shifted)
        model = build_cls_model('resnet18', 2).eval()
        checkpoint = root/'resnet18.pt'
        torch.save({'model_state_dict':model.state_dict(), 'arch':'resnet18'}, checkpoint)
        class ScriptedPolicy:
            def __init__(self):
                self.steps = iter([
                    {'tool':'load_and_deploy','args':{}},
                    {'tool':'plan','args':{'steps':['bn_reset','keep_source'], 'rationale':'Public-data software walkthrough'}},
                    {'tool':'bn_reset','args':{}},
                    {'tool':'keep_source','args':{'report':{'trust_level':'label_free_estimate','confidence':0.0,
                        'summary':'Scripted walkthrough with an untrained classifier; retain source'}}},
                ])
            def chat(self,prompt,**kwargs):
                if prompt.startswith('Extract the deployment request'):
                    return json.dumps({'model_ref':str(checkpoint),'data_path':str(root/'target.npz'),
                        'task_question':'chest X-ray binary classification','k_labels':0,'objective':'never_worse'})
                return json.dumps(next(self.steps))
        result = deploy_request('Public-data classification software walkthrough',llm=ScriptedPolicy(),
            cwd=str(root),device='cpu',source_data=str(root/'source.npz'),output_dir=str(root/'output'),
            compute_minutes=10,n_trials=3,max_steps=8)
        if result['run_status'] != 'completed' or result['decision'] != 'keep_source':
            raise RuntimeError('Public-data walkthrough did not complete the expected workflow')
        if any(t.get('kind')=='tool_error' for t in result['trace']) or not any(
            t.get('kind')=='tool' and t.get('tool')=='bn_reset' for t in result['trace']):
            raise RuntimeError('The adaptation tool did not complete successfully')
        spec = importlib.util.spec_from_file_location('public_predict',Path(result['artifact']['directory'])/'predict.py')
        predictor = importlib.util.module_from_spec(spec);spec.loader.exec_module(predictor)
        actual = predictor.predict([shifted[sample]],device='cpu')[0]
        with torch.no_grad():
            expected = model(_cls_tensor(shifted[sample],'cpu')).softmax(1)[0].numpy()
        np.testing.assert_array_equal(actual,expected)
        from PIL import Image
        def png(array):
            buf=io.BytesIO()
            Image.fromarray((np.clip(array,0,1)*255).astype('uint8')).resize((256,256),Image.Resampling.NEAREST).save(buf,format='PNG')
            return 'data:image/png;base64,'+base64.b64encode(buf.getvalue()).decode()
        label = metadata['labels'][str(int(arrays['target_labels'][sample].item()))]
        result['demo'] = {'mode':'public_data_scripted','dataset':metadata['dataset'],'seed':seed,'sample':sample,
            'sample_index':metadata['example_indices'][sample],'split':'test','dataset_label':label,
            'source_indices':metadata['source_indices'],'source_split':'train',
            'shift':shift,'shift_kind':'simulated brightness; not a cross-hospital split',
            'model':'untrained ResNet18','seconds':round(time.perf_counter()-started,2),'reload_identical':True,
            'source':png(originals[sample]),'target':png(shifted[sample]),'prediction':None,
            'probabilities':{metadata['labels'][str(i)]:float(p) for i,p in enumerate(actual)},
            'record':metadata['record'],'license':metadata['license'],'license_url':metadata['license_url'],
            'events':[{'kind':t.get('kind'),'tool':t.get('tool')} for t in result['trace']]}
        return result
