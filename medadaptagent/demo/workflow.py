"""CPU integration check with synthetic images and a scripted policy, not an experiment.

Runs the real checkpoint loader, intake, segmentation adapter, BN recalibration,
harness and artifact writer. The policy retains the source model. No LLM, real
medical data or published metrics are used. A live model is checked separately.
"""
import base64
import io
import time
import importlib.util
import json
from pathlib import Path
import tempfile
import numpy as np
import torch
from harness.deploy import deploy_request
from medadaptagent.models.unet import UNet


def run(seed=7, shift=0.25, dataset="synthetic", sample=0):
    if type(seed) is not int or not 0 <= seed <= 9999:
        raise ValueError("Seed must be an integer from 0 to 9999")
    if not isinstance(shift, (int, float)) or not np.isfinite(shift) or not 0 <= shift <= 0.6:
        raise ValueError("Shift must be between 0 and 0.6")
    if dataset == "pneumoniamnist":
        from .public_example import run_public
        return run_public(seed=seed, shift=shift, sample=sample)
    if dataset != "synthetic":
        raise ValueError("Unknown demo dataset")
    started = time.perf_counter()
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        yy, xx = np.mgrid[-1:1:32j, -1:1:32j]
        shape = ((xx / .62) ** 2 + (yy / .78) ** 2 < 1).astype('float32')
        source = np.clip(.12 + .5 * shape + rng.normal(0,.04,(32,32)), 0, 1)
        target = np.clip(source * (1-shift) + shift, 0, 1)
        for name in ('source', 'target'):
            (root/name).mkdir()
            for i in range(2):
                base = source if name == 'source' else target
                img = np.repeat(np.roll(base,i,axis=0)[...,None],3,axis=2).astype('float32')
                np.savez(root/name/f'{i}.npz', image=img)
        checkpoint = root/'unet.pt'
        model = UNet(in_channels=3, num_classes=2, base_ch=64).eval()
        torch.save({'model_state_dict':model.state_dict(), 'arch':'unet'}, checkpoint)
        class ScriptedPolicy:
            def __init__(self):
                self.steps = iter([
                    {'tool':'load_and_deploy','args':{}},
                    {'tool':'plan','args':{'steps':['bn_reset','keep_source'], 'rationale':'Synthetic software check'}},
                    {'tool':'bn_reset','args':{}},
                    {'tool':'keep_source','args':{'report':{'trust_level':'label_free_estimate','confidence':0.0,'summary':'Synthetic test; retain source'}}},
                ])
            def chat(self, prompt, **kwargs):
                if prompt.startswith('Extract the deployment request'):
                    return json.dumps({'model_ref':str(checkpoint),'data_path':str(root/'target'),
                        'task_question':'segmentation','k_labels':0,'objective':'never_worse'})
                return json.dumps(next(self.steps))
        result = deploy_request('Synthetic segmentation check', llm=ScriptedPolicy(),
            cwd=str(root), device='cpu', source_data=str(root/'source'),
            output_dir=str(root/'output'), compute_minutes=10, n_trials=3, max_steps=8)
        assert result['run_status']=='completed', result
        assert result['decision']=='keep_source', result
        assert any(t.get('kind') == 'tool' and t.get('tool') == 'bn_reset' for t in result['trace']), result
        assert not any(t.get('kind') == 'tool_error' for t in result['trace']), result
        artifact = Path(result['artifact']['directory'])
        assert (artifact/'model.pt').exists()
        spec = importlib.util.spec_from_file_location('delivered_predict', artifact/'predict.py')
        mod = importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
        sample = np.repeat(target[...,None],3,axis=2).astype('float32')
        actual = mod.predict([sample], device='cpu')[0]
        from scripts.run_baseline_tta import _to_tensor
        with torch.no_grad():expected = model(_to_tensor(sample,'cpu')).argmax(1)[0].numpy()
        np.testing.assert_array_equal(actual, expected)
        print('PASS: checkpoint intake, real BN tool, source retention, artifact save/reload, identical predictions (32x32).')
        from PIL import Image
        def png(array):
            buf = io.BytesIO()
            Image.fromarray(np.asarray(array, dtype='uint8')).resize((256,256), Image.Resampling.NEAREST).save(buf, format='PNG')
            return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()
        palette = np.array([[27,47,64],[159,227,207]], dtype='uint8')
        result['demo'] = {
            'mode':'synthetic_scripted', 'seed':seed, 'shift':shift,
            'seconds':round(time.perf_counter()-started,2), 'reload_identical':True,
            'source':png(source*255), 'target':png(target*255), 'prediction':png(palette[actual]),
            'events':[{'kind':t.get('kind'), 'tool':t.get('tool')} for t in result['trace']],
        }
        return result


if __name__=='__main__':
    run()
