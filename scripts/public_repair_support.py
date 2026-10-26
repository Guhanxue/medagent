"""Bounded public-model repair runs: API accounting and explicit export comparison."""
from pathlib import Path
import json, os, time, hashlib, fcntl


def install_api_meter(path, cap_usd):
    """Enforce a shared conservative spend estimate across all Azure LLM clients.

    Reserve a request upper estimate before network use. Account all reported
    tokens at $10/M input and $80/M output, above published GPT-5.2 and GPT-5.5 list rates.
    Cached tokens receive no discount. Failed requests retain their reservation.
    """
    from medadaptagent.knowledge.llm_client import UnifiedLLMClient
    if getattr(UnifiedLLMClient, '_repair_metered', False): return
    original=UnifiedLLMClient.cached_complete
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    def request(self,messages,max_tokens=2000,temperature=0,seed=42,system=None):
        # Local loopback inference has no provider charge; cloud calls remain metered.
        from urllib.parse import urlsplit
        if getattr(self, 'local', False) and urlsplit(self._endpoint_id).hostname in ('localhost', '127.0.0.1', '::1'):
            return original(self, messages, max_tokens=max_tokens, temperature=temperature, seed=seed, system=system)
        if not (self.model.startswith('gpt-5.2') or self.model in ('gpt-5.5', 'gpt-5.5-2026-04-23')):
            raise ValueError('This budget supports GPT-5.2 and GPT-5.5 cloud models')
        # UTF-8 bytes upper-bound text tokens; reserve 8,192 tokens per image.
        text_bytes=len((system or '').encode()); images=0
        for msg in messages:
            content=msg.get('content','')
            if isinstance(content,str):text_bytes+=len(content.encode())
            else:
                for item in content:
                    if item.get('type')=='text':text_bytes+=len(item.get('text','').encode())
                    elif item.get('type')=='image_url':images+=1
                    else:raise ValueError('Unmetered message content type')
        reserve=(text_bytes+8192*images+512*len(messages))*10/1e6+max_tokens*80/1e6
        with path.open('a+') as f:
            fcntl.flock(f,fcntl.LOCK_EX);f.seek(0);raw=f.read()
            state=json.loads(raw) if raw else {'cap_usd':cap_usd,'charged_upper_usd':0,'requests':[]}
            if state['charged_upper_usd']+reserve>cap_usd:
                raise RuntimeError('Shared API budget exhausted before request')
            idx=len(state['requests']);state['charged_upper_usd']+=reserve
            state['requests'].append({'request':idx,'pid':os.getpid(),'status':'reserved','reserved_upper_usd':reserve,'model':self.model})
            f.seek(0);f.truncate();json.dump(state,f,indent=2);f.flush();os.fsync(f.fileno())
        try:
            response=original(self,messages,max_tokens=max_tokens,temperature=temperature,seed=seed,system=system)
        except Exception as exc:
            errors=[];cause=exc
            while cause is not None:
                errors.append({'type':type(cause).__name__,'status':getattr(cause,'status_code',None),
                               'code':getattr(cause,'code',None)})
                cause=cause.__context__
            with path.open('r+') as f:
                fcntl.flock(f,fcntl.LOCK_EX);state=json.load(f)
                state['requests'][idx].update(status='failed',errors=errors)
                f.seek(0);f.truncate();json.dump(state,f,indent=2);f.flush();os.fsync(f.fileno())
            raise
        if response.usage is None:raise RuntimeError('Provider omitted usage; reserved amount retained')
        usage=response.usage.model_dump();charged=usage['prompt_tokens']*10/1e6+usage['completion_tokens']*80/1e6
        with path.open('r+') as f:
            fcntl.flock(f,fcntl.LOCK_EX);state=json.load(f)
            state['charged_upper_usd']+=charged-reserve
            state['requests'][idx].update(status='completed',usage=usage,charged_upper_usd=charged)
            f.seek(0);f.truncate();json.dump(state,f,indent=2);f.flush();os.fsync(f.fileno())
        if charged>reserve:raise RuntimeError('API token usage exceeded reservation; stop and inspect accounting')
        return response
    UnifiedLLMClient.cached_complete=request;UnifiedLLMClient._repair_metered=True


def export_and_verify_eager(task, model, indices, out_dir, metadata):
    import copy
    import numpy as np
    import torch
    from harness.seg3d_delivery import save_eager_package, load_eager_package
    from medadaptagent.tasks.segmentation_3d import _sliding_window_predict
    out_dir=Path(out_dir);diagnostics={'artifact_format':'python_eager_cloudpickle','cases':[]}
    package=out_dir/'eager_package'
    save_eager_package(model, task._to_tensor, package, metadata)
    fresh,fresh_task=load_eager_package(package, task.device, trusted=True)
    t3d=task._t3d(); model.eval()
    # TorchScript is a separately assessed conversion, never a silent fallback.
    example=task._center_patch(t3d.to_tensor(task.timgs[indices[0]],task.device))
    traced=None
    try:
        with torch.no_grad():traced=torch.jit.trace(copy.deepcopy(model).eval(),example,strict=False,check_trace=False)
        torch.jit.save(traced,str(out_dir/'diagnostic_model.ts'))
        del traced
        traced=torch.jit.load(str(out_dir/'diagnostic_model.ts'),map_location=task.device).eval()
        diagnostics['torchscript_status']='loaded'
    except Exception as exc:
        traced=None
        diagnostics['torchscript_status']='conversion_failed'
        diagnostics['torchscript_error_type']=type(exc).__name__
    rows=[];matched=0
    for i in indices:
        x=t3d.to_tensor(task.timgs[i],task.device)
        x_reload=fresh_task.to_tensor(task.timgs[i],task.device)
        if not torch.equal(x,x_reload):raise RuntimeError('Reloaded preprocessing differs')
        with torch.no_grad():
            logits=_sliding_window_predict(model,x)
            again=_sliding_window_predict(model,x)
            loaded=_sliding_window_predict(fresh,x_reload)
        mask=logits.argmax(1)[0].cpu().numpy();repeat=again.argmax(1)[0].cpu().numpy();reload_mask=loaded.argmax(1)[0].cpu().numpy()
        record={'index':i,'voxels':int(mask.size),'eager_repeat_changed_voxels':int(np.count_nonzero(mask!=repeat)),
                'eager_reload_changed_voxels':int(np.count_nonzero(mask!=reload_mask)),
                'eager_reload_max_logit_difference':float((logits-loaded).abs().max().item())}
        if traced is not None:
            with torch.no_grad():ts_logits=_sliding_window_predict(traced,x)
            ts_mask=ts_logits.argmax(1)[0].cpu().numpy();changed=mask!=ts_mask
            record['torchscript_changed_voxels']=int(np.count_nonzero(changed))
            record['torchscript_max_logit_difference']=float((logits-ts_logits).abs().max().item())
            if changed.any():
                top=torch.topk(logits,2,dim=1).values; margins=(top[:,0]-top[:,1])[0].cpu().numpy()
                record['changed_voxel_max_eager_margin']=float(margins[changed].max())
            del ts_logits
        diagnostics['cases'].append(record)
        (out_dir/'export_diagnostics.json').write_text(json.dumps(diagnostics,indent=2))
        if record['eager_repeat_changed_voxels'] or record['eager_reload_changed_voxels']:
            raise RuntimeError(f'Eager inference/reload mask mismatch on case {i}; inspect export_diagnostics.json')
        matched+=1
        label=np.asarray(task.tlbls[i]);pred=mask
        if pred.shape!=label.shape:pred=task._resize_pred_to(pred,label.shape)
        dice=task._fg_dice(pred,label)
        if dice is None:raise ValueError('Evaluation case lacks target foreground')
        rows.append({'index':i,'dice':round(float(dice),4)})
        print(json.dumps({'export_case':i,**record}),flush=True)
        del x,x_reload,logits,again,loaded
    verification={'artifact':'eager_package','artifact_format':'python_eager_cloudpickle','verified':True,
                  'verification_set':f'all {len(indices)} hidden-test cases','exact_matching_prediction_masks':matched,
                  'mean_hidden_test_dice':round(float(np.mean([r['dice'] for r in rows])),4),
                  'max_absolute_case_dice_difference':0.0}
    (out_dir/'reload_verification.json').write_text(json.dumps(verification,indent=2))
    return verification,rows
