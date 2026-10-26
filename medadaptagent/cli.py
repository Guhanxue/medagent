"""Command line for the recovered budget-aware deployment harness."""
import argparse
import json
import os
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('request', nargs='?', help='Describe the checkpoint, target dataset, task and label budget')
    parser.add_argument('--source-data', help='Source reference NPZ directory (segmentation) or NPZ archive (classification)')
    parser.add_argument('--output', default='outputs/adaptation')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--minutes', type=int, default=60)
    parser.add_argument('--trials', type=int, default=16)
    parser.add_argument('--max-steps', type=int, default=44)
    parser.add_argument('--backend', choices=['local', 'openai', 'azure', 'anthropic'], default=os.environ.get('MEDADAPT_LLM_BACKEND', 'local'))
    parser.add_argument('--model', default=os.environ.get('MEDADAPT_LLM_MODEL'))
    parser.add_argument('--check-llm', action='store_true', help='Send a synthetic text and image connectivity check only')
    args = parser.parse_args(argv)
    if min(args.minutes, args.trials, args.max_steps) <= 0:
        parser.error('Budgets must be positive')
    if not args.check_llm and not args.request:
        parser.error('Provide a deployment request or use --check-llm')
    from medadaptagent.knowledge.llm_client import create_llm_client
    try:
        llm = create_llm_client(args.backend, args.model)
        if args.check_llm:
            import base64
            import io
            from PIL import Image
            buf = io.BytesIO()
            Image.new('RGB', (64,64), (255,0,0)).save(buf, format='PNG')
            response = llm.cached_complete([{'role':'user','content':[
                {'type':'text','text':'What is the dominant color of this image? Reply with one color word.'},
                {'type':'image_url','image_url':{'url':'data:image/png;base64,'+base64.b64encode(buf.getvalue()).decode()}}
            ]}], max_tokens=128)
            if 'red' not in response.choices[0].message.content.lower():
                raise RuntimeError('Vision response did not identify the synthetic red image')
            print('LLM text/image check passed using a synthetic image. No medical data sent.')
            return 0
        from harness.deploy import deploy_request
        result = deploy_request(args.request, llm=llm, source_data=args.source_data,
            output_dir=args.output, device=args.device, compute_minutes=args.minutes,
            n_trials=args.trials, max_steps=args.max_steps)
        dest = Path(args.output)
        dest.mkdir(parents=True, exist_ok=True)
        (dest/'run.json').write_text(json.dumps(result, indent=2, default=str))
        print(json.dumps({'status': result['run_status'], 'decision': result.get('decision'), 'report': str(dest/'run.json')}))
        return 0 if result['run_status']=='completed' else 2
    except Exception as exc:
        # Exception messages from arbitrary model loaders/providers may contain private input.
        print(f'Run failed ({type(exc).__name__}). Check local inputs and endpoint configuration. No successful result was produced.')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
