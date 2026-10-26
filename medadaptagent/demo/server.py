"""Serve the bundled public-data and synthetic demo on loopback, without external services."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
import json
import threading
from urllib.parse import urlsplit
import webbrowser

RUN_LOCK = threading.Lock()

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def reply(self, status, body, content_type='application/json'):
        if not isinstance(body, bytes):
            body = json.dumps(body).encode()
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'")
        self.end_headers()
        self.wfile.write(body)

    def valid_host(self):
        return self.headers.get('Host') in {f'127.0.0.1:{self.server.server_port}', f'localhost:{self.server.server_port}'}

    def do_GET(self):
        if not self.valid_host():
            return self.reply(403, {'error':'Open this demo through its local address.'})
        if urlsplit(self.path).path != '/':
            return self.reply(404, {'error':'Not found'})
        self.reply(200, files('medadaptagent.demo').joinpath('index.html').read_bytes(), 'text/html; charset=utf-8')

    def do_POST(self):
        origin = self.headers.get('Origin')
        if not self.valid_host() or (origin and origin != 'http://' + self.headers.get('Host','')):
            return self.reply(403, {'error':'Use the demo from its local page.'})
        if self.path != '/api/run':
            return self.reply(404, {'error':'Not found'})
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= 1024 or self.headers.get_content_type() != 'application/json':
                raise ValueError
            params = json.loads(self.rfile.read(length))
            if not isinstance(params, dict) or set(params)-{'seed','shift','dataset','sample'}:
                raise ValueError
            seed, shift = params.get('seed',7), params.get('shift',.25)
            dataset, sample = params.get('dataset','pneumoniamnist'), params.get('sample',0)
            if dataset not in ('pneumoniamnist','synthetic') or type(sample) is not int or not 0 <= sample < 3:
                raise ValueError
            if type(seed) is not int or not 0 <= seed <= 9999 or type(shift) not in (float,int) or not 0 <= shift <= .6:
                raise ValueError
        except (ValueError, TypeError):
            return self.reply(400, {'error':'Choose a valid seed and brightness shift.'})
        if not RUN_LOCK.acquire(blocking=False):
            return self.reply(409, {'error':'A demo is already running. Please wait for it to finish.'})
        try:
            from .workflow import run
            result = run(seed=seed, shift=shift, dataset=dataset, sample=sample)
            # Return only the selected demo payload, never temporary checkpoint paths.
            self.reply(200, {'demo':result['demo'], 'decision':result['decision'], 'steps':result['steps']})
        except Exception as exc:
            print(f'Demo failed: {type(exc).__name__}', flush=True)
            self.reply(500, {'error':'The workflow could not finish. Check the installation and try again; no successful result was produced.'})
        finally:
            RUN_LOCK.release()


def main(argv=None):
    parser = argparse.ArgumentParser(description='Open a friendly, local MedAdaptAgent demo. No API key needed.')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--no-browser', action='store_true')
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error('Port must be between 1 and 65535')
    with ThreadingHTTPServer(('127.0.0.1', args.port), Handler) as server:
        url = f'http://127.0.0.1:{args.port}'
        print(f'MedAdaptAgent demo → {url}\nPublic-data examples · scripted policy · no API calls\nPress Ctrl+C to stop.', flush=True)
        if not args.no_browser:
            webbrowser.open(url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass

if __name__ == '__main__':
    main()
