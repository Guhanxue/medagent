"""HTTP contract and credential-boundary tests, not live model evaluations."""
import contextlib
import io
import json
import os
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

import llm_client


class ClientTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                cls.requests.append((self.path, dict(self.headers), json.loads(self.rfile.read(int(self.headers['Content-Length'])))))
                self.send_response(cls.status)
                if cls.status == 302:
                    self.send_header('Location', cls.url + '/redirected')
                self.end_headers()
                self.wfile.write(cls.body)
        cls.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        cls.url = f'http://127.0.0.1:{cls.server.server_port}/v1'
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def setUp(self):
        self.env = patch.dict(os.environ, {'MEDADAPT_LLM_BASE_URL': self.url, 'MEDADAPT_LLM_MODEL': 'test-model', 'MEDADAPT_LLM_API_KEY': ''})
        self.env.start()
        self.addCleanup(self.env.stop)
        type(self).requests = []
        type(self).status = 200
        self.respond('{"status":"ok"}')

    def respond(self, content, finish='stop'):
        type(self).body = json.dumps({'choices': [{'message': {'content': content}, 'finish_reason': finish}]}).encode()

    def test_local_json_request_has_no_key_by_default(self):
        self.assertEqual(json.loads(llm_client.call_llm('Synthetic prompt')), {'status': 'ok'})
        path, headers, payload = self.requests[0]
        self.assertEqual(path, '/v1/chat/completions')
        self.assertNotIn('Authorization', headers)
        self.assertEqual(payload['model'], 'test-model')
        self.assertEqual(payload['response_format'], {'type': 'json_object'})
        self.assertFalse(payload['stream'])

    def test_environment_key_sent_only_in_header(self):
        with patch.dict(os.environ, {'MEDADAPT_LLM_API_KEY': 'unit-test-placeholder'}):
            llm_client.call_llm('Synthetic prompt')
        path, headers, payload = self.requests[0]
        self.assertEqual(headers['Authorization'], 'Bearer unit-test-placeholder')
        self.assertNotIn('unit-test-placeholder', path + json.dumps(payload))

    def test_error_body_and_key_not_logged(self):
        type(self).status = 401
        type(self).body = b'unit-test-placeholder private-server-detail'
        with patch.dict(os.environ, {'MEDADAPT_LLM_API_KEY': 'unit-test-placeholder'}):
            with self.assertRaises(llm_client.LLMError) as caught:
                llm_client.call_llm('Synthetic prompt')
        message = str(caught.exception)
        self.assertIn('401', message)
        self.assertNotIn('unit-test-placeholder', message)
        self.assertNotIn('private-server-detail', message)
        self.assertEqual(len(self.requests), 1)

    def test_redirects_not_followed(self):
        type(self).status = 302
        with self.assertRaises(llm_client.LLMError):
            llm_client.call_llm('Synthetic prompt')
        self.assertEqual(len(self.requests), 1)

    def test_remote_http_rejected_before_network(self):
        with patch.dict(os.environ, {'MEDADAPT_LLM_BASE_URL': 'http://example.com/v1'}):
            with self.assertRaisesRegex(llm_client.LLMError, 'HTTPS'):
                llm_client.call_llm('Synthetic prompt')
        self.assertEqual(self.requests, [])

    def test_credentials_in_url_rejected_without_echo(self):
        for url in ('https://user:unit-test-placeholder@example.com/v1', 'https://example.com/v1?key=unit-test-placeholder', 'https://example.com/v1#unit-test-placeholder'):
            with self.subTest(url_type=url.split(':')[0]):
                with patch.dict(os.environ, {'MEDADAPT_LLM_BASE_URL': url}):
                    with self.assertRaises(llm_client.LLMError) as caught:
                        llm_client.call_llm('Synthetic prompt')
                self.assertNotIn('unit-test-placeholder', str(caught.exception))
        self.assertEqual(self.requests, [])

    def test_empty_model_has_no_fallback(self):
        with patch.dict(os.environ, {'MEDADAPT_LLM_MODEL': ''}):
            with self.assertRaises(llm_client.LLMError):
                llm_client.call_llm('Synthetic prompt')
        self.assertEqual(self.requests, [])

    def test_invalid_response_rejected(self):
        for content in ('not JSON', '[]', '{"unfinished":'):
            self.respond(content)
            with self.subTest(content=content), self.assertRaises(llm_client.LLMError):
                llm_client.call_llm('Synthetic prompt')

    def test_truncated_response_rejected_even_when_json_parses(self):
        self.respond('{"status":"ok"}', finish='length')
        with self.assertRaisesRegex(llm_client.LLMError, 'finish'):
            llm_client.call_llm('Synthetic prompt')

    def test_invalid_key_header_rejected_before_network(self):
        for key in ('unit-test\nplaceholder', 'unit-test-\u2603'):
            with patch.dict(os.environ, {'MEDADAPT_LLM_API_KEY': key}):
                with self.assertRaises(llm_client.LLMError) as caught:
                    llm_client.call_llm('Synthetic prompt')
                self.assertNotIn(key, str(caught.exception))
        self.assertEqual(self.requests, [])

    def test_malformed_choice_rejected(self):
        type(self).body = b'{"choices":[null]}'
        with self.assertRaises(llm_client.LLMError):
            llm_client.call_llm('Synthetic prompt')

    def test_missing_choices_rejected(self):
        type(self).body = b'{}'
        with self.assertRaises(llm_client.LLMError):
            llm_client.call_llm('Synthetic prompt')

    def test_cli_success_uses_only_synthetic_prompt(self):
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            self.assertEqual(llm_client.main(), 0)
        self.assertIn('connectivity check passed', stream.getvalue())
        self.assertEqual(self.requests[0][2]['messages'][1]['content'], 'Return exactly this JSON object: {"status":"ok"}')

    def test_cli_rejects_wrong_model_answer_without_printing_it(self):
        self.respond('{"status":"private-server-detail"}')
        stream = io.StringIO()
        with contextlib.redirect_stderr(stream):
            self.assertEqual(llm_client.main(), 1)
        self.assertNotIn('private-server-detail', stream.getvalue())


if __name__ == '__main__':
    unittest.main()
