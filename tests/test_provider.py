"""Provider contract and credential isolation, using an in-process HTTP transport."""
import json
import os
import unittest
from unittest.mock import patch
import httpx
from medadaptagent.knowledge.llm_client import create_llm_client, _endpoint
from harness.intake import parse_request
from harness.llm_policy import make_llm_policy

class ProviderTests(unittest.TestCase):
    def test_loopback_ignores_cloud_credentials_and_preserves_images(self):
        with patch.dict(os.environ, {'AZURE_OPENAI_API_KEY':'synthetic-azure', 'OPENAI_API_KEY':'synthetic-openai'}, clear=True):
            client = create_llm_client()
        captured = []
        def handle(request):
            captured.append(request)
            return httpx.Response(200, json={'id':'test','object':'chat.completion','created':0,'model':client.model,
                'choices':[{'index':0,'finish_reason':'stop','message':{'role':'assistant','content':'red'}}]})
        client._client._client.close()
        client._client._client = httpx.Client(transport=httpx.MockTransport(handle))
        messages=[{'role':'user','content':[{'type':'text','text':'color?'},
            {'type':'image_url','image_url':{'url':'data:image/png;base64,synthetic'}}]}]
        self.assertEqual(client.cached_complete(messages).choices[0].message.content,'red')
        self.assertEqual(json.loads(captured[0].content)['messages'], messages)
        self.assertIn('max_tokens', json.loads(captured[0].content))
        self.assertNotIn('synthetic-azure', str(captured[0].headers))
        self.assertNotIn('synthetic-openai', str(captured[0].headers))
        client._client.close()

    def test_endpoint_rejects_credential_urls_and_remote_plain_http(self):
        for url in ['https://user:password@example.org/v1','https://example.org/v1?key=x','http://example.org/v1']:
            with self.assertRaises(ValueError):_endpoint(url)

    def test_provider_error_does_not_echo_response_or_key(self):
        with patch.dict(os.environ, {}, clear=True): client=create_llm_client()
        client._client._client.close()
        client._client._client = httpx.Client(transport=httpx.MockTransport(lambda req:httpx.Response(401,json={'error':'synthetic-sensitive-server-text'})))
        with self.assertRaises(RuntimeError) as err:client.chat('hello')
        self.assertNotIn('synthetic-sensitive',str(err.exception))
        client._client.close()

    def test_invalid_intake_cannot_fall_back_to_regex(self):
        class Bad:
            def chat(self,*a,**kw):return 'invalid response'
        with self.assertRaises(ValueError):parse_request('model.pt data/target',llm=Bad())

    def test_invalid_action_does_not_execute_a_substitute(self):
        result=make_llm_policy(lambda _: 'invalid')({'tools':[], 'evidence':{}, 'goal':'test',
            'task':{'type':'segmentation','dims':'2D','metric':'Dice'},'step':0,'deployed':False,'budget_remaining':{},'candidates_tried':[], 'k_labels_used':0, 'best_metric_estimate':None, 'guardrail_feedback':None})
        self.assertEqual(result['tool'],'escalate')
        self.assertIn('no fallback',result['args']['report']['summary'])
