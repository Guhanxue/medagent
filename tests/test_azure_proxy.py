import os,unittest
from unittest.mock import patch
from medadaptagent.knowledge.llm_client import UnifiedLLMClient

class AzureProxyTests(unittest.TestCase):
    def test_explicit_environment_proxy_reaches_http_client(self):
        env={'AZURE_OPENAI_ENDPOINT':'https://example.org/general','AZURE_OPENAI_API_KEY':'test-only',
             'HTTPS_PROXY':'http://localhost:22222'}
        with patch.dict(os.environ,env,clear=True), patch('httpx.Client') as transport, patch('openai.AzureOpenAI') as api:
            UnifiedLLMClient('azure','gpt-5.2-2025-12-11')
            self.assertEqual(transport.call_args.kwargs['proxy'],env['HTTPS_PROXY'])
            self.assertIs(api.call_args.kwargs['http_client'],transport.return_value)
            self.assertEqual(api.call_args.kwargs['max_retries'],0)
    def test_explicit_proxy_argument_takes_priority(self):
        env={'AZURE_OPENAI_ENDPOINT':'https://example.org/general','AZURE_OPENAI_API_KEY':'test-only',
             'HTTPS_PROXY':'http://localhost:22222'}
        with patch.dict(os.environ,env,clear=True), patch('httpx.Client') as transport, patch('openai.AzureOpenAI'):
            UnifiedLLMClient('azure','gpt-5.2-2025-12-11',proxy='http://localhost:23333')
            self.assertEqual(transport.call_args.kwargs['proxy'],'http://localhost:23333')
if __name__=='__main__':unittest.main()
