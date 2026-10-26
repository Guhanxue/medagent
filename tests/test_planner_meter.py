"""Budget isolation when comparing a local planner with cloud analysis."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from medadaptagent.knowledge.llm_client import UnifiedLLMClient
from scripts.public_repair_support import install_api_meter

class PlannerMeterTests(unittest.TestCase):
    def setUp(self):
        self.original=UnifiedLLMClient.cached_complete
        self.flag=getattr(UnifiedLLMClient,'_repair_metered',None)
        if hasattr(UnifiedLLMClient,'_repair_metered'):del UnifiedLLMClient._repair_metered
        self.addCleanup(self.restore)
    def restore(self):
        UnifiedLLMClient.cached_complete=self.original
        if self.flag is None:
            if hasattr(UnifiedLLMClient,'_repair_metered'):del UnifiedLLMClient._repair_metered
        else:UnifiedLLMClient._repair_metered=self.flag
    def test_loopback_planner_does_not_consume_cloud_budget(self):
        calls=[]
        def complete(client,*args,**kwargs):
            calls.append(client.model)
            return 'local response'
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'budget.json'
            UnifiedLLMClient.cached_complete=complete
            install_api_meter(path,0)
            client=SimpleNamespace(local=True,_endpoint_id='http://127.0.0.1:18434/v1',model='Qwen/Qwen3-VL-4B-Instruct')
            self.assertEqual(UnifiedLLMClient.cached_complete(client,[]),'local response')
            self.assertEqual(calls,[client.model])
            self.assertFalse(path.exists())
    def test_cloud_call_cannot_bypass_exhausted_budget(self):
        calls=[]
        def complete(*args,**kwargs):calls.append(True)
        with tempfile.TemporaryDirectory() as tmp:
            UnifiedLLMClient.cached_complete=complete
            install_api_meter(Path(tmp)/'budget.json',0)
            client=SimpleNamespace(local=False,model='gpt-5.2-2025-12-11')
            with self.assertRaisesRegex(RuntimeError,'budget exhausted'):
                UnifiedLLMClient.cached_complete(client,[{'role':'user','content':'test'}])
            self.assertEqual(calls,[])
    def test_gpt55_cannot_bypass_exhausted_budget(self):
        calls=[]
        with tempfile.TemporaryDirectory() as tmp:
            UnifiedLLMClient.cached_complete=lambda *a,**kw: calls.append(True)
            install_api_meter(Path(tmp)/'budget.json',0)
            client=SimpleNamespace(local=False,model='gpt-5.5')
            with self.assertRaisesRegex(RuntimeError,'budget exhausted'):
                UnifiedLLMClient.cached_complete(client,[{'role':'user','content':'test'}])
            self.assertEqual(calls,[])
    def test_remote_local_backend_is_not_assumed_free(self):
        with tempfile.TemporaryDirectory() as tmp:
            install_api_meter(Path(tmp)/'budget.json',0)
            client=SimpleNamespace(local=True,_endpoint_id='https://example.org/v1',model='Qwen/Qwen3-VL-4B-Instruct')
            with self.assertRaises(ValueError):UnifiedLLMClient.cached_complete(client,[])

if __name__=='__main__':unittest.main()
