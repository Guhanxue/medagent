import unittest
from harness.core import Budget, Tool, SafetyVerdict, build_harness
REPORT = {'trust_level': 'label_verified', 'confidence': 0.8, 'summary': 'Selected from audit evidence'}
class Task:
    type, dims, metric_name = 'segmentation', '3D', 'Dice'
    def __init__(self, safe=True): self.calls=0; self.safe=safe
    def toolbox(self):
        def trial(st):
            self.calls += 1
            return {'candidate': 'adapted', 'metric_estimate': 0.5,
                    'k_labels_used': int(self.calls == 1),
                    'evidence': {'last_method': 'adapted'}}
        return [Tool('trial', 'experiment', trial)]
    def safety_check(self, c, s, e): return SafetyVerdict(self.safe, 'audit veto')
COMMON=[Tool(n,n,lambda st,**kw:{},terminal=True) for n in ('ship','keep_source','escalate')]
class FinalizationTests(unittest.TestCase):
    def run_case(self, safe=True, finalize=True, malformed=False):
        task=Task(safe); views=[]
        def policy(v):
            views.append(v)
            if 'trial' in {x['name'] for x in v['tools']}: return {'tool':'trial','args':{}}
            if malformed:return {'tool':'trial','args':{}}
            return {'tool':'ship','args':{'candidate':'adapted','report':REPORT}}
        out=build_harness(policy,task,COMMON,Budget(k_labels=1),max_steps=10,
                          no_progress_cap=1,finalize_on_no_progress=finalize).run('adapt','source')
        return task,views,out
    def test_existing_candidate_can_be_released_after_search(self):
        task,views,out=self.run_case()
        self.assertEqual(out.decision,'ship');self.assertEqual(out.delivered,'adapted')
        self.assertEqual(task.calls,2)
        self.assertTrue(any(e['kind']=='search_closed' for e in out.trace))
        self.assertEqual({x['name'] for x in views[-1]['tools']},{'ship','keep_source','escalate'})
    def test_gate_still_vetoes_unsafe_candidate(self):
        task,views,out=self.run_case(safe=False)
        self.assertEqual(out.decision,'escalate');self.assertEqual(task.calls,2)
        self.assertEqual(sum(e['kind']=='safety_veto' for e in out.trace),3)
    def test_no_more_experiments_after_search_closes(self):
        task,views,out=self.run_case(malformed=True)
        self.assertEqual(out.decision,'escalate');self.assertEqual(task.calls,2)
        self.assertEqual(sum(e['kind']=='invalid_action' for e in out.trace),3)
    def test_legacy_default_remains_a_stop(self):
        task,views,out=self.run_case(finalize=False)
        self.assertEqual(out.decision,'escalate');self.assertEqual(task.calls,2)
        self.assertFalse(any(e['kind']=='search_closed' for e in out.trace))
if __name__=='__main__':unittest.main()
