"""Offline software checks. No paper experiments, credentials or clinical data."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def main():
    checks = [[sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-v']]
    checks += [[sys.executable, str(ROOT/'harness'/name)] for name in
        ['test_core.py','test_llm_policy.py','test_delivery.py','test_spine.py','test_step0.py',
         'test_ship_policy.py','test_analysis_C.py','test_lessons.py','test_tool_advisor.py']]
    checks += [[sys.executable, str(ROOT/'examples/smoke_workflow.py')]]
    failed=[]
    for cmd in checks:
        result=subprocess.run(cmd,cwd=ROOT)
        if result.returncode:failed.append(cmd[-1])
    print(f'Offline check groups: {len(checks)-len(failed)}/{len(checks)} passed; failures={failed}')
    return bool(failed)

if __name__=='__main__':raise SystemExit(main())
