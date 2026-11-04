"""Save a local, credential-free reproducibility bundle and latest task case."""
import hashlib
import json
import sqlite3
import shutil
from datetime import datetime
from pathlib import Path

root = Path(__file__).resolve().parent
dest = root / 'baselines' / datetime.now().strftime('%Y%m%d-%H%M%S')
dest.mkdir(parents=True)
for folder in ['backend', 'frontend', 'models', 'tests', 'samples']:
    shutil.copytree(root/folder,dest/folder,ignore=shutil.ignore_patterns('__pycache__'))
for name in ['requirements.txt','requirements.lock.txt','.env.example','README.md','start.ps1']:
    shutil.copy2(root/name,dest/name)
for name in ['tasks.sqlite','checkpoints.sqlite']:
    with sqlite3.connect(root/'data'/name) as source, sqlite3.connect(dest/name) as target:
        source.backup(target)
with sqlite3.connect(dest/'tasks.sqlite') as c:
    row = c.execute("SELECT id,payload,status,result,error FROM tasks WHERE status='completed' ORDER BY updated DESC LIMIT 1").fetchone()
if row:
    result = json.loads(row[3])
    case = dict(id=row[0],request=json.loads(row[1]),status=row[2],state=result,error=row[4])
    (dest/'case.json').write_text(json.dumps(case,ensure_ascii=False,indent=2),encoding='utf-8')
    initial=result.get('initial_paper',{}); final=result.get('current_paper',{})
    before={r['feature']:r['contribution'] for r in initial.get('linguistic_contributions',[])}
    changes=[dict(feature=r['feature'],before=before.get(r['feature'],0),after=r['contribution'],delta=r['contribution']-before.get(r['feature'],0)) for r in final.get('linguistic_contributions',[])]
    changes.sort(key=lambda x:abs(x['delta']),reverse=True)
    summary=dict(task_id=row[0],status=row[2],initial_probability=initial.get('ai_class_probability'),final_probability=final.get('ai_class_probability'),rules=result.get('rules'),review=result.get('review'),tool_calls=[h['tool'] for h in result.get('history',[]) if 'tool' in h],linguistic_contribution_changes=changes,
        note='语言学贡献变化仅解释融合 logit 的语言学部分；词和字符贡献也会改变，不能仅凭此归因全部概率变化。')
    (dest/'case_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
manifest={str(p.relative_to(dest)):hashlib.sha256(p.read_bytes()).hexdigest() for p in dest.rglob('*') if p.is_file()}
(dest/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print(dest)
if row:
    print(json.dumps({k:summary[k] for k in ['task_id','initial_probability','final_probability','tool_calls']},ensure_ascii=True))
