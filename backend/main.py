import asyncio
import json
import os
import sqlite3
import time
import uuid
import secrets
from contextlib import asynccontextmanager
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from langchain_deepseek import ChatDeepSeek
from langgraph.checkpoint.sqlite import SqliteSaver
from .workflow import build_workflow
from .paper import score as paper_score
from .errors import public_error
from .natural_workflow import build_natural_workflow

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / '.env', override=True)
DATA = Path(os.getenv('LINGUASCULPT_DATA_DIR',str(ROOT / 'data')))
DATA.mkdir(exist_ok=True)
MODEL = os.getenv('DEEPSEEK_MODEL', 'deepseek-reasoner')
FAST_MODEL = os.getenv('DEEPSEEK_FAST_MODEL','deepseek-chat')
POOL = ThreadPoolExecutor(max_workers=2)


def db():
    conn = sqlite3.connect(DATA / 'tasks.sqlite')
    conn.execute('CREATE TABLE IF NOT EXISTS tasks (id TEXT PRIMARY KEY, payload TEXT NOT NULL, status TEXT NOT NULL, result TEXT, error TEXT, updated REAL)')
    return conn


def update(task_id, **fields):
    fields['updated'] = time.time()
    with db() as c:
        c.execute('UPDATE tasks SET ' + ','.join(f'{k}=?' for k in fields) + ' WHERE id=?', [*fields.values(), task_id])


def read(task_id):
    with db() as c:
        row = c.execute('SELECT id,payload,status,result,error FROM tasks WHERE id=?', (task_id,)).fetchone()
    if not row:
        raise HTTPException(404, 'Task not found')
    return {'id': row[0], 'payload': json.loads(row[1]), 'status': row[2], 'result': json.loads(row[3]) if row[3] else None, 'error': row[4]}


def run(task_id, payload, resume=False):
    try:
        model = ChatDeepSeek(model=MODEL, api_key=os.environ['DEEPSEEK_API_KEY'],
            api_base=os.getenv('DEEPSEEK_BASE_URL', 'https://api.deepseek.com'),
            # R1 max_tokens covers reasoning AND the final answer.
            max_tokens=int(os.getenv('DEEPSEEK_MAX_OUTPUT_TOKENS','32768')), timeout=180, max_retries=1)
        with SqliteSaver.from_conn_string(str(DATA / 'checkpoints.sqlite')) as saver:
            if payload.get('workflow_version')=='v8':
                fast=ChatDeepSeek(model=FAST_MODEL,api_key=os.environ['DEEPSEEK_API_KEY'],
                    api_base=os.getenv('DEEPSEEK_BASE_URL','https://api.deepseek.com'),max_tokens=8192,
                    extra_body={'thinking':{'type':'disabled'}},timeout=90,max_retries=1)
                graph=build_natural_workflow(fast,model,paper_score,saver)
            else:
                graph = build_workflow(model, saver, paper_scorer=paper_score,
                                       repair_model=model.bind(max_tokens=4096), eager_candidates=True)
            config = {'configurable': {'thread_id': task_id}, 'recursion_limit': 60}
            state = graph.get_state(config) if resume else None
            if state and not state.next:
                final = dict(state.values)
            else:
                final = {}
                for snapshot in graph.stream(None if resume else payload, config, stream_mode='values'):
                    final = dict(snapshot)
                    update(task_id, result=json.dumps(final, ensure_ascii=False))
            update(task_id, status=final.get('status', 'needs_review'), result=json.dumps(final, ensure_ascii=False), error=None)
    except Exception as exc:
        # Never log credentials, API response bodies, or private drafts as errors.
        update(task_id, status='failed', error=public_error(exc))


@asynccontextmanager
async def lifespan(app):
    with db() as c:
        c.execute("UPDATE tasks SET status='interrupted' WHERE status='running'")
    yield


app = FastAPI(title='LinguaSculpt Agent', lifespan=lifespan)


@app.middleware('http')
async def gateway_access(request,call_next):
    secret=os.getenv('BACKEND_PROXY_SECRET','')
    if secret and request.url.path.startswith('/api/') and request.url.path!='/api/health':
        supplied=request.headers.get('x-linguasculpt-secret','')
        if not secrets.compare_digest(supplied,secret):
            return JSONResponse({'detail':'Unauthorized'},status_code=401)
    return await call_next(request)


class Request(BaseModel):
    source: str = Field(min_length=20, max_length=30000)
    style: str = Field(default='Strict Academic', max_length=200)
    depth: int = Field(default=2, ge=1, le=3)
    terms: list[str] = Field(default_factory=list, max_length=100)


@app.get('/api/health')
def health():
    return {'status': 'ok', 'hosted_llm_ready': bool(os.getenv('DEEPSEEK_API_KEY')), 'default_model': FAST_MODEL,
            'deep_model':MODEL,'architecture': 'LangGraph v8: fast editor + conditional R1 escalation'}


@app.post('/api/tasks', status_code=202)
def create(req: Request):
    if not os.getenv('DEEPSEEK_API_KEY'):
        raise HTTPException(503, '请先在项目 .env 中填写 DEEPSEEK_API_KEY。')
    task_id = uuid.uuid4().hex
    payload=req.model_dump() | {'workflow_version':'v8'}
    with db() as c:
        if c.execute("SELECT count(*) FROM tasks WHERE status='running'").fetchone()[0] >= 2:
            raise HTTPException(429, '已有两个任务执行中，请稍后再试。')
        c.execute('INSERT INTO tasks VALUES (?,?,?,?,?,?)', (task_id, json.dumps(payload), 'running', None, None, time.time()))
    POOL.submit(run, task_id, payload)
    return {'id': task_id}


@app.get('/api/tasks/{task_id}')
def task(task_id: str):
    result = read(task_id)
    # Return operational state, never private model reasoning or internal prompts.
    state = result['result'] or {}
    if result['status'] not in ('completed','needs_review'):
        result['result']={k:state[k] for k in ['events','rounds','steps','tokens','usage_log'] if k in state}
        result.pop('payload')
        return result
    # Rebuild old saved tasks as well; never reuse incomplete historical snippets.
    if state.get('source') and (state.get('best') or state.get('draft')):
        from .change_records import change_records
        state['edits']=change_records(state['source'],state.get('best') or state['draft'])
        selected=state.get('selected_round')
        for record in state.get('round_records',[]):
            record['edits']=change_records(state['source'],record['draft'])
            if record.get('passed'):
                if record['round']==selected:
                    state['issues']=record['rules']['issues']+record['review']['issues']
    result['result'] = {k: state[k] for k in ['source', 'draft', 'best', 'events', 'issues', 'rules', 'review', 'edits', 'rounds', 'steps', 'tokens', 'passed', 'initial_paper', 'current_paper', 'stop_reason', 'score_feedback', 'round_records', 'usage_log', 'selected_round'] if k in state}
    result.pop('payload')
    return result


@app.post('/api/tasks/{task_id}/resume')
def resume(task_id: str):
    if not os.getenv('DEEPSEEK_API_KEY'):
        raise HTTPException(503, '请先配置密钥。')
    with db() as c:
        if c.execute("SELECT count(*) FROM tasks WHERE status='running'").fetchone()[0] >= 2:
            raise HTTPException(429, '请等待当前任务完成。')
        row = c.execute("SELECT payload FROM tasks WHERE id=? AND status IN ('failed','interrupted')", (task_id,)).fetchone()
        if not row:
            raise HTTPException(409, '只有失败或中断的任务可以恢复。')
        c.execute("UPDATE tasks SET status='running',error=NULL WHERE id=?", (task_id,))
    POOL.submit(run, task_id, json.loads(row[0]), True)
    return {'id': task_id}


@app.get('/api/tasks/{task_id}/export')
def export_task(task_id: str):
    record = read(task_id)
    state = record.get('result') or {}
    record['result'] = {k:v for k,v in state.items() if k != 'history'}
    record['tool_calls'] = [h for h in state.get('history',[]) if 'tool' in h]
    record['metadata'] = {'configured_deep_model':MODEL,'configured_fast_model':FAST_MODEL,
                          'protocol':record['payload'].get('workflow_version','legacy R1 JSON actions'),
                          'note':'Configured model ID is not proof of provider model identity. No private reasoning exported.'}
    return JSONResponse(record,headers={'Content-Disposition':f'attachment; filename="linguasculpt-{record["id"]}.json"'})


@app.get('/api/samples')
def samples():
    return json.loads((ROOT / 'samples' / 'examples.json').read_text(encoding='utf-8'))


class AnalyzeRequest(BaseModel):
    text: str = Field(min_length=1, max_length=30000)


@app.post('/api/analyze')
def analyze(req: AnalyzeRequest):
    return paper_score(req.text)


@app.get('/')
def index():
    return FileResponse(ROOT / 'frontend' / 'index.html')


app.mount('/static', StaticFiles(directory=ROOT / 'frontend'), name='static')
