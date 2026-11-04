import os
import json
import uuid
from pathlib import Path
from fastapi import FastAPI,HTTPException
from fastapi.responses import FileResponse,JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel,Field
from psycopg.types.json import Jsonb
from .cloud_store import database
from .cloud_worker import enqueue
from .change_records import change_records

ROOT=Path(__file__).resolve().parents[1]
app=FastAPI(title='LinguaSculpt on Vercel')


class EditRequest(BaseModel):
    source:str=Field(min_length=20,max_length=30000)
    style:str=Field(default='Strict Academic',max_length=200)
    depth:int=Field(default=2,ge=1,le=3)
    terms:list[str]=Field(default_factory=list,max_length=100)


def read(task_id):
    with database() as conn:
        row=conn.execute('SELECT * FROM linguasculpt_tasks WHERE id=%s',(task_id,)).fetchone()
    if not row:raise HTTPException(404,'Task not found')
    return row


@app.get('/api/health')
def health():
    return dict(status='ok',hosted_llm_ready=bool((os.getenv('DEEPSEEK_API_KEY') or os.getenv('OWNER_API_KEY')) and os.getenv('DATABASE_URL')),
                default_model=os.getenv('DEEPSEEK_FAST_MODEL','deepseek-chat'))


@app.post('/api/tasks',status_code=202)
async def create(req:EditRequest):
    if not (os.getenv('DEEPSEEK_API_KEY') or os.getenv('OWNER_API_KEY')) or not os.getenv('DATABASE_URL'):
        raise HTTPException(503,'服务尚未配置完成')
    task_id=uuid.uuid4().hex
    with database() as conn:
        conn.execute('SELECT pg_advisory_xact_lock(8159301)')
        if conn.execute("SELECT count(*) AS n FROM linguasculpt_tasks WHERE status IN ('queued','running')").fetchone()['n']>=2:
            raise HTTPException(429,'已有两个任务执行中，请稍后再试。')
        conn.execute("INSERT INTO linguasculpt_tasks(id,payload,status) VALUES (%s,%s,'queued')",(task_id,Jsonb(req.model_dump())))
    try:await enqueue(task_id,0)
    except Exception:
        with database() as conn:conn.execute("UPDATE linguasculpt_tasks SET status='failed',error='队列提交失败，请恢复任务' WHERE id=%s",(task_id,))
    return {'id':task_id}


@app.get('/api/tasks/{task_id}')
def task(task_id:str):
    row=read(task_id);state=row['state']
    if row['status'] in ['completed','needs_review']:
        state['edits']=change_records(state['source'],state.get('best') or state['draft'])
    else:state={k:state[k] for k in ['events','rounds','steps','tokens','usage_log'] if k in state}
    return dict(id=task_id,status='running' if row['status']=='queued' else row['status'],result=state,error=row['error'])


@app.post('/api/tasks/{task_id}/resume')
async def resume(task_id:str):
    with database() as conn:
        row=conn.execute('SELECT * FROM linguasculpt_tasks WHERE id=%s FOR UPDATE',(task_id,)).fetchone()
        if not row or row['status']!='failed':raise HTTPException(409,'只有失败任务可以恢复')
        version=row['version']+1
        conn.execute("UPDATE linguasculpt_tasks SET status='queued',version=%s,error=NULL WHERE id=%s",(version,task_id))
    try:await enqueue(task_id,version)
    except Exception:
        with database() as conn:conn.execute("UPDATE linguasculpt_tasks SET status='failed',error='队列提交失败，请重试' WHERE id=%s",(task_id,))
    return {'id':task_id}


@app.get('/api/tasks/{task_id}/export')
def export(task_id:str):
    row=read(task_id)
    return JSONResponse(dict(id=task_id,status=row['status'],payload=row['payload'],result=row['state'],error=row['error']),headers={'Content-Disposition':f'attachment; filename="linguasculpt-{uuid.UUID(task_id).hex}.json"'})


@app.get('/api/samples')
def samples():return json.loads((ROOT/'samples'/'examples.json').read_text(encoding='utf-8'))


@app.get('/')
def home():return FileResponse(ROOT/'frontend'/'index.html')


app.mount('/static',StaticFiles(directory=ROOT/'frontend'),name='static')
