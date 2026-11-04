"""Each queue delivery executes one LangGraph node, with durable state afterward."""
import os
import asyncio
from psycopg.types.json import Jsonb
from vercel.queue import subscribe,send
from langchain_deepseek import ChatDeepSeek
from .cloud_store import database
from .natural_workflow import build_natural_workflow
from .paper import score
from .errors import public_error

TOPIC='linguasculpt-steps'


async def enqueue(task_id,version):
    await send(TOPIC,{'id':task_id,'version':version},idempotency_key=f'{task_id}-{version}')


def process_step(payload):
    # Row lock covers one node. Redeliveries never rerun an already committed node.
    with database() as conn:
        row=conn.execute('SELECT * FROM linguasculpt_tasks WHERE id=%s FOR UPDATE',(payload['id'],)).fetchone()
        if not row or row['status'] not in ['running','queued']:return None
        if payload['version']!=row['version']:return (row['id'],row['version'])
        common=dict(api_key=os.getenv('DEEPSEEK_API_KEY') or os.environ['OWNER_API_KEY'],
                    api_base=os.getenv('DEEPSEEK_BASE_URL') or os.getenv('OWNER_BASE_URL','https://api.deepseek.com'),
                    max_retries=0,timeout=120)
        fast=ChatDeepSeek(model=os.getenv('DEEPSEEK_FAST_MODEL','deepseek-chat'),max_tokens=8192,extra_body={'thinking':{'type':'disabled'}},**common)
        deep=ChatDeepSeek(model=os.getenv('DEEPSEEK_MODEL','deepseek-reasoner'),max_tokens=int(os.getenv('DEEPSEEK_MAX_OUTPUT_TOKENS','32768')),**common)
        stage=row['stage']
        graph=build_natural_workflow(fast,deep,score,entry_stage=stage,one_step=True)
        try:
            state=graph.invoke(row['payload'] if stage=='prepare' else row['state'])
        except Exception as exc:
            conn.execute("UPDATE linguasculpt_tasks SET status='failed',error=%s,updated=now() WHERE id=%s",(public_error(exc),row['id']))
            return None
        next_stage={'prepare':'edit','edit':'verify','verify':'edit'}.get(stage,'finish')
        if stage!='finish' and state.get('status')!='running':next_stage='finish'
        status=state.get('status','running') if stage=='finish' else 'running'
        version=row['version']+1
        conn.execute('UPDATE linguasculpt_tasks SET state=%s,status=%s,stage=%s,version=%s,error=NULL,updated=now() WHERE id=%s',
                     (Jsonb(state),status,next_stage,version,row['id']))
        return (row['id'],version) if status=='running' else None


@subscribe(topic=TOPIC,max_concurrency=2,max_attempts=3,retry_after=30)
async def run_step(payload:dict[str,object]):
    next_message=await asyncio.to_thread(process_step,payload)
    if next_message:await enqueue(*next_message)
