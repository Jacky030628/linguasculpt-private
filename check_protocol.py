"""Tiny paid R1 protocol smoke check: one tool action followed by one finish."""
import json,os
from pathlib import Path
from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek
from backend.protocol import parse_object
from backend.tools import analyze_style
load_dotenv(Path(__file__).parent/'.env',override=True)
m=ChatDeepSeek(model=os.getenv('DEEPSEEK_MODEL','deepseek-reasoner'),api_key=os.environ['DEEPSEEK_API_KEY'],api_base=os.getenv('DEEPSEEK_BASE_URL','https://api.deepseek.com'),max_tokens=2048,timeout=90,max_retries=0)
source='The study included 48 participants. The findings may improve recall but do not establish causality.'
first=m.invoke([('system','Protocol smoke test. Return only JSON {"action":"analyze_style","arguments":{"text":"..."}} using exactly the supplied text. No explanation.'),('human',source)])
a=parse_object(first.content)
assert a['action']=='analyze_style'
observation=analyze_style.invoke(a['arguments'])
second=m.invoke([('system','Protocol smoke test. After the supplied tool observation return only JSON {"action":"finish","draft":"..."} preserving the supplied source exactly. No explanation.'),('human',json.dumps({'source':source,'observation':observation}))])
b=parse_object(second.content)
assert b['action']=='finish' and b['draft']==source
print(json.dumps({'tool_dispatch':'passed','observation_feedback':'passed','finish':'passed','tokens':sum((v.usage_metadata or {}).get('total_tokens',0) for v in [first,second])}))
