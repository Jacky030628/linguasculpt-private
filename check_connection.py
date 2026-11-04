"""One tiny connection probe; no keys or provider response bodies printed."""
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek
from backend.errors import public_error
load_dotenv(Path(__file__).parent / '.env', override=True)
try:
    m=ChatDeepSeek(model=os.getenv('DEEPSEEK_MODEL','deepseek-reasoner'),api_key=os.environ['DEEPSEEK_API_KEY'],api_base=os.getenv('DEEPSEEK_BASE_URL','https://api.deepseek.com'),max_tokens=64,timeout=30,max_retries=0)
    m.invoke([('human','Reply OK.')])
    print('CONNECTION_OK')
except Exception as e:
    print(public_error(e).encode('ascii','backslashreplace').decode())
