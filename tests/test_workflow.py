import json
import sqlite3
import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from backend.workflow import build_workflow
from backend.tools import check_constraints


SOURCE = 'The study included 48 people and may improve recall.'


class FakeModel:
    def __init__(self, replies):
        self.replies = iter(replies)

    def invoke(self, messages):
        try:
            reply = next(self.replies)
        except StopIteration:
            raise RuntimeError('Simulated provider outage') from None
        return AIMessage(content=json.dumps(reply), usage_metadata={'input_tokens': 10, 'output_tokens': 10, 'total_tokens': 20})


def initial():
    return {'source': SOURCE, 'style': 'Academic', 'depth': 2, 'terms': []}


def finish(text=SOURCE):
    return {'action': 'finish', 'draft': text}


def test_actual_tool_dispatch_and_checkpoint(tmp_path):
    model = FakeModel([{'action': 'analyze_style', 'arguments': {'text': SOURCE}}, finish(), {'passed': True, 'issues': []}])
    config = {'configurable': {'thread_id': 'test'}}
    with SqliteSaver.from_conn_string(str(tmp_path / 'checkpoint.sqlite')) as saver:
        graph = build_workflow(model, saver)
        result = graph.invoke(initial(), config)
        assert result['passed'] and result['status'] == 'completed'
        assert result['history'][0]['tool'] == 'analyze_style'
        assert graph.get_state(config).values['best'] == SOURCE


def test_rule_failure_cannot_be_overridden_by_reviewer():
    bad = SOURCE.replace('48', '49')
    model = FakeModel([finish(bad), {'passed': True, 'issues': []}] * 3)
    result = build_workflow(model).invoke(initial(), {'recursion_limit': 60})
    assert result['status'] == 'needs_review'
    assert not result['passed'] and not result['best']
    assert result['rounds'] == 3


def test_semantic_failure_gets_repaired():
    model = FakeModel([finish(SOURCE.replace('may', 'will')), {'passed': False, 'issues': ['may changed to will']}, finish(), {'passed': True, 'issues': []}])
    result = build_workflow(model).invoke(initial())
    assert result['passed'] and result['rounds'] == 2


def test_unknown_tool_is_bounded():
    model = FakeModel([{'action': 'execute_shell', 'arguments': {}}] * 2)
    result = build_workflow(model, max_steps=2).invoke(initial())
    assert result['status'] == 'needs_review'
    assert result['steps'] == 2


def test_number_count_and_terms():
    result = check_constraints.invoke({'source': 'ART found 5 cases.', 'draft': 'Art found 5 cases and 5 more.', 'spans': ['ART', '5']})
    assert not result['passed']


def test_resume_after_provider_failure(tmp_path):
    config = {'configurable': {'thread_id': 'resume'}}
    with SqliteSaver.from_conn_string(str(tmp_path / 'resume.sqlite')) as saver:
        graph = build_workflow(FakeModel([{'action': 'analyze_style', 'arguments': {'text': SOURCE}}]), saver)
        with pytest.raises(RuntimeError, match='Simulated provider outage'):
            graph.invoke(initial(), config)
        assert graph.get_state(config).next == ('editor',)
        restored = build_workflow(FakeModel([finish(), {'passed': True, 'issues': []}]), saver)
        result = restored.invoke(None, config)
        assert result['passed']
        assert result['history'][0]['tool'] == 'analyze_style'


def test_api_without_credentials(monkeypatch):
    monkeypatch.delenv('DEEPSEEK_API_KEY', raising=False)
    from fastapi.testclient import TestClient
    from backend.main import app
    # Import loads .env; override afterward so tests never submit real LLM jobs.
    monkeypatch.setenv('DEEPSEEK_API_KEY', '')
    with TestClient(app) as client:
        assert client.get('/').status_code == 200
        assert client.get('/static/app.js').status_code == 200
        assert len(client.get('/api/samples').json()) == 3
        assert client.post('/api/tasks', json={'source': SOURCE}).status_code == 503
        assert client.post('/api/tasks', json={'source': 'short'}).status_code == 422
