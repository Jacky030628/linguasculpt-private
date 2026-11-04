import pytest
from backend.workflow import build_workflow
from test_workflow import FakeModel, initial, finish, SOURCE
from backend.tools import analyze_style


@pytest.mark.parametrize('bad',[
    {'action':'finish','draft':SOURCE,'arguments':{'unexpected':1}},
    {'action':'analyze_style','arguments':{'text':SOURCE},'draft':SOURCE},
    {'action':'analyze_style','arguments':{'text':SOURCE,'extra':True}},
])
def test_invalid_action_is_repaired(bad):
    result=build_workflow(FakeModel([bad,finish(),{'passed':True,'issues':[]}])).invoke(initial())
    assert result['passed']
    assert any(e['stage']=='protocol_error' for e in result['events'])
    assert len(result['round_records'])==1


def test_duplicate_call_gets_observation_instead_of_execution():
    action={'action':'analyze_style','arguments':{'text':SOURCE}}
    result=build_workflow(FakeModel([action,action,finish(),{'passed':True,'issues':[]}])).invoke(initial())
    assert result['passed']
    assert len([h for h in result['history'] if 'tool' in h])==1


def test_tool_error_returns_to_agent(monkeypatch):
    def fail(*args,**kwargs):
        raise RuntimeError('private internal details')
    monkeypatch.setattr(analyze_style,'func',fail)
    result=build_workflow(FakeModel([{'action':'analyze_style','arguments':{'text':SOURCE}},finish(),{'passed':True,'issues':[]}])).invoke(initial())
    assert result['passed']
    observation=result['history'][0]['result']
    assert observation['error']=='RuntimeError'
    assert 'private internal details' not in str(observation)
