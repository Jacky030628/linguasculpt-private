import pytest
from langchain_core.messages import AIMessage
from backend.protocol import parse_object
from backend.workflow import build_workflow
from test_workflow import FakeModel,initial,finish,SOURCE


def test_wrapped_json_is_accepted():
    assert parse_object('Here is the action:\n```json\n{"action":"finish","draft":"a {braced} string"}\n```')['draft']=='a {braced} string'


@pytest.mark.parametrize('text',['','[]','{"a":1} {"a":2}','incomplete text'])
def test_ambiguous_or_empty_output_rejected(text):
    with pytest.raises(ValueError): parse_object(text)


def test_repair_invalid_text_argument_without_reediting():
    editor=FakeModel([{'action':'analyze_style','arguments':{'text':{'text':SOURCE}}},finish(),{'passed':True,'issues':[]}])
    repair=FakeModel([{'action':'analyze_style','arguments':{'text':SOURCE}}])
    r=build_workflow(editor,repair_model=repair).invoke(initial())
    assert r['passed'] and r['steps']==2
    assert r['tokens']==80
    assert r['history'][0]['arguments']['text']==SOURCE


def test_unrecoverable_repair_stops_within_protocol_budget():
    editor=FakeModel([{'bad':'output'}]*2)
    repair=FakeModel([{'unrecoverable':True}]*2)
    r=build_workflow(editor,repair_model=repair).invoke(initial())
    assert r['stop_reason']=='protocol_errors'
    assert r['tokens']==80


def test_empty_final_answer_skips_repair():
    class Empty:
        def invoke(self,messages): return AIMessage(content='')
    r=build_workflow(Empty(),repair_model=FakeModel([])).invoke(initial())
    assert r['stop_reason']=='protocol_errors'


def test_reasoning_exhaustion_stops_after_one_call():
    class Truncated:
        calls=0
        def invoke(self,messages):
            self.calls+=1
            return AIMessage(content='',response_metadata={'finish_reason':'length'})
    model=Truncated()
    result=build_workflow(model,repair_model=FakeModel([])).invoke(initial())
    assert result['stop_reason']=='output_limit'
    assert model.calls==1
