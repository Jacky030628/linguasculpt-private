import json
from langchain_core.messages import AIMessage
from backend.natural_workflow import build_natural_workflow
from test_workflow import SOURCE,initial


class Model:
    model_name='test'
    def __init__(self,replies): self.replies=iter(replies);self.calls=0
    def invoke(self,messages):
        self.calls+=1
        reply=next(self.replies)
        return AIMessage(content=json.dumps(reply) if isinstance(reply,dict) else reply,
                         usage_metadata={'input_tokens':10,'output_tokens':10,'total_tokens':20})


PASS={'passed':True,'issues':[]}


def scores(values):
    it=iter(values)
    return lambda text:dict(available=True,ai_class_probability=next(it),locked_threshold=.34)


def test_fast_first_pass_early_stop():
    fast=Model([SOURCE+' Revised.',PASS]);deep=Model([])
    r=build_natural_workflow(fast,deep,scores([.7,.2])).invoke(initial())
    assert r['rounds']==1 and r['status']=='completed' and deep.calls==0


def test_above_threshold_is_deliverable_after_three_revisions():
    fast=Model([SOURCE+' A.',PASS,SOURCE+' B.',PASS,SOURCE+' C.',PASS]);deep=Model([])
    r=build_natural_workflow(fast,deep,scores([.7,.9,.8,.85])).invoke(initial())
    assert r['rounds']==3 and r['status']=='completed' and deep.calls==0
    assert r['current_paper']['ai_class_probability']==.8
    assert r['draft'].endswith(' B.')


def test_difficult_case_escalates_for_semantic_repair():
    fast=Model([SOURCE+' Changed.',{'passed':False,'issues':['Meaning changed']}])
    deep=Model([SOURCE+' Repaired.',PASS])
    r=build_natural_workflow(fast,deep,scores([.7,.1,.2])).invoke(initial())
    assert r['rounds']==2 and r['status']=='completed'
    assert [x['route'] for x in r['usage_log']]==['fast','fast','deep','deep']


def test_deep_mode_still_starts_fast():
    fast=Model([SOURCE+' First.',PASS,PASS]);deep=Model([SOURCE+' Better.'])
    args=initial()|{'depth':3}
    r=build_natural_workflow(fast,deep,scores([.7,.6,.2])).invoke(args)
    assert r['rounds']==2 and deep.calls==1
    assert r['usage_log'][0]['route']=='fast'
