import json
from backend.natural_workflow import extract_revision,build_natural_workflow
from test_natural import Model,PASS,scores
from test_workflow import initial,SOURCE


def test_plan_separated_from_final_text():
    prose,plan=extract_revision('[[EDIT_PLAN]]\n建议重构句式。\n[[REVISED_TEXT]]\n'+SOURCE+' Revised.')
    assert prose==SOURCE+' Revised.'
    assert plan=='建议重构句式。'
    assert extract_revision('[[EDIT_PLAN]] incomplete')==('','')


def test_top_ten_weights_given_to_model_and_plan_saved():
    class Capture(Model):
        def invoke(self,messages):
            if self.calls==0:self.context=json.loads(messages[1][1])
            return super().invoke(messages)
    fast=Capture(['[[EDIT_PLAN]]\n调整从句。\n[[REVISED_TEXT]]\n'+SOURCE+' Revised.',PASS])
    def score(text):
        return dict(available=True,ai_class_probability=.2 if 'Revised' in text else .7,locked_threshold=.34,
                    linguistic_contributions=[{'feature':str(i),'coefficient':i,'contribution':.1,'raw_value':1} for i in range(15)])
    result=build_natural_workflow(fast,Model([]),score).invoke(initial())
    assert [v['coefficient'] for v in fast.context['top_weighted_linguistic_features']]==list(range(14,4,-1))
    assert result['best']==SOURCE+' Revised.'
    assert result['round_records'][0]['editor_suggestions']=='调整从句。'


def test_unchanged_text_not_successful_edit():
    fast=Model([SOURCE,PASS])
    r=build_natural_workflow(fast,Model([]),scores([.7,.2]),token_budget=40).invoke(initial())
    assert r['status']=='needs_review' and not r['selected']
