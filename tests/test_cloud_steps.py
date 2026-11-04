from backend.natural_workflow import build_natural_workflow
from test_natural import Model,PASS,scores
from test_workflow import initial,SOURCE


def test_cloud_entrypoint_and_subscriber_import():
    from backend.cloud_app import app
    from backend.cloud_worker import run_step
    assert app.title=='LinguaSculpt on Vercel'
    assert run_step is not None


def test_duplicate_queue_delivery_does_not_repeat_model_call(monkeypatch):
    from contextlib import contextmanager
    import backend.cloud_worker as worker
    class Connection:
        def execute(self,*args):return self
        def fetchone(self):return {'id':'abc','version':3,'status':'running'}
    @contextmanager
    def database():yield Connection()
    monkeypatch.setattr(worker,'database',database)
    assert worker.process_step({'id':'abc','version':2})==('abc',3)


def test_durable_node_boundaries_preserve_state():
    fast=Model([SOURCE+' Revised.',PASS]);deep=Model([])
    scorer=scores([.7,.2])
    state=initial()
    for stage in ['prepare','edit','verify','finish']:
        graph=build_natural_workflow(fast,deep,scorer,entry_stage=stage,one_step=True)
        state=graph.invoke(state)
        if stage=='prepare':assert fast.calls==0
        if stage=='edit':assert fast.calls==1 and state['candidate'].endswith('Revised.')
        if stage=='verify':assert fast.calls==2 and state['rounds']==1
    assert state['status']=='completed'
    assert state['best']==SOURCE+' Revised.'
    assert state['source']==SOURCE
