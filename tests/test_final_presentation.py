def test_pending_api_does_not_expose_intermediate_drafts(monkeypatch):
    import backend.main as main
    for status in ['running','failed','interrupted']:
        monkeypatch.setattr(main,'read',lambda task_id:dict(id=task_id,status=status,payload={},error=None,
            result={'source':'source','draft':'intermediate','best':'partial','edits':[{'before':'a','after':'b'}],
                    'events':[],'rounds':1}))
        result=main.task('test')
        assert 'draft' not in result['result']
        assert 'edits' not in result['result']
        assert result['result']['rounds']==1
