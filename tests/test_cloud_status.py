from datetime import datetime, timezone, timedelta
from backend.cloud_status import progress, public_status


def task(age=0, status='running', lease=None):
    now=datetime.now(timezone.utc)
    return dict(status=status,updated=now-timedelta(seconds=age),stage='verify',version=2,lease_until=lease)


def test_stalled_tasks_stop_waiting_without_claiming_completion():
    row=task(400)
    assert public_status(row)=='interrupted'
    assert progress(row)['stalled']
    assert progress(row)['stage']=='verify'


def test_active_lease_and_completed_results_are_not_interrupted():
    row=task(400,lease=datetime.now(timezone.utc)+timedelta(seconds=20))
    assert public_status(row)=='running'
    assert progress(row)['executing']
    assert public_status(task(400,status='completed'))=='completed'


def test_queued_tasks_show_running_until_timeout():
    assert public_status(task(10,status='queued'))=='running'
