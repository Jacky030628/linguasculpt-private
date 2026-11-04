"""Bounded waiting and diagnostic metadata for durable cloud tasks."""
from datetime import datetime, timezone

STALE_SECONDS = 360
ACTIVE = ('queued', 'running')


def progress(row, now=None):
    now = now or datetime.now(timezone.utc)
    updated = row['updated']
    idle = max(0, int((now - updated).total_seconds()))
    lease = row.get('lease_until')
    stale = row['status'] in ACTIVE and idle >= STALE_SECONDS and (not lease or lease <= now)
    labels = {'prepare': '准备与语言特征分析', 'edit': '文本改写',
              'verify': '规则与语义检查', 'finish': '整理最终结果'}
    return {'stage': row['stage'], 'label': labels.get(row['stage'], row['stage']),
            'version': row['version'], 'updated_at': updated.isoformat(),
            'idle_seconds': idle, 'stalled': stale,
            'executing': bool(lease and lease > now)}


def public_status(row):
    return 'interrupted' if progress(row)['stalled'] else ('running' if row['status'] == 'queued' else row['status'])
