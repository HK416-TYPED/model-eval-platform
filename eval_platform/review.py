"""Measured execution summaries and explicitly human-authored case reviews."""
from collections import Counter
from statistics import median
import math

VERDICTS = {'unreviewed': '待评审', 'pass': '符合预期', 'mixed': '部分符合', 'fail': '不符合预期'}
TASK_LABELS = {'edit_dual': '双图参考', 'edit_single': '单图编辑', 't2i': '文生图'}


def summarize(jobs, reviews=()):
    jobs = list(jobs)
    groups = []
    for model, task in sorted({(j['model_id'], j['task']) for j in jobs}):
        selected = [j for j in jobs if (j['model_id'], j['task']) == (model, task)]
        counts = Counter(j['state'] for j in selected)
        def measured(key):
            values = [(j.get('result') or {}).get(key) for j in selected if j['state'] == 'succeeded']
            return [x for x in values if isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) and x >= 0]
        times, memory = measured('generation_seconds'), measured('peak_allocated_bytes')
        finished = counts['succeeded'] + counts['failed']
        groups.append({'model_id': model, 'task': task, 'planned': len(selected),
                       'succeeded': counts['succeeded'], 'failed': counts['failed'],
                       'pending': counts['queued'] + counts['running'], 'cancelled': counts['cancelled'],
                       'success_rate': counts['succeeded'] / finished if finished else None,
                       'median_seconds': median(times) if times else None, 'timing_samples': len(times),
                       'peak_gib': max(memory) / 2**30 if memory else None, 'memory_samples': len(memory)})
    reviewed = [r for r in reviews if r['verdict'] != 'unreviewed']
    return {'groups': groups, 'reviewed_cases': len(reviewed),
            'review_verdicts': dict(Counter(r['verdict'] for r in reviewed)),
            'failures': [{'job_id': j['id'], 'model_id': j['model_id'], 'case_id': j['case_id'],
                          'seed': j['seed'], 'error': j.get('error') or '未记录原因'}
                         for j in jobs if j['state'] == 'failed'],
            'scope': '成功率 = 已生成 / (已生成 + 失败)，不包含待运行、运行中和取消项。耗时与显存仅统计有记录的成功输出。生成成功不代表图像质量合格。'}
