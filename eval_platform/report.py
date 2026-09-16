import html
import json
import shutil
import zipfile
from pathlib import Path
from PIL import Image, ImageDraw, ImageOps
from .core import atomic_bytes, inside, read_json, write_json


def build_report(store,run_id,boards=False):
    run=store.run(run_id);spec=run['spec'];suite=read_json(Path(spec['suite_dir'])/'suite.lock.json')
    run_dir=store.root/'runs'/run_id;out=run_dir/'report';out.mkdir(parents=True,exist_ok=True)
    assets=out/'assets';assets.mkdir(exist_ok=True)
    def copy_asset(source):
        p=Path(source);target=assets/(p.parent.name+'-'+p.name)
        from .core import file_sha
        if not target.exists() or file_sha(target)!=file_sha(p):shutil.copyfile(p,target)
        return 'assets/'+target.name
    jobs=store.jobs(run_id);summary=store.summary(run_id);index={(j['model_id'],j['case_id'],j['seed']):j for j in jobs}
    for j in jobs:
        if j['state']=='succeeded' and j['result']:
            try:
                output=inside(run_dir,j['result']['image'])
                j['report_image']=copy_asset(output)
                previews=out/'previews';previews.mkdir(exist_ok=True)
                preview=previews/(j['id']+'-'+j['result']['sha256'][:12]+'.webp')
                if not preview.exists():
                    with Image.open(output) as original:
                        thumb=ImageOps.exif_transpose(original).convert('RGB')
                        thumb.thumbnail((480,480),Image.Resampling.LANCZOS)
                        import io
                        buffer=io.BytesIO();thumb.save(buffer,format='WEBP',quality=84)
                        atomic_bytes(preview,buffer.getvalue())
            except OSError:j['report_error']='Output file unavailable'
            j['prepared_report_images']=[{'id':a['id'],'path':copy_asset(inside(run_dir,a['path']))} for a in j['result'].get('preprocessed_inputs',[])]
    with store.connect() as db:
        scores=[{'job_id':r['job_id'],'scorer':r['scorer'],'result':json.loads(r['result'])} for r in db.execute('SELECT * FROM scores WHERE run_id=?',(run_id,))]
    report={'schema':'eval-report/v1','summary':summary,'suite':suite,'experiment':spec,'jobs':jobs,'scores':scores,
            'evaluation_scope':'training_set_replay','scoring_status':'configured' if scores else 'not_configured'}
    write_json(out/'report.json',report)
    write_json(out/'suite.lock.json',suite)
    for case in suite['cases']:
        for asset in case['inputs']+([case['dataset_target']] if 'dataset_target' in case else []):
            p=inside(out,asset['path']);p.parent.mkdir(parents=True,exist_ok=True)
            if not p.exists():shutil.copyfile(inside(spec['suite_dir'],asset['path']),p)
    for name in ['models.lock.json','environment.json','experiment.lock.json']:
        p=run_dir/name
        if p.exists():shutil.copyfile(p,out/name)
    atomic_bytes(out/'results.jsonl',''.join(json.dumps(j,ensure_ascii=False)+'\n' for j in jobs).encode())
    esc=lambda x:html.escape(str(x),quote=True)
    state_label={'completed':'已完成','queued':'待运行','running':'运行中','failed':'失败','cancelled':'已取消','completed_with_errors':'部分失败','unavailable':'待接入'}.get(summary['state'],summary['state'])
    parts=['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
           '<title>'+esc(run_id)+' · FIELDNOTE 图像报告</title><link rel="stylesheet" href="theme.css"><body><main class="report-shell">',
           '<div class="topline"><span class="eyebrow">FIELDNOTE / IMAGE REPORT</span><span class="mono">TRAINING-SET REPLAY</span></div>',
           '<header class="hero report-hero"><div><div class="eyebrow">CHECKPOINT OBSERVATORY</div><h1>每一个 epoch，<br>都值得被看见。</h1><p class="run-name">'+esc(run_id)+'</p></div><div class="hero-index" aria-hidden="true">'+str(suite.get('count_per_task',''))+'×'+str(len(suite['seeds']))+'</div></header>',
           '<p class="status">'+esc(state_label)+' · 成功 '+str(summary['counts']['succeeded'])+' / '+str(summary['planned'])+'</p>',
           '<p class="muted">同一行固定案例和 seed，列为不同 checkpoint。目标图像素不进入推理；Edit 可使用其宽高比确定采样尺寸。评分：'+('已配置' if scores else '未配置')+'</p>',
           '<nav class="report-nav"><label for="task" style="margin:0">任务</label><select id="task"><option value="">全部</option><option value="edit_dual">双图参考</option><option value="edit_single">单图参考</option><option value="t2i">文生图</option></select><label for="seed" style="margin:0">Seed</label><select id="seed"><option value="">全部</option>'+''.join('<option>'+str(s)+'</option>' for s in suite['seeds'])+'</select><a href="report.json">JSON</a> · <a href="results.jsonl">逐项记录</a>'+(' · <a class="button primary" href="report.pdf">下载 PDF ↓</a>' if (out/'report.pdf').is_file() else '')+'</nav>']
    for unavailable in spec.get('unavailable',[]):
        reason=str(unavailable.get('reason','等待资源')).split(': ',1)[-1]
        parts.append('<p class="muted">待接入：'+esc(reason)+'</p>')
    parts.append('<details><summary>抽样范围与数据版本</summary><pre>'+esc(json.dumps(suite['sources'],ensure_ascii=False,indent=2))+'</pre></details>')
    case_number=0
    for family in sorted({m['family'] for m in spec['models']}):
        models=[m for m in spec['models'] if m['family']==family]
        for case in sorted(suite['cases'],key=lambda c:({'edit_dual':0,'edit_single':1,'t2i':2}[c['task']],c['case_id'])):
            columns=[m for m in models if any(j['model_id']==m['id'] and j['case_id']==case['case_id'] for j in jobs)]
            if not columns:continue
            task=case['task'];case_number+=1
            task_label={'edit_dual':'双图参考','edit_single':'单图参考','t2i':'文生图'}[task]
            parts.append('<section class="panel" data-task="'+esc(task)+'"><div class="case-head"><span class="case-number">'+str(case_number).zfill(2)+'</span><div><h2>'+esc(family.upper()+' / '+task_label)+'</h2><small class="mono">CASE / '+esc(case['provenance']['row_id'])+'</small></div></div><div class="eyebrow muted">ORIGINAL PROMPT / INSTRUCTION</div><pre class="case-prompt">'+esc(case['prompt'])+'</pre>')
            parts.append('<p class="muted">语义：'+esc(case['semantic_mode'])+' · 来源：'+esc(case['provenance']['source_id'])+'</p><div class="refs">')
            for label,asset in [(f'输入 {i+1}',a) for i,a in enumerate(case['inputs'])]+([('数据集目标图（仅对照）',case['dataset_target'])] if 'dataset_target' in case else []):
                src=copy_asset(inside(spec['suite_dir'],asset['path']))
                parts.append('<figure><a href="'+src+'"><img loading="lazy" src="'+src+'"></a><figcaption>'+esc(label)+'</figcaption></figure>')
            parts.append('</div><div class="table-scroll"><table class="compare-table"><thead><tr><th style="width:105px">seed</th>'+''.join('<th>'+esc(m['id'])+'<small>'+esc(m['identity']['assets']['checkpoint']['sha256'][:12])+'</small></th>' for m in columns)+'</tr></thead><tbody>')
            for seed in suite['seeds']:
                parts.append('<tr data-seed="'+str(seed)+'"><th>'+str(seed)+'</th>')
                for m in columns:
                    j=index[(m['id'],case['case_id'],seed)];parts.append('<td>')
                    if j.get('report_image'):
                        parts.append('<a href="'+j['report_image']+'"><img loading="lazy" src="'+j['report_image']+'"></a>')
                        parts.append('<small>'+str(round(j['result']['generation_seconds'],2))+' s · '+str(round(j['result']['peak_allocated_bytes']/2**30,2))+' GiB</small>')
                        if j.get('prepared_report_images'):
                            parts.append('<details><summary>实际预处理输入</summary>'+''.join('<a href="'+a['path']+'">'+esc(a['id'])+'</a> ' for a in j['prepared_report_images'])+'</details>')
                        for scored in [s for s in scores if s['job_id']==j['id']]:parts.append('<small>'+esc(scored['scorer']+': '+json.dumps(scored['result'],ensure_ascii=False))+'</small>')
                    else:parts.append('<span class="'+('failed' if j['state']=='failed' else 'muted')+'">'+esc(j['state'])+'</span><small>'+esc(j['error'] or j.get('report_error') or '')+'</small>')
                    parts.append('</td>')
                parts.append('</tr>')
            exemplar=index[(columns[0]['id'],case['case_id'],suite['seeds'][0])].get('request',{})
            parts.append('</tbody></table></div><details><summary>冻结参数与案例身份</summary><pre>'+esc(json.dumps({'profile':exemplar.get('profile',spec['profiles'][family][task]),'geometry':exemplar.get('geometry'),'case_digest':case['digest'],'provenance':case['provenance']},ensure_ascii=False,indent=2))+'</pre></details></section>')
            if boards:_board(out,case,columns,suite['seeds'],index,run_dir,family)
    parts.append('''<footer><span>FIELDNOTE / FIXED CONDITIONS. VISIBLE PROGRESS.</span><span>END OF REPORT</span></footer></main><script>const task=document.getElementById('task'),seed=document.getElementById('seed');function filter(){document.querySelectorAll('section[data-task]').forEach(x=>x.hidden=!!task.value&&x.dataset.task!==task.value);document.querySelectorAll('tr[data-seed]').forEach(x=>x.hidden=!!seed.value&&x.dataset.seed!==seed.value)}task.onchange=filter;seed.onchange=filter</script></body></html>''')
    atomic_bytes(out/'theme.css',(Path(__file__).parent/'static/theme.css').read_bytes())
    atomic_bytes(out/'index.html',''.join(parts).encode())
    return str(out/'index.html')

def _board(out,case,models,seeds,index,run_dir,family):
    cell=288;label=130;top=80
    canvas=Image.new('RGB',(label+cell*len(models),top+cell*len(seeds)),(245,247,250));draw=ImageDraw.Draw(canvas)
    draw.text((12,12),f"{family} / {case['task']} / {case['provenance']['row_id']}",fill='black')
    for c,m in enumerate(models):draw.text((label+c*cell+8,45),m['id'][:42],fill='black')
    for r,seed in enumerate(seeds):
        y=top+r*cell;draw.text((10,y+12),str(seed),fill='black')
        for c,m in enumerate(models):
            job=index[(m['id'],case['case_id'],seed)];x=label+c*cell
            try:
                if job['state']!='succeeded':raise ValueError(job['state'])
                with Image.open(inside(run_dir,job['result']['image'])) as im:
                    tile=ImageOps.contain(im.convert('RGB'),(cell-12,cell-12));canvas.paste(tile,(x+(cell-tile.width)//2,y+(cell-tile.height)//2))
            except (OSError,ValueError,KeyError):draw.text((x+8,y+12),job['state'],fill='red')
    folder=out/'boards';folder.mkdir(exist_ok=True);canvas.save(folder/(family+'-'+case['case_id']+'.png'))

def export_zip(store,run_id):
    export_pdf(store,run_id);build_report(store,run_id,boards=True);root=store.root/'runs'/run_id
    destination=root/(run_id+'-report.zip');tmp=destination.with_suffix('.partial')
    with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED) as archive:
        for p in sorted((root/'report').rglob('*')):
            if p.is_file():archive.write(p,p.relative_to(root/'report').as_posix())
        for p in sorted((root/'items').rglob('*')):
            if p.is_file() and p.suffix in ('.json','.log'):archive.write(p,'logs/'+p.relative_to(root/'items').as_posix())
        for p in sorted(root.glob('*.log')):
            archive.write(p,'logs/'+p.name)
        for p in sorted(root.glob('load-*.json')):
            archive.write(p,'logs/'+p.name)
    tmp.replace(destination);return str(destination)


def export_pdf(store,run_id):
    from .pdf_report import build_pdf
    report_path=Path(build_report(store,run_id)).parent
    path=build_pdf(report_path/'report.json')
    # Rebuild navigation after the PDF exists, also for offline HTML exports.
    build_report(store,run_id)
    return path
