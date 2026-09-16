import html
import json
import shutil
import zipfile
from pathlib import Path
from PIL import Image, ImageDraw, ImageOps
from .core import atomic_bytes, inside, read_json, write_json
from .review import summarize, VERDICTS


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
                j['report_preview']=preview.relative_to(out).as_posix()
            except OSError:j['report_error']='Output file unavailable'
            j['prepared_report_images']=[{'id':a['id'],'path':copy_asset(inside(run_dir,a['path']))} for a in j['result'].get('preprocessed_inputs',[])]
    with store.connect() as db:
        scores=[{'job_id':r['job_id'],'scorer':r['scorer'],'result':json.loads(r['result'])} for r in db.execute('SELECT * FROM scores WHERE run_id=?',(run_id,))]
    reviews=store.reviews(run_id)
    analysis=summarize(jobs,reviews)
    report={'schema':'eval-report/v1','summary':summary,'suite':suite,'experiment':spec,'jobs':jobs,'scores':scores,'reviews':reviews,'analysis':analysis,
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
           '<title>'+esc(run_id)+' · FIELDNOTE 图像报告</title><link rel="stylesheet" href="theme.css"><link rel="stylesheet" href="report.css?v=3"><body class="compact-report"><main class="report-shell">',
           '<div class="topline"><span class="eyebrow">FIELDNOTE / IMAGE REPORT</span><span class="mono">TRAINING-SET REPLAY</span></div>',
           '<header class="hero report-hero"><div><h1>模型效果测试报告</h1><p class="run-name">'+esc(run_id)+'</p></div><div class="report-totals">'+str(len(suite['cases']))+' 案例 · '+str(len(suite['seeds']))+' seeds · '+str(len(spec['models']))+' checkpoints</div></header>',
           '<p class="status">'+esc(state_label)+' · 成功 '+str(summary['counts']['succeeded'])+' / '+str(summary['planned'])+'</p>',
           '<p class="muted report-scope">每组固定案例，可横向浏览 seed 或对比 checkpoint。目标图仅作对照；Edit 可使用其宽高比。评分：'+('已配置' if scores else '未配置')+'。点击图片放大或并排查看参考图。</p>',
           '<nav class="report-nav"><label for="task" style="margin:0">任务</label><select id="task"><option value="">全部</option><option value="edit_dual">双图参考</option><option value="edit_single">单图参考</option><option value="t2i">文生图</option></select><label for="seed" style="margin:0">Seed</label><select id="seed"><option value="">全部</option>'+''.join('<option>'+str(s)+'</option>' for s in suite['seeds'])+'</select><a href="report.json">JSON</a> · <a href="results.jsonl">逐项记录</a>'+(' · <a class="button primary" href="report.pdf?v='+str((out/'report.pdf').stat().st_mtime_ns)+'">下载 PDF ↓</a>' if (out/'report.pdf').is_file() else '')+'</nav>']
    parts.append('<div class="report-tools"><label for="caseSearch">搜索案例 / 指令<input id="caseSearch" type="search" placeholder="案例编号或指令关键词"></label><label for="modelFilter">Checkpoint<select id="modelFilter"><option value="">全部</option>'+''.join('<option value="'+esc(m['id'])+'">'+esc(m['id'])+'</option>' for m in spec['models'])+'</select></label><label for="density">图像密度<select id="density"><option value="compact">紧凑</option><option value="comfortable">大图</option></select></label><span id="caseCount" role="status"></span></div>')
    for unavailable in spec.get('unavailable',[]):
        reason=str(unavailable.get('reason','等待资源')).split(': ',1)[-1]
        parts.append('<p class="muted">待接入：'+esc(reason)+'</p>')
    parts.append('<details><summary>抽样范围与数据版本</summary><pre>'+esc(json.dumps(suite['sources'],ensure_ascii=False,indent=2))+'</pre></details>')
    case_number=0
    parts.append('<details class="panel report-statistics"><summary>运行统计与人工评审 · 已评审 '+str(analysis['reviewed_cases'])+' 个模型 / 案例组合</summary><p>'+esc(analysis['scope'])+'</p><div class="table-scroll"><table><tr><th>模型 / 任务</th><th>已生成 / 计划</th><th>失败</th><th>耗时中位数</th></tr>')
    for g in analysis['groups']:
        parts.append('<tr><td>'+esc(g['model_id']+' / '+g['task'])+'</td><td>'+str(g['succeeded'])+' / '+str(g['planned'])+'</td><td>'+str(g['failed'])+'</td><td>'+ (str(round(g['median_seconds'],2))+' s' if g['median_seconds'] is not None else '未记录')+'</td></tr>')
    parts.append('</table></div></details>')
    for family in sorted({m['family'] for m in spec['models']}):
        models=[m for m in spec['models'] if m['family']==family]
        for case in sorted(suite['cases'],key=lambda c:({'edit_dual':0,'edit_single':1,'t2i':2}[c['task']],c['case_id'])):
            columns=[m for m in models if any(j['model_id']==m['id'] and j['case_id']==case['case_id'] for j in jobs)]
            if not columns:continue
            task=case['task'];case_number+=1
            task_label={'edit_dual':'双图参考','edit_single':'单图参考','t2i':'文生图'}[task]
            parts.append('<section class="panel report-case" data-task="'+esc(task)+'" data-case="'+esc(case['case_id'])+'"><div class="case-head"><span class="case-number">'+str(case_number).zfill(3)+'</span><h2>'+esc(family.upper()+' / '+task_label)+'</h2><span class="mono">CASE / '+esc(case['provenance']['row_id'])+'</span></div><div class="case-context"><div class="refs">')
            for label,asset in [(f'输入 {i+1}',a) for i,a in enumerate(case['inputs'])]+([('数据集目标图（仅对照）',case['dataset_target'])] if 'dataset_target' in case else []):
                src=copy_asset(inside(spec['suite_dir'],asset['path']))
                thumb=out/'previews'/(Path(src).stem+'-ref.webp');thumb.parent.mkdir(exist_ok=True)
                if not thumb.exists():
                    with Image.open(out/src) as original:
                        im=ImageOps.exif_transpose(original).convert('RGB');im.thumbnail((480,480));im.save(thumb,'WEBP',quality=82)
                parts.append('<figure><a href="'+esc(src)+'"><img loading="lazy" src="'+esc(thumb.relative_to(out).as_posix())+'" alt="'+esc(label)+'"></a><figcaption>'+esc(label)+'</figcaption></figure>')
            parts.append('</div><div class="case-instruction"><span class="eyebrow muted">指令 / PROMPT</span><pre class="case-prompt">'+esc(case['prompt'])+'</pre><p class="case-provenance">'+esc(case['semantic_mode'])+' · '+esc(case['provenance']['source_id'])+'</p></div></div>')
            for m in columns:
                parts.append('<div class="checkpoint-group" data-model="'+esc(m['id'])+'"><div class="checkpoint-heading"><b>'+esc(m['id'])+'</b><span class="mono">SHA '+esc(m['identity']['assets']['checkpoint']['sha256'][:12])+'</span></div><div class="result-grid">')
                for seed in suite['seeds']:
                    j=index[(m['id'],case['case_id'],seed)];parts.append('<div class="result-card" data-seed="'+str(seed)+'"><div class="result-label mono">SEED '+str(seed)+'</div>')
                    if j.get('report_image'):
                        parts.append('<a class="result-image" href="'+esc(j['report_image'])+'"><img loading="lazy" src="'+esc(j.get('report_preview',j['report_image']))+'" alt="'+esc(m['id']+' / seed '+str(seed))+'"></a>')
                        parts.append('<small class="result-metrics">'+str(round(j['result']['generation_seconds'],2))+' s <span>·</span> '+str(round(j['result']['peak_allocated_bytes']/2**30,2))+' GiB</small>')
                        if j.get('prepared_report_images'):
                            parts.append('<details><summary>实际预处理输入</summary>'+''.join('<a href="'+a['path']+'">'+esc(a['id'])+'</a> ' for a in j['prepared_report_images'])+'</details>')
                        for scored in [s for s in scores if s['job_id']==j['id']]:parts.append('<small>'+esc(scored['scorer']+': '+json.dumps(scored['result'],ensure_ascii=False))+'</small>')
                    else:parts.append('<div class="result-missing '+('failed' if j['state']=='failed' else 'muted')+'">'+esc({'failed':'生成失败','queued':'等待生成','running':'生成中','cancelled':'已取消'}.get(j['state'],j['state']))+'</div><small>'+esc(j['error'] or j.get('report_error') or '')+'</small>')
                    parts.append('</div>')
                parts.append('</div></div>')
            exemplar=index[(columns[0]['id'],case['case_id'],suite['seeds'][0])].get('request',{})
            parts.append('<details class="case-parameters"><summary>冻结参数与案例身份</summary><pre>'+esc(json.dumps({'profile':exemplar.get('profile',spec['profiles'][family][task]),'geometry':exemplar.get('geometry'),'case_digest':case['digest'],'provenance':case['provenance']},ensure_ascii=False,indent=2))+'</pre></details>')
            if boards:_board(out,case,columns,suite['seeds'],index,run_dir,family)
            for review in reviews:
                if review['case_id']==case['case_id'] and review['model_id'] in {m['id'] for m in columns}:
                    parts.append('<aside class="panel"><b>'+esc(review['model_id']+' · '+VERDICTS[review['verdict']])+'</b><p style="white-space:pre-wrap">'+esc(review['note'])+'</p><small>'+esc(review['reviewer'])+'</small></aside>')
            parts.append('</section>')
    parts.append('''<p id="noCases" hidden>没有匹配的案例，请调整筛选条件。</p><nav class="report-pagination" aria-label="案例分页"><button id="prevPage" class="secondary">上一页</button><span id="pageInfo" aria-live="polite"></span><button id="nextPage" class="secondary">下一页</button><label for="pageSize">每页<select id="pageSize"><option value="12">12 案例</option><option value="24">24 案例</option><option value="0">全部</option></select></label></nav><footer><span>FIELDNOTE / FIXED CONDITIONS. VISIBLE PROGRESS.</span><span>END OF REPORT</span></footer></main><script src="report-ui.js?v=3"></script></body></html>''')
    atomic_bytes(out/'theme.css',(Path(__file__).parent/'static/theme.css').read_bytes())
    atomic_bytes(out/'viewer.js',(Path(__file__).parent/'static/viewer.js').read_bytes())
    for name in ['report.css','report-ui.js']:
        atomic_bytes(out/name,(Path(__file__).parent/'static'/name).read_bytes())
    parts[-1]=parts[-1].replace('</body>', '<script src="viewer.js?v=3"></script></body>')
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
