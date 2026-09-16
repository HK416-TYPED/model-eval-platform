"""Compact review pages with inputs and results kept together."""
import io,json,os,tempfile
from pathlib import Path
from xml.sax.saxutils import escape
from PIL import Image as PILImage,ImageOps
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4,landscape
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,PageBreak,Table,TableStyle,Image,KeepTogether
from .core import inside,read_json
from .review import summarize,VERDICTS,TASK_LABELS

INK=colors.HexColor('#20211e');MUTED=colors.HexColor('#707367');LINE=colors.HexColor('#d4d7c9')
WIDTH,HEIGHT=landscape(A4);CONTENT_WIDTH=WIDTH-72

def register_font(font_path=None):
    for path in [font_path,os.environ.get('EVAL_PDF_FONT'),Path(__file__).parent/'assets/fonts/wqy-microhei.ttc','/usr/share/fonts/truetype/wqy/wqy-microhei.ttc','C:/Windows/Fonts/simhei.ttf']:
        if path and Path(path).is_file():
            pdfmetrics.registerFont(TTFont('FieldnoteCJK',str(path),subfontIndex=0));return 'FieldnoteCJK'
    raise RuntimeError('PDF 中文字体缺失：运行 tools/install_pdf_font.sh 或设置 EVAL_PDF_FONT')

def build_pdf(report_json,destination=None,font_path=None):
    report_json=Path(report_json).resolve();report=read_json(report_json);root=report_json.parent
    destination=Path(destination or root/'report.pdf');destination.parent.mkdir(parents=True,exist_ok=True)
    font=register_font(font_path)
    styles={name:ParagraphStyle(name,fontName=font,fontSize=size,leading=leading,textColor=MUTED if name=='small' else INK,wordWrap='CJK',spaceAfter=3) for name,size,leading in [('body',9,13),('small',7,9),('case',11,14),('heading',16,21),('title',25,32)]}
    def p(text,style='body'):return Paragraph(escape(str(text)).replace('\n','<br/>'),styles[style])
    def excerpt(text,limit=150):
        text=' '.join(str(text).split());return text if len(text)<=limit else text[:limit]+'…（全文见附录）'
    cache={}
    def picture(path,width,height):
        source=inside(root,path);key=(source,round(width*135/72),round(height*135/72))
        if key not in cache:
            with PILImage.open(source) as original:
                im=ImageOps.exif_transpose(original).convert('RGB');im.thumbnail(key[1:],PILImage.Resampling.LANCZOS)
                stream=io.BytesIO();im.save(stream,'JPEG',quality=82,optimize=True);cache[key]=(stream.getvalue(),im.size)
        raw,(w,h)=cache[key];scale=min(width/w,height/h)
        image=Image(io.BytesIO(raw),width=w*scale,height=h*scale);image.hAlign='CENTER';return image
    def safe_picture(path,width,height):
        try:return picture(path,width,height)
        except (OSError,ValueError):return p('图像不可用','small')
    def grid(rows,widths,header=False):
        table=Table(rows,colWidths=widths,hAlign='LEFT',repeatRows=int(header))
        table.setStyle(TableStyle([('VALIGN',(0,0),(-1,-1),'TOP'),('BOX',(0,0),(-1,-1),.4,LINE),('INNERGRID',(0,0),(-1,-1),.3,LINE),('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]));return table
    summary,suite,spec=report['summary'],report['suite'],report['experiment']
    jobs={(j['model_id'],j['case_id'],j['seed']):j for j in report['jobs']}
    reviews=report.get('reviews',[]);review_index={(r['model_id'],r['case_id']):r for r in reviews}
    analysis=summarize(report['jobs'],reviews);counts=summary['counts']
    story=[p('FIELDNOTE / 评审报告','small'),p('模型效果测试报告','title'),p(summary['id'],'heading'),p(f"计划 {summary['planned']} 张 · 已生成 {counts.get('succeeded',0)} · 失败 {counts.get('failed',0)} · 待运行/运行中 {counts.get('queued',0)+counts.get('running',0)} · 取消 {counts.get('cancelled',0)}"),p('运行统计与评审摘要','heading'),p(analysis['scope'],'small')]
    rows=[[p(x,'small') for x in ['checkpoint / 任务','已生成 / 计划','失败','成功率','耗时中位数','峰值显存']]]
    for g in analysis['groups']:
        rate,seconds,memory=g['success_rate'],g['median_seconds'],g['peak_gib']
        rows.append([p(g['model_id']+' / '+TASK_LABELS.get(g['task'],g['task']),'small'),p(f"{g['succeeded']} / {g['planned']}",'small'),p(g['failed'],'small'),p(f'{rate:.1%}' if rate is not None else '无已完成项','small'),p(f"{seconds:.2f}s (n={g['timing_samples']})" if seconds is not None else '未记录','small'),p(f"{memory:.2f} GiB (n={g['memory_samples']})" if memory is not None else '未记录','small')])
    story += [grid(rows,[260,95,55,80,140,CONTENT_WIDTH-630],True),Spacer(1,10),p(f"人工已评审 {analysis['reviewed_cases']} 个模型/案例组合；"+'、'.join(f'{VERDICTS[k]} {v}' for k,v in analysis['review_verdicts'].items())),p('未评分的图像不记为 0 分。视觉质量以人工评语或独立评分记录为依据；运行统计不构成画质排名。'),p('固定案例与 seed 全量展示，无择优挑图。训练集复测不能代替未见数据验证。','small')]
    if len(spec['models'])==1:story.append(p('本次仅一个 checkpoint，尚不能判断版本间提升。','small'))
    for item in summary.get('unavailable',[]):story.append(p('待接入：'+str(item.get('reason','等待资源')),'small'))
    for f in analysis['failures'][:4]:story.append(p(f"失败定位：{f['model_id']} / {f['case_id']} / seed {f['seed']}（完整原因见附录）",'small'))
    def result_cell(model,case,seed,width):
        job=jobs.get((model['id'],case['case_id'],seed));content=[p('SEED '+str(seed),'small')]
        if not job:return content+[p('未安排此项','small')]
        if job['state']=='succeeded' and job.get('report_image'):
            content.append(safe_picture(job['report_image'],width-18,86));seconds=(job.get('result') or {}).get('generation_seconds')
            memory=(job.get('result') or {}).get('peak_allocated_bytes')
            metrics=f'{seconds:.2f} s' if isinstance(seconds,(int,float)) else '耗时未记录'
            if isinstance(memory,(int,float)):metrics+=f' · {memory/2**30:.2f} GiB'
            content.append(p(metrics,'small'))
        else:content += [p({'failed':'失败','queued':'待运行','running':'运行中','cancelled':'已取消'}.get(job['state'],job['state'])),p(excerpt(job.get('error') or job.get('report_error') or '',72),'small')]
        return content
    number=0
    story.append(PageBreak())
    for family in sorted({m['family'] for m in spec['models']}):
        for case in sorted(suite['cases'],key=lambda c:({'edit_dual':0,'edit_single':1,'t2i':2}[c['task']],c['case_id'])):
            models=[m for m in spec['models'] if m['family']==family and any((m['id'],case['case_id'],s) in jobs for s in suite['seeds'])]
            if not models:continue
            number+=1;single=len(models)==1;groups=[models[i:i+3] for i in range(0,len(models),3)];per_page=5 if single else 2
            total=len(groups)*((len(suite['seeds'])+per_page-1)//per_page);part=0
            for group in groups:
                for offset in range(0,len(suite['seeds']),per_page):
                    seeds=suite['seeds'][offset:offset+per_page];part+=1
                    block = [p(f"{number:03d} / {TASK_LABELS[case['task']]} / {family.upper()}  {part}/{total}",'case'),p('案例 '+case['case_id']+' · '+excerpt(', '.join(m['id'] for m in group),125),'small')]
                    assets=[(f'输入 {i+1}',a) for i,a in enumerate(case['inputs'])]
                    if case.get('dataset_target'):assets.append(('目标图 · 仅对照',case['dataset_target']))
                    refs=[ [p(label,'small'),safe_picture(a['path'],68,44)] for label,a in assets ]
                    context=refs+[[p('指令 / PROMPT','small'),p(excerpt(case['prompt'],210),'small')]]
                    block.append(grid([context],[80]*len(refs)+[CONTENT_WIDTH-80*len(refs)]))
                    block.append(Spacer(1,4));width=CONTENT_WIDTH/(max(1,len(seeds)) if single else len(group))
                    if single:
                        cells=[result_cell(group[0],case,s,width) for s in seeds]
                        block.append(grid([cells],[width]*len(cells)))
                    else:
                        rows=[[p(excerpt(m['id'],65),'small') for m in group]]+[[result_cell(m,case,s,width) for m in group] for s in seeds]
                        block.append(grid(rows,[width]*len(group),True))
                    notes=[]
                    for model in group:
                        review=review_index.get((model['id'],case['case_id']))
                        if review:notes.append(f"{model['id']}：{VERDICTS[review['verdict']]}；{review['note']}")
                    block += [Spacer(1,3),p('人工评审：'+excerpt(' / '.join(notes),140) if notes else '人工评审：待评审。','small'),Spacer(1,10)]
                    story.append(KeepTogether(block))
    story += [PageBreak(),p('附录 / 完整指令与复现信息','heading')]
    for case in suite['cases']:story += [p('案例 '+case['case_id']),p(case['prompt']),p('来源：'+json.dumps(case.get('provenance',{}),ensure_ascii=False),'small')]
    for model in spec['models']:story += [p(model['id']),p('SHA256: '+str(model.get('identity',{}).get('assets',{}).get('checkpoint',{}).get('sha256','未记录')),'small')]
    story += [p('冻结参数：'+json.dumps(spec.get('profiles',{}),ensure_ascii=False),'small'),p('套件摘要：'+str(suite.get('digest','未记录')),'small'),p('数据版本：'+json.dumps(suite.get('sources',[]),ensure_ascii=False),'small')]
    for review in reviews:story += [p('人工评审 / '+review['model_id']+' / '+review['case_id']),p(VERDICTS[review['verdict']]+' · '+(review['reviewer'] or '未署名')),p(review['note'])]
    for f in analysis['failures']:story.append(p('失败详情 / '+f['model_id']+' / '+f['case_id']+' / seed '+str(f['seed'])+'：'+f['error'],'small'))
    for score in report.get('scores',[]):story.append(p('独立评分：'+json.dumps(score,ensure_ascii=False),'small'))
    story.append(p('PDF 使用适配版面尺寸的压缩预览。原始图片、完整参数和日志保存在离线 ZIP 中。','small'))
    def frame(canvas,doc):
        canvas.saveState();canvas.setFillColor(INK);canvas.rect(0,HEIGHT-27,WIDTH,27,fill=1,stroke=0);canvas.setFillColor(colors.HexColor('#e5f34a'));canvas.setFont('Helvetica',9);canvas.drawString(36,HEIGHT-18,'FIELDNOTE / COMPACT REVIEW / v0.3');canvas.setFillColor(MUTED);canvas.setFont('Helvetica',8);canvas.drawString(36,16,'FIXED CASES / MEASURED EXECUTION / HUMAN REVIEW');canvas.drawRightString(WIDTH-36,16,str(doc.page));canvas.restoreState()
    handle,temporary=tempfile.mkstemp(prefix='report-',suffix='.pdf.partial',dir=destination.parent);os.close(handle)
    try:
        doc=SimpleDocTemplate(temporary,pagesize=(WIDTH,HEIGHT),leftMargin=36,rightMargin=36,topMargin=39,bottomMargin=32,title='FIELDNOTE / '+summary['id'],author='Model Evaluation Platform',pageCompression=1)
        doc.build(story,onFirstPage=frame,onLaterPages=frame);Path(temporary).replace(destination)
    finally:Path(temporary).unlink(missing_ok=True)
    return str(destination)
