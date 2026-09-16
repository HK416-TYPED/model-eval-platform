"""Portable, embedded-font PDF reports from an existing report snapshot."""
import io
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from xml.sax.saxutils import escape

from PIL import Image as PILImage, ImageOps
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, PageBreak,
                               Table, TableStyle, Image)

from .core import inside, read_json

INK = colors.HexColor('#20211e')
MUTED = colors.HexColor('#707367')
LINE = colors.HexColor('#d4d7c9')
PAPER = colors.HexColor('#f7f8f2')
ACID = colors.HexColor('#e5f34a')
TASKS = {'edit_dual': '双图参考', 'edit_single': '单图参考', 't2i': '文生图'}
STATES = {'queued': '待运行', 'running': '运行中', 'succeeded': '已生成',
          'completed': '已完成', 'failed': '失败', 'cancelled': '已取消',
          'completed_with_errors': '部分失败', 'unavailable': '待接入'}
WIDTH, HEIGHT = landscape(A4)
CONTENT_WIDTH = WIDTH - 80


def register_font(font_path=None):
    candidates = [font_path, os.environ.get('EVAL_PDF_FONT'),
                  Path(__file__).parent/'assets/fonts/wqy-microhei.ttc',
                  '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
                  'C:/Windows/Fonts/simhei.ttf']
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            # Re-registration is harmless and lets a caller change the font explicitly.
            pdfmetrics.registerFont(TTFont('FieldnoteCJK', str(candidate), subfontIndex=0))
            return 'FieldnoteCJK'
    raise RuntimeError('PDF 中文字体缺失：运行 tools/install_pdf_font.sh，或设置 EVAL_PDF_FONT 为中文 TrueType 字体路径')


def build_pdf(report_json, destination=None, font_path=None):
    report_json = Path(report_json).resolve()
    report = read_json(report_json)
    root = report_json.parent
    destination = Path(destination or root/'report.pdf')
    destination.parent.mkdir(parents=True, exist_ok=True)
    font = register_font(font_path)
    styles = {
        'body': ParagraphStyle('body', fontName=font, fontSize=9, leading=14, textColor=INK,
                               spaceAfter=8, wordWrap='CJK'),
        'small': ParagraphStyle('small', fontName=font, fontSize=7.5, leading=11,
                                textColor=MUTED, wordWrap='CJK', spaceAfter=4),
        'title': ParagraphStyle('title', fontName=font, fontSize=30, leading=39,
                                textColor=INK, spaceAfter=18, wordWrap='CJK'),
        'heading': ParagraphStyle('heading', fontName=font, fontSize=17, leading=23,
                                  textColor=INK, spaceAfter=12, wordWrap='CJK'),
        'label': ParagraphStyle('label', fontName='Helvetica', fontSize=9, leading=13,
                                textColor=MUTED, spaceAfter=10),
    }
    def p(text, style='body'):
        return Paragraph(escape(str(text)).replace('\n', '<br/>'), styles[style])

    image_cache = {}
    def picture(path, max_width, max_height):
        source = inside(root, path)
        if source not in image_cache:
            with PILImage.open(source) as original:
                im = ImageOps.exif_transpose(original).convert('RGB')
                im.thumbnail((1400, 1400), PILImage.Resampling.LANCZOS)
                stream = io.BytesIO()
                im.save(stream, format='JPEG', quality=92, subsampling=0)
                image_cache[source] = (stream.getvalue(), im.size)
        raw, (w, h) = image_cache[source]
        factor = min(max_width/w, max_height/h)
        im = Image(io.BytesIO(raw), width=w*factor, height=h*factor)
        im.hAlign = 'CENTER'
        return im

    def grid(rows, widths, header=False):
        table = Table(rows, colWidths=widths, hAlign='LEFT', repeatRows=int(header))
        commands = [('VALIGN', (0, 0), (-1, -1), 'TOP'),
                    ('LEFTPADDING', (0, 0), (-1, -1), 10), ('RIGHTPADDING', (0, 0), (-1, -1), 10),
                    ('TOPPADDING', (0, 0), (-1, -1), 9), ('BOTTOMPADDING', (0, 0), (-1, -1), 9),
                    ('BOX', (0, 0), (-1, -1), .5, LINE), ('INNERGRID', (0, 0), (-1, -1), .4, LINE)]
        if header:
            commands.append(('BACKGROUND', (0, 0), (-1, 0), PAPER))
        table.setStyle(TableStyle(commands))
        return table

    summary, suite, spec = report['summary'], report['suite'], report['experiment']
    run_id = summary['id']
    jobs = {(j['model_id'], j['case_id'], j['seed']): j for j in report['jobs']}
    scores = defaultdict(list)
    for score in report.get('scores', []):
        scores[score['job_id']].append(score)
    story = [Spacer(1, 20), p('FIELDNOTE / MODEL EVALUATION LAB', 'label'),
             p('模型效果测试报告', 'title'), p(run_id, 'heading'),
             p('训练集固定案例对比 / 同模型不同 epoch', 'body'), Spacer(1, 16)]
    counts = summary['counts']
    metrics = grid([[p('计划输出'), p('成功'), p('失败'), p('状态')],
                    [p(summary['planned'], 'heading'), p(counts['succeeded'], 'heading'),
                     p(counts['failed'], 'heading'), p(STATES.get(summary['state'], summary['state']), 'heading')]],
                   [CONTENT_WIDTH/4]*4, header=True)
    metrics.setStyle(TableStyle([('BACKGROUND', (1, 1), (1, 1), ACID)]))
    story += [metrics, Spacer(1, 22),
              p(f"冻结套件包含 {len(suite['cases'])} 个案例，固定 seed：" + ', '.join(map(str, suite['seeds']))),
              p('每个任务使用同一组输入、原始 prompt 和采样设置。目标图像素不参与生成；Edit 可使用目标图宽高比确定输出尺寸。'),
              p(f"单个 checkpoint 展示 {len(suite['seeds'])} 个 seed 的图像图版；多个 checkpoint 按相同 case + seed 并排列示，每组最多 3 列。"),
              p('本报告反映训练集固定案例上的表现，不代表未见数据的泛化结果。'),
              p('评分状态：' + ('已接入；详见文末评分记录。' if scores else '未评分。未配置或失败的评分不记为 0 分。'))]
    for missing in summary.get('unavailable', []):
        reason=str(missing.get('reason','等待资源')).split(': ',1)[-1]
        story.append(p('待接入：'+reason, 'small'))

    case_number = 0
    for family in sorted({m['family'] for m in spec['models']}):
        family_models = [m for m in spec['models'] if m['family'] == family]
        for case in sorted(suite['cases'], key=lambda c: ({'edit_dual': 0, 'edit_single': 1, 't2i': 2}[c['task']], c['case_id'])):
            models = [m for m in family_models if any((m['id'], case['case_id'], s) in jobs for s in suite['seeds'])]
            if not models:
                continue
            case_number += 1
            title = f"{case_number:02d} / {family.upper()} / {TASKS[case['task']]}"
            story += [PageBreak(), p('CASE FILE / INPUT CONDITIONS', 'label'), p(title, 'heading'),
                      p(f"案例 {case['case_id']} · 来源 ID {case.get('provenance', {}).get('row_id', '-')}", 'small'),
                      p('原始 PROMPT / INSTRUCTION', 'label')]
            # Paragraphs can split across pages: prompts are never truncated or shrunk to fit.
            for line in case['prompt'].splitlines() or ['']:
                story.append(p(line))
            assets = [(f'输入 {i+1}', a) for i, a in enumerate(case['inputs'])]
            if 'dataset_target' in case:
                assets.append(('数据集目标图（仅对照）', case['dataset_target']))
            if assets:
                cells = []
                for label, asset in assets:
                    try:
                        content = [p(label, 'small'), picture(asset['path'], 205, 174)]
                    except (OSError, ValueError) as error:
                        content = [p(label), p('图像不可用：' + str(error), 'small')]
                    cells.append(content)
                story += [Spacer(1, 12), grid([cells], [CONTENT_WIDTH/len(cells)]*len(cells))]
            provenance = case.get('provenance', {})
            exemplar = jobs.get((models[0]['id'],case['case_id'],suite['seeds'][0]),{}).get('request',{})
            story += [Spacer(1, 12), p('数据来源：' + str(provenance.get('source_id', '-')), 'small'),
                      p('语义模式：' + case.get('semantic_mode', '-') + ' / 输入顺序与冻结套件一致。', 'small'),
                      p('推理参数：' + json.dumps(exemplar.get('profile',spec['profiles'][family][case['task']]), ensure_ascii=False), 'small')]
            if exemplar.get('geometry'):story.append(p('采样尺寸：'+json.dumps(exemplar['geometry'],ensure_ascii=False),'small'))

            def result_cell(model, seed, max_width, max_height):
                job = jobs.get((model['id'], case['case_id'], seed))
                result = [p('SEED ' + str(seed), 'label')]
                if not job:
                    return result + [p('未安排此项')]
                if job['state'] == 'succeeded' and job.get('report_image'):
                    try:
                        result.append(picture(job['report_image'], max_width, max_height))
                        info = job.get('result') or {}
                        seconds = info.get('generation_seconds')
                        memory = info.get('peak_allocated_bytes')
                        result.append(Spacer(1, 5))
                        result.append(p((f'{seconds:.2f} s' if seconds is not None else '耗时未记录') +
                                        (f' / {memory/2**30:.2f} GiB' if memory is not None else ''), 'small'))
                        return result
                    except (OSError, ValueError) as error:
                        return result + [p('输出图像不可用：' + str(error), 'small')]
                return result + [p(STATES.get(job['state'], job['state'])),
                                 p(job.get('error') or job.get('report_error') or '', 'small')]

            if len(models) == 1:
                story += [PageBreak(), p('OUTPUT ATLAS / FIXED SEEDS', 'label'), p(title, 'heading'),
                          p(models[0]['id'], 'small')]
                cells = [result_cell(models[0], seed, 210, 150) for seed in suite['seeds']]
                while len(cells) % 3:
                    cells.append('')
                story.append(grid([cells[i:i+3] for i in range(0, len(cells), 3)], [CONTENT_WIDTH/3]*3))
            else:
                for group_start in range(0, len(models), 3):
                    group = models[group_start:group_start+3]
                    for seed_start in range(0, len(suite['seeds']), 2):
                        story += [PageBreak(), p('CHECKPOINT COMPARISON / SAME CASE + SEED', 'label'),
                                  p(title, 'heading'),
                                  p(f"模型列 {group_start+1}-{group_start+len(group)} / 共 {len(models)} 个 checkpoint", 'small')]
                        cell_width = CONTENT_WIDTH/len(group)
                        rows = [[p(m['id'], 'small') for m in group]]
                        for seed in suite['seeds'][seed_start:seed_start+2]:
                            rows.append([result_cell(m, seed, cell_width-26, 130) for m in group])
                        story.append(grid(rows, [cell_width]*len(group), header=True))

    story += [PageBreak(), p('APPENDIX / REPRODUCIBILITY', 'label'), p('版本与数据来源', 'heading')]
    for model in spec['models']:
        checkpoint = model.get('identity', {}).get('assets', {}).get('checkpoint', {})
        story += [p(model['id']), p('SHA256: ' + str(checkpoint.get('sha256', '未记录')), 'small')]
    for source in suite.get('sources', []):
        story.append(p(json.dumps(source, ensure_ascii=False), 'small'))
    story += [p('套件摘要：' + str(suite.get('digest', '未记录')), 'small'),
              p('原始尺寸图片、完整运行环境、逐项 JSON 和日志保存在完整离线 ZIP 报告中。PDF 图片按比例排版，嵌入最高 1400 像素预览。', 'small')]
    if scores:
        story += [PageBreak(), p('SCORING / RECORDED RESULTS', 'label'), p('评分记录', 'heading')]
        for job in report['jobs']:
            for score in scores[job['id']]:
                story += [p(f"{job['model_id']} / {job['case_id']} / seed {job['seed']}"),
                          p(score['scorer'] + ': ' + json.dumps(score['result'], ensure_ascii=False), 'small')]

    def page_frame(canvas, doc):
        canvas.saveState()
        canvas.setFillColor(INK)
        canvas.rect(0, HEIGHT-29, WIDTH, 29, fill=1, stroke=0)
        canvas.setFillColor(ACID)
        canvas.setFont('Helvetica-Bold', 9)
        canvas.drawString(40, HEIGHT-19, 'FIELDNOTE / MODEL EVALUATION LAB')
        canvas.setStrokeColor(LINE)
        canvas.line(40, 28, WIDTH-40, 28)
        canvas.setFillColor(MUTED)
        canvas.setFont('Helvetica', 8)
        canvas.drawString(40, 15, 'TRAINING-SET REPLAY / FIXED CONDITIONS')
        canvas.drawRightString(WIDTH-40, 15, f'{doc.page:03d}')
        canvas.restoreState()

    handle, temporary = tempfile.mkstemp(prefix='report-', suffix='.pdf.partial', dir=destination.parent)
    os.close(handle)
    try:
        doc = SimpleDocTemplate(temporary, pagesize=(WIDTH, HEIGHT), leftMargin=40, rightMargin=40,
                                topMargin=48, bottomMargin=42, title='FIELDNOTE / '+run_id,
                                author='Model Evaluation Platform', pageCompression=1)
        doc.build(story, onFirstPage=page_frame, onLaterPages=page_frame)
        Path(temporary).replace(destination)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return str(destination)
