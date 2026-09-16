"""Read-only public entry point; private administration uses web.create_app."""
from pathlib import Path
from fastapi.responses import HTMLResponse, JSONResponse
from .web import create_app

PUBLIC_UI = '''
<style>
#start,#smoke,#retry,#cancel,#export,#saveReview,#epochPanel,#configuration,#dataMount,
.rail-nav a[href="#configuration"],.rail-nav a[href="#dataInput"]{display:none!important}
</style>
<script>
document.addEventListener('DOMContentLoaded',()=>{
 const banner=document.createElement('p');banner.textContent='只读访问 · 可查看结果与下载已有报告';
 banner.setAttribute('role','status');banner.style='padding:12px 20px;background:#eef4ee;color:#23452d';
 document.querySelector('main').prepend(banner);
 for(const id of ['reviewVerdict','reviewNote','reviewer']){const el=document.getElementById(id);if(el)el.disabled=true}
 const review=document.querySelector('#humanReview p');if(review)review.textContent='加载案例以查看已有评语。此入口不支持修改。';
 document.getElementById('report').onclick=null;
 const pdf=document.getElementById('pdf');pdf.textContent='↓ 下载已有 PDF';
 pdf.onclick=()=>{try{location.href=artifact(needRun(),'report/report.pdf')+'?v='+Date.now()}catch(e){notice(e.message)}};
});
</script>
'''

def create_public_app(state_root):
    app = create_app(state_root)
    # Keep API documentation consistent with the permissions of this entry point.
    app.routes[:] = [route for route in app.routes
                    if not getattr(route, 'methods', None)
                    or route.methods <= {'GET', 'HEAD', 'OPTIONS'}]
    app.openapi_schema = None

    @app.middleware('http')
    async def read_only(request, call_next):
        if request.method not in {'GET', 'HEAD', 'OPTIONS'}:
            return JSONResponse({'detail': 'Read-only access: writes are disabled'}, status_code=403)
        if request.method == 'GET' and request.url.path in {'/', '/static/index.html'}:
            html = (Path(__file__).parent/'static/index.html').read_text(encoding='utf-8')
            return HTMLResponse(html.replace('</head>', PUBLIC_UI+'</head>'),
                                headers={'Cache-Control': 'no-store', 'X-Read-Only': 'true'})
        response = await call_next(request)
        response.headers['X-Read-Only'] = 'true'
        if request.url.path.lower().endswith('.pdf'):
            response.headers['Cache-Control'] = 'no-store'
        return response

    return app
