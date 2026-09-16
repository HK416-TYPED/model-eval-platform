import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from .core import read_json
from .store import Store

def spawn_worker(state_root,run_id,limit=None,retry=False,device='cuda:0'):
    store=Store(state_root);store.run(run_id)
    args=[sys.executable,'-m','eval_platform.cli','--state',str(store.root),'worker',run_id,'--device',device]
    if limit is not None:args+=['--limit',str(limit)]
    if retry:args+=['--retry']
    log=store.root/'runs'/run_id/'worker.log'
    options={'start_new_session':True} if os.name!='nt' else {'creationflags':subprocess.CREATE_NEW_PROCESS_GROUP|subprocess.DETACHED_PROCESS|subprocess.CREATE_NO_WINDOW}
    with open(log,'ab') as out:
        proc=subprocess.Popen(args,stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT,
                              cwd=str(Path(__file__).resolve().parent.parent),**options)
    return {'pid':proc.pid,'log':str(log)}

def main():
    p=argparse.ArgumentParser(description='固定案例 × seed × checkpoint 的自动效果测试')
    p.add_argument('--state',default=os.environ.get('EVAL_STATE','state'))
    sub=p.add_subparsers(dest='command',required=True)
    s=sub.add_parser('freeze');s.add_argument('bindings');s.add_argument('name')
    for name in ['import-hf','import-wds']:
        s=sub.add_parser(name);s.add_argument('binding');s.add_argument('destination')
    s=sub.add_parser('preflight');s.add_argument('config')
    s=sub.add_parser('prepare-v7');s.add_argument('source');s.add_argument('destination');s.add_argument('runtime_dir')
    s=sub.add_parser('create');s.add_argument('config');s.add_argument('run_id')
    s=sub.add_parser('append');s.add_argument('run_id');s.add_argument('model')
    for name in ['worker','start']:
        s=sub.add_parser(name);s.add_argument('run_id');s.add_argument('--limit',type=int);s.add_argument('--retry',action='store_true');s.add_argument('--device',default='cuda:0');s.add_argument('--model')
    s=sub.add_parser('status');s.add_argument('run_id',nargs='?')
    s=sub.add_parser('cancel');s.add_argument('run_id')
    s=sub.add_parser('report');s.add_argument('run_id');s.add_argument('--export',action='store_true');s.add_argument('--pdf',action='store_true')
    s=sub.add_parser('score');s.add_argument('run_id');s.add_argument('scorer')
    s=sub.add_parser('serve');s.add_argument('--port',type=int,default=8765)
    s.add_argument('--host',default='127.0.0.1');s.add_argument('--read-only',action='store_true')
    args=p.parse_args()
    if args.command=='serve' and args.host not in {'127.0.0.1','localhost','::1'} and not args.read_only:
        p.error('Non-loopback serving requires --read-only; use an SSH tunnel for administration')
    store=Store(args.state)
    if args.command in ('import-hf','import-wds'):
        from .datasets import import_hf_jsonl,import_webdataset
        value=(import_hf_jsonl if args.command=='import-hf' else import_webdataset)(read_json(args.binding),args.destination)
    elif args.command=='freeze':
        from .suites import freeze_suite
        config=read_json(args.bindings)
        value=freeze_suite(store.root/'suites'/args.name,config['bindings'],config.get('selection_seed',20260915),config.get('seeds'),config.get('count',5))
        value={'suite':args.name,'digest':value['digest'],'cases':len(value['cases'])}
    elif args.command=='prepare-v7':
        from .checkpoint import prepare_v7
        value=prepare_v7(args.source,args.destination,args.runtime_dir)
    elif args.command=='preflight':
        from .experiment import preflight
        value=preflight(read_json(args.config))
        # Digest tables live in lock files, not terminal output.
        value['models']=[{'id':m['id'],'tasks':m['tasks'],'digest':m['identity']['digest']} for m in value['models']]
    elif args.command=='append':
        from .comparison import append_checkpoint
        from .worker import GpuLock
        with GpuLock(store.root,'cuda:0'):value=append_checkpoint(store,args.run_id,read_json(args.model))
    elif args.command=='create':
        from .experiment import create_run
        value=create_run(store,args.run_id,read_json(args.config))
    elif args.command=='worker':
        from .worker import run_worker
        value=run_worker(store.root,args.run_id,args.device,args.limit,args.retry,args.model)
    elif args.command=='start':value=spawn_worker(store.root,args.run_id,args.limit,args.retry,args.device)
    elif args.command=='status':value=store.summary(args.run_id) if args.run_id else store.list_runs()
    elif args.command=='cancel':store.cancel(args.run_id);value=store.summary(args.run_id)
    elif args.command=='report':
        from .report import build_report,export_zip,export_pdf
        value={'report':export_zip(store,args.run_id) if args.export else export_pdf(store,args.run_id) if args.pdf else build_report(store,args.run_id,boards=True)}
    elif args.command=='score':
        from .scoring import score_run
        from .worker import GpuLock
        with GpuLock(store.root,'cuda:0'):value=score_run(store,args.run_id,read_json(args.scorer))
    else:
        import uvicorn
        if args.read_only:
            from .public_readonly import create_public_app as create_app
        else:
            from .web import create_app
        uvicorn.run(create_app(store.root),host=args.host,port=args.port);return
    print(json.dumps(value,ensure_ascii=False,indent=2))
    if args.command=='preflight' and not value['ready']:raise SystemExit(2)

if __name__=='__main__':main()
