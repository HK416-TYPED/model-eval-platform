"""Submit an import JSON specification; prompt securely for optional HF access."""
import argparse,getpass,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from eval_platform.core import read_json
from eval_platform.data_jobs import start_import

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('spec');parser.add_argument('--state',default='state')
    parser.add_argument('--ask-token',action='store_true');args=parser.parse_args()
    token=getpass.getpass('HF read token: ') if args.ask_token else None
    print(json.dumps(start_import(args.state,read_json(args.spec),token),ensure_ascii=False))
