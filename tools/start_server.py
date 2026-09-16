"""Start the loopback server; restart only the matching evaluation server."""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.request

root = Path(__file__).resolve().parent.parent
parser = argparse.ArgumentParser()
parser.add_argument('--port', type=int, default=8765)
parser.add_argument('--restart', action='store_true')
args = parser.parse_args()
configs = root/'state/configs'
configs.mkdir(parents=True, exist_ok=True)
for name in ['autodl-anima-v7.json']:
    target = configs/name
    if not target.exists() and (root/'configs'/name).is_file():
        shutil.copyfile(root/'configs'/name, target)

if args.restart:
    if not Path('/proc').is_dir():
        raise SystemExit('--restart currently requires Linux /proc')
    for process in Path('/proc').iterdir():
        if not process.name.isdigit():
            continue
        try:
            command = (process/'cmdline').read_bytes().decode().strip('\0').split('\0')
            if 'eval_platform.cli' not in command or 'serve' not in command:
                continue
            state = command[command.index('--state')+1] if '--state' in command else 'state'
            port = int(command[command.index('--port')+1]) if '--port' in command else 8765
            cwd = (process/'cwd').resolve()
            if (cwd/state).resolve() == (root/'state').resolve() and port == args.port:
                os.kill(int(process.name), signal.SIGTERM)
        except (OSError, ValueError, IndexError):
            continue
    for attempt in range(50):
        with socket.socket() as sock:
            if sock.connect_ex(('127.0.0.1', args.port)) != 0:
                break
        time.sleep(.1)
    else:
        raise SystemExit('Existing service did not stop; no replacement launched')

try:
    with urllib.request.urlopen(f'http://127.0.0.1:{args.port}/api/capabilities', timeout=3) as response:
        data = json.load(response)
    if 'anima_v7' in data:
        print('Evaluation server already running')
        raise SystemExit()
except OSError:
    pass
with socket.socket() as sock:
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(('127.0.0.1', args.port))
with open(root/'state/server.log', 'ab') as log:
    process = subprocess.Popen([sys.executable, '-m', 'eval_platform.cli', '--state', str(root/'state'),
                                'serve', '--port', str(args.port)], cwd=root,
                               stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)
(root/'state'/f'server-{args.port}.pid').write_text(str(process.pid))
for attempt in range(50):
    try:
        with urllib.request.urlopen(f'http://127.0.0.1:{args.port}/api/capabilities', timeout=1):
            print(json.dumps({'pid': process.pid, 'url': f'http://127.0.0.1:{args.port}'}))
            break
    except OSError:
        if process.poll() is not None:
            raise SystemExit('Server exited; inspect state/server.log')
        time.sleep(.1)
else:
    raise SystemExit('Server did not become ready; inspect state/server.log')
