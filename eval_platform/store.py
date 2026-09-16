import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from .core import canonical, digest, safe_id

class Store:
    def __init__(self, root):
        self.root = Path(root).resolve(); self.root.mkdir(parents=True, exist_ok=True)
        self.db = self.root/'jobs.sqlite3'
        with self.connect() as c:
            c.executescript('''
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS runs (
              id TEXT PRIMARY KEY, spec TEXT NOT NULL, created REAL NOT NULL,
              cancel INTEGER NOT NULL DEFAULT 0);
            CREATE TABLE IF NOT EXISTS jobs (
              id TEXT PRIMARY KEY, run_id TEXT NOT NULL, model_id TEXT NOT NULL,
              case_id TEXT NOT NULL, task TEXT NOT NULL, seed INTEGER NOT NULL,
              state TEXT NOT NULL, request TEXT NOT NULL, result TEXT,
              error TEXT, attempts INTEGER NOT NULL DEFAULT 0, updated REAL NOT NULL);
            CREATE INDEX IF NOT EXISTS jobs_run ON jobs(run_id,state);
            CREATE TABLE IF NOT EXISTS scores (
              id TEXT PRIMARY KEY, run_id TEXT NOT NULL, job_id TEXT NOT NULL,
              scorer TEXT NOT NULL, result TEXT NOT NULL, updated REAL NOT NULL);
            ''')

    @contextmanager
    def connect(self):
        c=sqlite3.connect(self.db, timeout=30)
        c.row_factory=sqlite3.Row
        try:
            yield c
            c.commit()
        except BaseException:
            c.rollback();raise
        finally:c.close()

    def create(self, run_id, spec, jobs):
        safe_id(run_id)
        with self.connect() as c:
            c.execute('INSERT INTO runs(id,spec,created) VALUES(?,?,?)',(run_id,canonical(spec),time.time()))
            c.executemany('INSERT INTO jobs(id,run_id,model_id,case_id,task,seed,state,request,updated) VALUES(?,?,?,?,?,?,?,?,?)',
                          [(j['id'],run_id,j['model_id'],j['case_id'],j['task'],j['seed'],'queued',canonical(j['request']),time.time()) for j in jobs])

    def run(self, run_id):
        with self.connect() as c:r=c.execute('SELECT * FROM runs WHERE id=?',(run_id,)).fetchone()
        if not r:raise KeyError('Unknown run: '+run_id)
        v=dict(r);v['spec']=json.loads(v['spec']);return v

    def jobs(self, run_id):
        with self.connect() as c:rows=c.execute('SELECT * FROM jobs WHERE run_id=? ORDER BY model_id,task,case_id,seed',(run_id,)).fetchall()
        values=[]
        for r in rows:
            v=dict(r);v['request']=json.loads(v['request']);v['result']=json.loads(v['result']) if v['result'] else None;values.append(v)
        return values

    def summary(self, run_id):
        run=self.run(run_id);counts={s:0 for s in ['queued','running','succeeded','failed','cancelled']}
        with self.connect() as c:
            for r in c.execute('SELECT state,COUNT(*) n FROM jobs WHERE run_id=? GROUP BY state',(run_id,)):counts[r['state']]=r['n']
        if counts['running']:state='running'
        elif counts['queued']:state='queued'
        elif counts['cancelled']:state='cancelled'
        elif counts['failed']:state='completed_with_errors' if counts['succeeded'] else 'failed'
        else:state='completed' if counts['succeeded'] else 'unavailable'
        return {'id':run_id,'created':run['created'],'state':state,'counts':counts,'planned':sum(counts.values()),
                'unavailable':run['spec'].get('unavailable',[]),'cancel_requested':bool(run['cancel'])}

    def list_runs(self):
        with self.connect() as c:ids=[r[0] for r in c.execute('SELECT id FROM runs ORDER BY created DESC')]
        return [self.summary(i) for i in ids]

    def update_job(self, job_id, state, result=None, error=None):
        with self.connect() as c:
            c.execute('UPDATE jobs SET state=?,result=?,error=?,updated=?, attempts=attempts+? WHERE id=?',
                      (state,canonical(result) if result else None,error,time.time(),int(state=='running'),job_id))

    def cancel(self, run_id):
        self.run(run_id)
        with self.connect() as c:
            c.execute('UPDATE runs SET cancel=1 WHERE id=?',(run_id,))
            c.execute("UPDATE jobs SET state='cancelled',updated=? WHERE run_id=? AND state='queued'",(time.time(),run_id))

    def clear_cancel(self, run_id):
        with self.connect() as c:c.execute('UPDATE runs SET cancel=0 WHERE id=?',(run_id,))

    def cancelled(self, run_id):return bool(self.run(run_id)['cancel'])
