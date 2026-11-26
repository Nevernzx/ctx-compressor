"""Run the preregistered DeepSeek full subset in the admitted detached process."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from decimal import Decimal
import json
import os
from pathlib import Path
import sys
import threading

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from scripts import deepseek_api_subset as subset
from scripts.deepseek_api_client import APIStop, DeepSeekClient, peak_end
from scripts.deepseek_api_rollout import aggregate_b, run_b_task
from src.qa_metrics import score_answer
from src.reader import parse_answer

WORK = subset.WORK
SMALL = subset.SMALL
WORKERS = 8


def require(condition, description):
    if not condition:
        raise ValueError(description)


def claim_full(directory):
    """The launcher created only these three files before exec, all under one PID."""
    require(directory.resolve() == directory and directory.is_dir(), 'invalid stage directory')
    owned = {'full.pid', 'events.jsonl', 'status.json'}
    for file in directory.iterdir():
        if file.name not in owned:
            raise FileExistsError('full stage already has other artifacts')
        require(file.is_file() and not file.is_symlink(), 'invalid launcher artifact')
    status = subset.read(directory / 'status.json')
    pid = int((directory / 'full.pid').read_text().strip())
    require(status['stage'] == 'F' and status['pid'] == pid == os.getpid(), 'detached PID mismatch')
    require(status['status'] in ('running', 'paused_peak'), 'stage already stopped')
    with (directory / 'events.jsonl').open() as f:
        last = None
        for row in f:
            last = json.loads(row)
    require(last == status, 'stage status and last event differ')


def expected_calls(reader_rows, compressor):
    subset.validate_order(reader_rows)
    reader_keys = []
    for row in reader_rows:
        ident = subset.key(row['strategy'], row['mode'], row['tier'], row['task'], row['question_index'])
        reader_keys.append(subset.reader_kind(row) + '/' + ident)
    step_keys = []
    b_reader_keys = []
    for traj in compressor['rows']:
        mode, task, n_steps = traj['mode'], traj['task'], traj['n_steps']
        for step in range(n_steps):
            step_keys.append(subset.key('B_compressor', mode, 'mid', task, step))
        for qi in range(4):
            b_reader_keys.append('B_reader/' + subset.key('deepseek-flash', mode, 'mid', task, qi))
    all_keys = set(reader_keys + step_keys + b_reader_keys)
    require(len(reader_keys) + len(step_keys) + len(b_reader_keys) == len(all_keys), 'duplicate call in plan')
    return all_keys


def run_parallel(jobs, run_job, stop):
    """Each future executes a whole trajectory; only that future issues its sequential calls."""
    failures = []
    lock = threading.Lock()
    def guarded(job):
        if stop.is_set():
            return
        try:
            run_job(job)
        except Exception as exc:
            with lock:
                if not failures:
                    failures.append(exc)
                stop.set()
            raise
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        pending = [pool.submit(guarded, job) for job in jobs]
        try:
            for future in as_completed(pending):
                future.result()
        except Exception:
            for future in pending:
                future.cancel()
    if failures:
        raise failures[0]


class FullStage:
    def __init__(self, directory, client, data, b, calls, predictions):
        self.directory, self.client = directory, client
        self.data, self.b = data, b
        self.calls, self.predictions = calls, predictions
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.by_kind = Counter()
        self.sent = set()
        self.predicted = set()
        self.paid = Decimal(0)
        self.trajectories = []
        self.stopped = None

    def notify(self, event):
        with self.lock:
            status = event['status']
            if status.startswith('stopped_'):
                self.stopped = self.stopped or status
                self.stop.set()
            if self.stopped and status != self.stopped:
                return
            subset.update_status(self.directory, {'stage':'F', 'pid':os.getpid(), 'paid_calls_full':sum(self.by_kind.values()), **event})

    def query(self, kind, ident, messages, limit, qwen_tokens=0):
        with self.lock:
            if self.stop.is_set():
                raise APIStop(self.stopped or 'stopped_worker')
            require(ident not in self.sent, 'duplicate logical call')
            self.sent.add(ident)
        response = self.client.complete(ident, messages, max_tokens=limit)
        require(response['call_id'] == ident and peak_end(datetime.fromisoformat(response['started_utc'])) is None,
                'receipt identity or off-peak time mismatch')
        event = {**response, 'kind':kind, 'qwen_tokens':qwen_tokens}
        with self.lock:
            subset.write_row(self.calls, event)
            self.paid += Decimal(response['cost_yuan'])
            self.by_kind[kind] += 1
            n = sum(self.by_kind.values())
            if n % 25 == 0 and not self.stop.is_set() and peak_end(self.client.clock()) is None:
                subset.update_status(self.directory, {'stage':'F', 'pid':os.getpid(), 'status':'running',
                                                     'paid_calls_full':n, 'full_spent_yuan':str(self.paid)})
        return response

    def answers(self, rows, from_b=False):
        subset.validate_order(rows)
        for row in rows:
            kind = 'B_reader' if from_b else subset.reader_kind(row)
            ident = subset.key(row['strategy'],row['mode'],row['tier'],row['task'],row['question_index'])
            response = self.query(kind, kind + '/' + ident, row['messages'], 48, row.get('qwen_prompt_tokens', 0))
            pred = parse_answer(response['content'])
            scores = score_answer(pred, row['answer'], row['answer_aliases'])
            public = {k:v for k,v in row.items() if k not in ('messages','token_ids','final_pids')}
            with self.lock:
                require(ident not in self.predicted, 'duplicate scored prediction')
                subset.write_row(self.predictions, {**public, 'request_id':response['id'], 'prediction':pred,
                                 'raw_output':response['content'], 'finish_reason':response['finish_reason'], **scores})
                self.predicted.add(ident)

    def job(self, item):
        section, value = item
        if section == 'A':
            self.answers(value)
            return
        task, mode = value
        def generate(messages, task_id, mode, step):
            return self.query('B_compressor', subset.key('B_compressor', mode, 'mid', task_id, step), messages, 256)
        trajectory = run_b_task(self.b, task, mode, generate)
        subset.write_new(self.directory / 'trajectories' / f'{mode}_{task}.json', trajectory)
        self.answers(self.data.reader_rows(task,mode,trajectory['final_pids']), from_b=True)
        with self.lock:
            self.trajectories.append(trajectory)


def execute(directory):
    claim_full(directory)
    stage = None
    client = None
    try:
        require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and Path.cwd() == PROJECT, 'wrong CPU execution environment')
        pre = subset.read(SMALL / 'input_checks.json')
        suite = subset.read(SMALL / 'cpu_tests.json')
        smoke = subset.read(WORK / 'smoke/smoke_results.json')
        require(smoke['status'] == 'complete' and smoke['n_calls'] == 36 and suite['status'] == 'passed'
                and suite['previous_test_ids_matched'] == 729 and not suite['cpu_guard_events'], 'incomplete P/S receipts')
        # Validate every identity before the first paid request.
        data, b, rows, sample = subset.prepare_both()
        require(json.loads(subset.encoded(subset.records(data,b,rows,sample))) == pre, 'P/F input receipts differ')
        wanted = expected_calls(rows, pre['B'])
        require(len(wanted) == 2588 and pre['B']['n_steps'] == 548, 'unregistered F workload')
        (directory / 'trajectories').mkdir(exist_ok=False)
        client = DeepSeekClient(work=WORK, stage='full')
        initial = client.summary()
        require(initial['stage_totals']['attempts'] == 0 and initial['calls'] == smoke['n_calls']
                and Decimal(initial['spent_yuan']) == Decimal(smoke['smoke_spent_yuan'])
                and Decimal(initial['reserved_yuan']) == 0, 'cumulative smoke ledger differs or full stage already started')
        with (directory/'api_calls.jsonl').open('x') as raw, (directory/'predictions.jsonl').open('x') as preds:
            stage = FullStage(directory, client, data, b, raw, preds)
            client.publish = stage.notify
            balance_start = client.balance('start')
            stage.notify({'status':'running','registered_calls':len(wanted),'smoke_spent_yuan':initial['spent_yuan']})
            jobs = [('A',rows[i:i+4]) for i in range(0,len(rows),4)]
            jobs.extend(('B',(task,mode)) for task in data.s30 for mode in subset.MODES)
            run_parallel(jobs, stage.job, stage.stop)
        ledger = client.summary()
        stage_totals = ledger['stage_totals']
        require(stage.sent == wanted and sum(stage.by_kind.values()) == stage_totals['calls'] == 2588,
                'full logical/API coverage differs')
        require(stage.by_kind['B_compressor'] == 548 and stage.by_kind['B_reader'] == 240
                and len(stage.trajectories) == 60 and len(stage.predicted) == 2040, 'full prediction coverage differs')
        require(stage.paid == Decimal(stage_totals['spent_yuan']) and stage.paid + Decimal(initial['spent_yuan']) == Decimal(ledger['spent_yuan'])
                and Decimal(ledger['spent_yuan']) + Decimal(ledger['reserved_yuan']) <= 8, 'full ledger reconciliation failed')
        metrics = aggregate_b(b, sorted(stage.trajectories, key=lambda traj:(traj['task'],traj['mode'])))
        balance_end = client.balance('end')
        spent = Decimal(balance_start['total_balance']) - Decimal(balance_end['total_balance'])
        status = 'complete' if client.balance_ok else 'stopped_balance'
        subset.write_new(directory / 'full_results.json', {'status':status, 'n_calls':len(wanted),
                        'by_kind':dict(stage.by_kind), 'A_n':len(rows),'B_steps':548,'B_reader_n':240,'B_trajectories':60,
                        'B_metrics':metrics,'full_spent_yuan':str(stage.paid),'smoke_spent_yuan':initial['spent_yuan'],
                        'balance_before':balance_start,'balance_after':balance_end, 'account_drop_yuan':str(spent),
                        'account_drop_minus_ledger_yuan':str(spent-stage.paid),'ledger':client.summary()})
        stage.notify({'status':status,'n_calls':len(wanted),'cumulative_spent_yuan':ledger['spent_yuan']})
        return status == 'complete'
    except Exception as exc:
        # Only exception categories reach logs. Provider strings and authentication stay in process memory.
        status = exc.status if isinstance(exc, APIStop) else 'failed'
        if stage and stage.stopped:
            status = stage.stopped
        if client and not (directory / 'balance_end.json').exists():
            try:
                client.balance('end')
            except Exception:
                pass
        event = {'stage':'F','pid':os.getpid(),'status':status,'error_type':type(exc).__name__}
        if stage:
            event['paid_calls_full'] = sum(stage.by_kind.values())
        subset.update_status(directory,event)
        return False


if __name__ == '__main__':
    try:
        ok = execute(WORK / 'full')
    except Exception as exc:
        print(type(exc).__name__, file=sys.stderr)
        ok = False
    sys.exit(0 if ok else 1)

