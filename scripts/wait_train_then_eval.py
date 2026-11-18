#!/usr/bin/env python3
"""Detached Task 05 continuation: wait for the exact training PID, then evaluate.

No GPU initialization here. The model runner performs its own free-GPU checks;
no retry, signal, process-name matching, or hyperparameter changes are used.
"""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time

PROJECT=Path('/root/workplace/ctx-compressor')
ROOT=Path('/root/userfolder/ctx-compressor')
RUN=ROOT/'runs/sft_20k_20261117'
OUT=PROJECT/'eval/sft_run'


def save(value):
    path=RUN/'continuation_status.json'
    temp=path.with_suffix('.tmp')
    value['utc']=datetime.now(timezone.utc).isoformat()
    temp.write_text(json.dumps(value,indent=2)+'\n'); temp.replace(path)


def main():
    launch=json.loads((RUN/'pipeline_launch.json').read_text())
    pid=launch['pid']
    if (RUN/'continuation_started.json').exists():
        prior=json.loads((RUN/'continuation_started.json').read_text())
        prior_stat=Path(f"/proc/{prior['pid']}/stat")
        if prior_stat.exists() and prior_stat.read_text().rsplit(')',1)[1].split()[0]!='Z':
            raise RuntimeError('continuation already running')
    (RUN/'continuation_started.json').write_text(json.dumps({'pid':os.getpid(),'session_id':os.getsid(0),
                                                            'training_pid':pid},indent=2)+'\n')
    def training_alive():
        path=Path(f'/proc/{pid}/stat')
        if not path.exists(): return False
        # A detached process may briefly remain a zombie awaiting PID 1 reaping.
        return path.read_text().rsplit(')',1)[1].split()[0]!='Z'
    while training_alive():
        save({'status':'waiting_training_exit','training_pid':pid,'watcher_pid':os.getpid()})
        time.sleep(30)
    result_path=RUN/'worker_result.json'
    if not result_path.exists():
        save({'status':'stopped','reason':'training pipeline exited without a formal training result','training_pid':pid})
        return 1
    result=json.loads(result_path.read_text())
    if result.get('status')!='ok' or result.get('optimizer_steps')!=2500 or result.get('consumed_samples')!=40000:
        save({'status':'stopped','reason':'formal training did not meet its completion contract','training_pid':pid})
        return 1
    if (OUT/'training_stop.json').exists():
        save({'status':'stopped','reason':'explicit training failure marker exists','training_pid':pid})
        return 1
    with (RUN/'heldout_model_eval.log').open('xb') as stream:
        child=subprocess.Popen(['nohup','bash',str(PROJECT/'scripts/run_heldout_model.sh')],
                               cwd=str(PROJECT),stdin=subprocess.DEVNULL,stdout=stream,
                               stderr=subprocess.STDOUT,start_new_session=True)
        record={'pid':child.pid,'session_id':child.pid,'training_pid':pid,'watcher_pid':os.getpid(),
                'log':str(RUN/'heldout_model_eval.log'),'nohup':True,'detached':True}
        (RUN/'evaluation_launch.json').write_text(json.dumps(record,indent=2)+'\n')
        (OUT/'evaluation_launch.json').write_text(json.dumps(record,indent=2)+'\n')
        save({'status':'evaluating',**record})
        code=child.wait()
    save({'status':'evaluation_complete' if code==0 else 'evaluation_failed','exit_code':code,**record})
    return code


if __name__=='__main__': raise SystemExit(main())
