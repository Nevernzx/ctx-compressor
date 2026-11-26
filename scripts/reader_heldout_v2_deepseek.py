"""Read the Task 14 compressor contexts and update its Qwen column with cascade disabled."""
from itertools import groupby
import os
from pathlib import Path
import sys

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
from scripts import reader_heldout_v2 as v2
from scripts import deepseek_api_inputs as ds
from scripts import deepseek_api_rollout as rollout
from scripts import deepseek_api_full_report as stats
from scripts import deepseek_api_subset as exp
from scripts import run_reader_heldout as reader
from src import reader_heldout_inputs as inp
from src.qa_metrics import score_answer
from src.reader import parse_answer

WORK=v2.WORK/'deepseek_compressor'
SMALL=exp.SMALL
FULL=exp.WORK/'full'
MANIFEST=SMALL/'TASK14_SHA256SUMS'
PROOF=WORK/'input_preflight/preflight.json'
SPOOL=PROOF.with_name('requests.jsonl')
CONFIG=SMALL/'deepseek_compressor_qwen_v2_run_config.json'
CMETRICS=SMALL/'deepseek_compressor_qwen_v2.json'
CMARKDOWN=SMALL/'deepseek_compressor_qwen_v2.md'
QMETRICS=SMALL/'qwen_column_v2.json'
QMARKDOWN=SMALL/'qwen_column_v2.md'
need=inp.need


def signed_index():
    index=inp.sums(PROJECT/'SHA256SUMS')
    tag=str(MANIFEST.relative_to(PROJECT))
    inp.verify(MANIFEST,index[tag])
    return inp.sums(MANIFEST),index[tag]


def signed(path, signatures):
    name=str(path.relative_to(PROJECT)) if path.is_relative_to(PROJECT) else str(path)
    need(name in signatures and path.resolve()==path,'unregistered or redirected Task14 input')
    inp.verify(path,signatures[name])
    return signatures[name]


def f_records():
    hashes,manifest_hash=signed_index()
    sources={str(MANIFEST):manifest_hash}
    for p in [FULL/'status.json',FULL/'predictions.jsonl',SMALL/'deepseek_api_results.json']:
        sources[str(p)]=signed(p,hashes)
    need(inp.read(FULL/'status.json')['status']=='complete','Task14 F incomplete')
    result=inp.read(SMALL/'deepseek_api_results.json')
    need(result['status']=='complete' and result['scope']['full_calls']==2588,'Task14 F coverage')
    preds=stats.unique(inp.rows(FULL/'predictions.jsonl'),ds.key)
    need(len(preds)==2040,'Task14 prediction count')
    return result,preds,hashes,sources


def c_batches(rows):
    seen=set()
    for task,group in groupby(rows,key=lambda r:r['task']):
        batch=list(group)
        expected=[(ds.B_POLICY,m,'mid',task,q) for m in ds.MODES for q in range(4)]
        need(task not in seen and [ds.key(r) for r in batch]==expected,'C task, mode or question order mismatch')
        seen.add(task)
        yield task,batch


def validate_question(req,archived):
    fields=('strategy','task','mode','tier','question_index','example_id','answer','answer_aliases',
            'question_hops','evidence_complete','trajectory_lost','prompt_sha1')
    need(all(req[k]==archived[k] for k in fields),'F/C reader prompt or label mismatch')
    need(req['non_gold_count']==archived['non_gold_count'],'non-gold count mismatch')


def prepare():
    need(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU preflight only')
    f,outputs,hashes,sources=f_records()
    corpus,tasks,heldout=inp.load_task_inputs(inp.sums(PROJECT/'SHA256SUMS'))
    tasks={tid:dict(task,task=tid,split='heldout') for tid,task in tasks.items()}
    s60,s30=ds.subsets(tasks)
    need(s60==f['scope']['s60'] and s30==f['scope']['s30'],'S30 membership/order mismatch')
    prepared=rollout.prepare_b(corpus,tasks,s30)
    paths=list((FULL/'trajectories').glob('*.json'))
    need(len(paths)==60,'F trajectory coverage')
    trajectories={}
    for path in paths:
        sources[str(path)]=signed(path,hashes)
        t=inp.read(path);key=t['task'],t['mode']
        need(key not in trajectories and path.name==f"{t['mode']}_{t['task']}.json",'F trajectory file identity')
        trajectories[key]=t
    need(set(trajectories)=={(task,m) for task in s30 for m in ds.MODES},'F trajectory matrix')
    # Replay strictly validates every intermediate ordered keep_after, plus the final PID list.
    recomputed=rollout.aggregate_b(prepared,list(trajectories.values()))
    need(recomputed['combined']==f['B'][ds.B_POLICY]['combined']['metrics'],'closed-loop replay differs from F')
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(str(reader.ROOT/'models/Qwen3-8B'),local_files_only=True,trust_remote_code=False)
    def requests():
        for task in s30:
            for mode in ds.MODES:
                pids=trajectories[task,mode]['final_pids']
                passages=inp.render_passages(corpus,pids)
                for req in ds.question_rows(ds.B_POLICY,mode,task,pids,tasks,corpus):
                    validate_question(req,outputs[ds.key(req)])
                    tokens,digest=inp.request_tokens(tokenizer,passages,req['question'],False)
                    need(digest==req['prompt_sha1'],'tokenized prompt hash differs from F')
                    inp.check_length(len(tokens),ds.key(req))
                    yield {**{k:v for k,v in req.items() if k!='messages'},'token_ids':tokens,'prompt_tokens':len(tokens)}
    need(WORK.resolve()==WORK,'redirected C directory')
    PROOF.parent.mkdir(parents=True,exist_ok=False)
    receipt=inp.export_requests(SPOOL,requests())
    groups=list(c_batches(inp.rows(SPOOL)))
    need(receipt['n']==240 and [t for t,g in groups]==s30,'C request validation failed')
    inp.write_new(PROOF,{'status':'passed','F_sources':sources,'heldout_inputs':heldout,'s30':s30,
                  'n_trajectories':len(trajectories),'n_replayed_steps':recomputed['combined']['n_steps'],
                  'n_prompt_sha1_checked':240,'prompt_sha1_mismatches':0,'requests':receipt})
    print('C preflight passed: 60 ordered final contexts, 548 replayed actions, 240 signed F prompt matches')


def check_pre():
    p=inp.read(PROOF)
    need(p['status']=='passed' and p['n_trajectories']==60 and p['n_replayed_steps']==548 and
         p['n_prompt_sha1_checked']==p['requests']['n']==240 and p['prompt_sha1_mismatches']==0 and
         p['requests']['path']==str(SPOOL),'C preflight incomplete')
    hashes,mhash=signed_index()
    need(p['F_sources'][str(MANIFEST)]==mhash,'signed F manifest changed')
    inp.verify(SPOOL,p['requests']['sha256'])
    return p


def detach():
    check_pre()
    reader.require_new([WORK/f for f in ('pipeline.pid','status.json','status.next','events.jsonl','launcher.log','ipc_check.json',
                                          'receipt_preflight.json','receipt_gate.json','receipt_inference.json','predictions.jsonl')])
    reader.require_new([CONFIG,CMETRICS,CMARKDOWN])
    pid=v2.detach_process(WORK,Path(__file__).resolve(),['pipeline'])
    print(f'C detached PID {pid} -> {WORK}',flush=True)


def pipeline():
    need(os.getpid()==os.getsid(0) and Path(sys.prefix).resolve()==reader.ROOT/'envs/reader' and
         os.environ.get('CUDA_VISIBLE_DEVICES')=='1' and os.environ.get('CUDA_DEVICE_ORDER')=='PCI_BUS_ID',
         'detached GPU1 session required')
    with (WORK/'pipeline.pid').open('x') as f: print(os.getpid(),file=f)
    try:
        reader.publish(WORK,dict(stage='preflight',status='running',automatic_retry=False))
        p=check_pre()
        reader.receipts_write(WORK,'preflight',sha256=reader.digest(PROOF),n=p['requests']['n'])
        gpu=reader.wait_for_gpu('deepseek_compressor_qwen_v2',lambda row:reader.publish(WORK,row))
        reader.receipts_write(WORK,'gate',gpu=gpu)
        cfg=v2.run_engine(WORK,CONFIG,c_batches(inp.rows(SPOOL)),30,240,smoke=False,
                          provenance={'preflight_path':str(PROOF),'preflight_sha256':reader.digest(PROOF),'batch_size':8})
        reader.publish(WORK,dict(status='complete',stage='finished',predictions=cfg['predictions'],automatic_retry=False))
    except BaseException as exc:
        reader.publish(WORK,dict(status='failed',error=repr(exc),automatic_retry=False))
        raise


if __name__=='__main__' and sys.argv[1:] in (['prepare'],['detach'],['pipeline']):
    need(Path.cwd()==PROJECT,'run from 5880 project root')
    {'prepare':prepare,'detach':detach,'pipeline':pipeline}[sys.argv[1]]()
    raise SystemExit(0)


def aligned_predictions(candidates,reference):
    """Return only registered Qwen identities, with signed DeepSeek scores on the same rows."""
    matched={}
    for row in candidates:
        k=ds.key(row)
        if k in reference:
            need(k not in matched,'duplicate matched Qwen prediction')
            archived=reference[k]
            merged=ds.compare_qwen(archived,row)
            if 'qwen_prompt_tokens' in archived:
                need(archived['qwen_prompt_tokens']==row['prompt_tokens'],'Qwen prompt token count changed')
            matched[k]=merged
    need(matched.keys()==reference.keys(),'incomplete Qwen/DeepSeek intersection')
    return [matched[k] for k in reference]


def read_inference(directory,config_path,n):
    cfg=inp.read(config_path)
    need(inp.read(directory/'status.json')['status']=='complete' and
         cfg['status']=='complete' and cfg['engine_shutdown'] and cfg['predictions']==n and
         cfg['effective_disable_cascade_attn'] is True and cfg['cuda_visible_devices']=='1',
         'cascade-off inference incomplete')
    receipt=inp.read(directory/'receipt_inference.json')
    digest=inp.checksum(directory/'predictions.jsonl')
    need(receipt['predictions_sha256']==cfg['predictions_sha256']==digest and receipt['n']==n and
         receipt['run_config_sha256']==inp.checksum(config_path),'Qwen inference receipt mismatch')
    return {'predictions_sha256':digest,'run_config_sha256':receipt['run_config_sha256'],
            'effective_disable_cascade_attn':True,'effective_value_source':cfg['effective_disable_cascade_attn_source'],
            'path':str(directory/'predictions.jsonl'),'n':n}


def metric_line(score):
    return f"{stats.pct(score['f1'])} / {stats.pct(score['em'])}（n={score['n']}）"


def c_report():
    reader.require_new([CMETRICS,CMARKDOWN])
    check_pre()
    f,preds,_,sources=f_records()
    run=read_inference(WORK,CONFIG,240)
    archived={k:v for k,v in preds.items() if k[0]==ds.B_POLICY}
    need(len(archived)==240,'F B reader coverage')
    aligned=aligned_predictions(inp.rows(WORK/'predictions.jsonl'),archived)
    result={'status':'complete','scope':'任务14 C','input_preflight':str(PROOF),
            'input_preflight_sha256':inp.checksum(PROOF),'inference':run,'F_sources':sources,'scores':{}}
    rows=[]
    for scope in ('combined',*ds.MODES):
        group=[r for r in aligned if scope=='combined' or r['mode']==scope]
        need(len(group)==(240 if scope=='combined' else 120),'wrong C coverage')
        ds_score,qw=[stats.scores(group,x) for x in stats.READERS]
        need(ds_score==f['B'][ds.B_POLICY][scope]['DeepSeek_reader'],'reference F score differs')
        pair=stats.paired([(r,r) for r in group],*stats.READERS)
        result['scores'][scope]={'DeepSeek':ds_score,'Qwen3-8B':qw,'paired_deepseek_minus_qwen':pair}
        rows.append([scope,metric_line(ds_score),metric_line(qw),stats.f1_counts(pair)])
    text='\n\n'.join(['# 任务14 C　DeepSeek 压缩上下文的 Qwen3-8B 读者评测',
          '逐步重放 n=60 条轨迹、n=548 个动作，复原最终有序段落；读者提示逐条命中 F 阶段归档哈希（240/240）。每个任务将 reselect 与 append 各四题顺序拼为一批（n=30 批）。引擎核验 disable_cascade_attn=True，生成配置同任务16 主表。DeepSeek 列取自任务14 已归档的相同提示。',
          stats.table(['范围','DeepSeek F1 / EM','Qwen F1 / EM','逐题 F1 DeepSeek 高/平/Qwen 高'],rows)])+'\n'
    inp.write_new(CMETRICS,result)
    with CMARKDOWN.open('x') as fobj:fobj.write(text)
    print('C',result['scores']['combined'])


def qwen_column():
    reader.require_new([QMETRICS,QMARKDOWN])
    original,predictions,hashes,sources=f_records()
    run=read_inference(v2.WORK,v2.SMALL/'run_config.json',55776)
    call_path=FULL/'api_calls.jsonl'
    sources[str(call_path)]=signed(call_path,hashes)
    calls=stats.unique(inp.rows(call_path),lambda r:r['id'])
    archive={k:v for k,v in predictions.items() if k[0] in stats.STRATEGIES}
    sanity={(*config,i) for config in ds.configurations(original['scope']['s60'],original['scope']['s30'],'full') for i in range(4)}
    need(len(archive)==1800 and set(archive)==sanity,'unregistered Task14 A keys')
    old=stats.reader_results(list(archive.values()),original['scope']['s30'],calls)
    need(old==original['A'],'replay of signed Task14 A differs')
    corrected=aligned_predictions(inp.rows(v2.WORK/'predictions.jsonl'),archive)
    new=stats.reader_results(corrected,original['scope']['s30'],calls)
    pairedrows=[]
    for key,val in old['strategies'].items():
        src=val['scores'];dest=new['strategies'][key]
        need(src['DeepSeek']==dest['scores']['DeepSeek'],'DeepSeek scores changed while joining Qwen corrections')
        pairedrows.append([key,metric_line(src['DeepSeek']),metric_line(src['Qwen3-8B']),metric_line(dest['scores']['Qwen3-8B']),
                           stats.f1_counts(dest['paired_readers'])])
    payload={'status':'complete','input':'Task14 F A, same 1,800 prompts','matched_prompt_sha1':1800,'mismatches':0,
             'F_sources':sources,'Qwen_main':run,'historical':old,'cascade_disabled':new,'no_new_api_calls':True}
    markdown=['# 任务14 A　关 cascade 后的 Qwen3-8B 列',
              '逐条按策略、模式、预算、任务和题号关联任务16 的 Qwen 输出，提示哈希与 F 阶段逐条相同（1,800/1,800）。DeepSeek 复用原始回答。新旧 Qwen 均在相同题目与相同评分协议上计算。',
              stats.table(['策略','DeepSeek F1 / EM','历史 Qwen F1 / EM','当前 Qwen F1 / EM','F1 DeepSeek 高/平/当前 Qwen 高'],pairedrows),
              '## 预注册主比较']
    ctr=[]
    for tag,label in [('r2_minus_v3','r2 − v3'),('oracle_minus_no_compress_s30','oracle − 不压缩')]:
        for scope in ('combined',*ds.MODES):
            rows=new[tag][scope];models=[('DeepSeek',rows['DeepSeek']),('历史 Qwen',old[tag][scope]['Qwen3-8B']),('当前 Qwen',rows['Qwen3-8B'])]
            for model,m in models:
                ctr.append([label,scope,model,m['n'],rows['n_unique_questions'],stats.diff(m['f1_difference_pp']),stats.diff(m['em_difference_pp']),stats.f1_counts(m),
                            ' / '.join(str(t) for pair in m['em_matrix_correct_then_incorrect'] for t in pair)])
    markdown.append(stats.table(['比较','范围','读者','题次 n','不同题数','ΔF1（百分点）','ΔEM（百分点）','F1 左高/平/右高','EM 左右都对/仅左对/仅右对/均错'],ctr))
    evidence=[]
    for name,entry in new['strategies'].items():
        for group,values in entry['by_evidence_complete'].items():
            prior=old['strategies'][name]['by_evidence_complete'][group]
            evidence.append([name,'齐全' if group=='true' else '有缺失',metric_line(values['DeepSeek']),metric_line(prior['Qwen3-8B']),metric_line(values['Qwen3-8B'])])
    markdown.extend(['## 证据分层',stats.table(['策略','该题证据','DeepSeek F1 / EM','历史 Qwen F1 / EM','当前 Qwen F1 / EM'],evidence)])
    bins=[]
    for k,v in new['r2_minus_v3_non_gold_conditioned'].items():
        bins.append([k,v['Qwen3-8B']['n'],stats.diff(v['DeepSeek']['f1_difference_pp']),stats.diff(old['r2_minus_v3_non_gold_conditioned'][k]['Qwen3-8B']['f1_difference_pp']),stats.diff(v['Qwen3-8B']['f1_difference_pp'])])
    markdown.append(stats.table(['r2 多留非金段数（双方证据齐全）','配对 n','DeepSeek ΔF1','历史 Qwen ΔF1','当前 Qwen ΔF1'],bins))
    inp.write_new(QMETRICS,payload)
    with QMARKDOWN.open('x') as out:out.write('\n\n'.join(markdown)+'\n')
    print('A', {tag:new[tag]['combined']['Qwen3-8B']['f1_difference_pp'] for tag in ('r2_minus_v3','oracle_minus_no_compress_s30')})


if __name__=='__main__':
    need(Path.cwd()==PROJECT and os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU reports on 5880 only')
    if sys.argv[1:]==['report-c']:c_report()
    elif sys.argv[1:]==['qwen-column']:qwen_column()
    else:raise SystemExit('expected prepare, detach, internal pipeline, report-c, or qwen-column')

