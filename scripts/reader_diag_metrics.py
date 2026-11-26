"""Score the registered task-15 arms from their saved predictions, without an engine."""
from array import array
from collections import Counter, defaultdict
from itertools import chain
import hashlib
import json
import math
from pathlib import Path
import sys

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
from src.reader import parse_answer
from src.qa_metrics import score_answer
from src.reader_heldout_inputs import checksum, sums

EXT=Path('/root/userfolder/ctx-compressor')
X=EXT/'reader_diag'
SMALL=PROJECT/'eval/reader_diag'
HIST=EXT/'reader_heldout'
TRY=HIST/'grpo_r3_checkpoint-250_try2'
A,B='cascade_off_r3','mixed_r2r3'
R2,R3='grpo_r2:checkpoint-125','grpo_r3:checkpoint-250'
CELLS=[m+'|'+t for m in ('reselect','append') for t in ('tight','mid','loose')]
def require(ok,message):
    if not ok: raise ValueError(message)
def read(path):return json.loads(path.read_bytes())
def write_once(path,data):
    with path.open('xb') as out:out.write(data)
def encode(x):return (json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()
def lines(path):
    with path.open() as stream:
        for row in stream:yield json.loads(row)
def decision_key(r):
    return (r['mode'],r['tier'],r['task'],r['question_index'])
def ratio(k,n):return {'count':k,'n':n,'rate':k/n if n else None}
def mean(values):
    v=list(values)
    return {'mean':math.fsum(v)/len(v) if v else None,'n':len(v)}
def inconsistent(items):
    groups=defaultdict(list)
    for token_identity,raw in items: groups[token_identity].append(raw)
    duplicates=[v for v in groups.values() if len(v)>1]
    return ratio(sum(len(set(v))>1 for v in duplicates),len(duplicates))
def paired(left,right):
    require(set(left)==set(right),'paired keys differ')
    n=len(left); counts=Counter();em_counts=[[0,0],[0,0]]
    for k,a in left.items():
        b=right[k]
        counts['left_higher' if a['f1']>b['f1'] else 'right_higher' if a['f1']<b['f1'] else 'equal']+=1
        em_counts[1-int(a['em'])][1-int(b['em'])]+=1
    return {'n':n,'f1_difference_pp':100*math.fsum(a['f1']-right[k]['f1'] for k,a in left.items())/n,
            'em_difference_pp':100*math.fsum(a['em']-right[k]['em'] for k,a in left.items())/n,
            'f1_counts':{k:counts[k] for k in ('left_higher','equal','right_higher')},'em_matrix':em_counts}
def scores(rows):
    rs=list(rows)
    return {'n':len(rs),'f1':mean(r['f1'] for r in rs)['mean'],'em':mean(r['em'] for r in rs)['mean']}
def shape(rows):
    lens=[r['output_tokens'] for r in rows];n=len(lens)
    return {'mean_tokens':mean(lens),'at_least_20':ratio(sum(v>=20 for v in lens),n),'at_48':ratio(sum(v==48 for v in lens),n)}
def comparison(left,right,keys):
    keys=list(keys);diff=[k for k in keys if left[k]['raw_output']!=right[k]['raw_output']]
    return {'raw_differences':ratio(len(diff),len(keys)), 'left_scores':scores(left[k] for k in keys),'right_scores':scores(right[k] for k in keys),
            'length_changed':{'left':mean(left[k]['output_tokens'] for k in diff),'right':mean(right[k]['output_tokens'] for k in diff)},
            'length_all':{'left':mean(left[k]['output_tokens'] for k in keys),'right':mean(right[k]['output_tokens'] for k in keys)}}
def checked(row):
    pred=parse_answer(row['raw_output']); score=score_answer(pred,row['answer'],row['answer_aliases'])
    require(pred==row['pred'] and all(score[k]==row[k] for k in ('em','f1')),'raw QA score mismatch')
    require(not row['input_truncated'] and len(row['output_token_ids'])==row['output_tokens']<=48,'token length mismatch')
    return {k:row[k] for k in ('raw_output','em','f1','evidence_complete','output_tokens','example_id','prompt_sha1')}
def signature(prompt):
    # Store the exact packed token sequence, not text or a hash surrogate.
    return array('I',prompt['token_ids']).tobytes()
def baseline(predfile,reqfile,select):
    selected={};tokens={};within=[];sha_groups=[];token_ids={};token_to_prompt={}
    for p,r in zip(lines(reqfile),lines(predfile),strict=True):
        require(p['config']==r['config'] and p['task']==r['task'] and p['question_index']==r['question_index'] and p['prompt_sha1']==r['prompt_sha1'],'baseline alignment')
        sig=signature(p)
        if sig not in token_ids:token_ids[sig]=len(token_ids)
        raw=r['raw_output'];within.append((token_ids[sig],raw));sha_groups.append((r['prompt_sha1'],raw))
        if r['strategy']==select:
            key=decision_key(r);require(key not in selected,'duplicate baseline key')
            selected[key]=checked(r);tokens[key]=sig
    require(len(selected)==5976,'baseline count')
    return selected,tokens,{'within_token':inconsistent(within),'within_prompt_sha1':inconsistent(sha_groups),'n_requests':len(within),'shape':shape(selected.values())}
def load_arm(arm,tokens,refs):
    want=[R3] if arm==A else [R2,R3];lookup={k:{} for k in want};f=X/arm
    for row in lines(f/'predictions.jsonl'):
        name=row['strategy'];k=decision_key(row)
        require(name in lookup and k not in lookup[name] and k in refs[name],'unregistered/duplicate key')
        ref=refs[name][k]
        require(all(row[m]==ref[m] for m in ('example_id','prompt_sha1','evidence_complete')),'prediction input identity')
        lookup[name][k]=checked(row)
    require(all(set(rows)==set(refs[name]) for name,rows in lookup.items()),'incomplete arm')
    inputs=lines(HIST/'input_preflight/smoke.jsonl')
    smoke=0;n=0
    for src,row in zip(inputs,lines(f/'smoke_predictions.jsonl'),strict=True):
        require(src['prompt_sha1']==row['prompt_sha1'] and src['expected_raw_output']==row['expected_raw_output'],'smoke identity')
        smoke+=src['expected_raw_output']!=row['raw_output'];n+=1
    require(n==40,'smoke size')
    return lookup,ratio(smoke,n)
def by_strategy(name,rows,ids):
    return {'pooled':scores(rows.values()),'cells':{cell:scores(r for k,r in rows.items() if '|'.join(k[:2])==cell) for cell in CELLS},
            'by_evidence':{flag:scores(r for r in rows.values() if r['evidence_complete']==(flag=='complete')) for flag in ['complete','incomplete']},
            'shape':shape(rows.values()),'within':inconsistent((ids[name][k],v['raw_output']) for k,v in rows.items())}
def run():
    m12=sums(PROJECT/'eval/reader_heldout/TASK12_SHA256SUMS');m13=sums(PROJECT/'eval/grpo_r3/TASK13_SHA256SUMS')
    for manifest,folder in ((m12,HIST),(m13,TRY)):
        for f in (folder/'predictions.jsonl',folder/'input_preflight/requests.jsonl'):
            require(checksum(f)==manifest[str(f)],'historical input signature mismatch')
    r2,ids2,h12=baseline(HIST/'predictions.jsonl',HIST/'input_preflight/requests.jsonl',R2)
    r3,ids3,h13=baseline(TRY/'predictions.jsonl',TRY/'input_preflight/requests.jsonl',R3)
    for req in lines(X/'input_preflight/r2_requests.jsonl'):
        k=decision_key(req)
        require(signature(req)==ids2[k] and req['prompt_sha1']==r2[k]['prompt_sha1'],'regenerated r2 input differs')
    ids={R2:ids2,R3:ids3};refs={R2:r2,R3:r3}
    require(r2.keys()==r3.keys(),'historical populations differ')
    same=[k for k in r3 if ids2[k]==ids3[k]]
    require(len(same)==2116 and set(same)=={k for k in r3 if r2[k]['prompt_sha1']==r3[k]['prompt_sha1']},'historical identical subset differs')
    baseline_pair=comparison(r3,r2,same)
    require(baseline_pair['raw_differences']['count']==397,'historical overlap failed')
    require((h12['within_prompt_sha1']['count'],h12['within_prompt_sha1']['n'])==(383,5812) and (h13['within_prompt_sha1']['count'],h13['within_prompt_sha1']['n'])==(485,1612),'historical within-run check failed')
    historic_smoke=list(lines(HIST/'smoke_predictions.jsonl'))
    result={'task':15,'status':'complete','references':{'task12':h12,'try2':h13,'same_inputs_try2_vs_task12':baseline_pair,
            'task12_smoke':ratio(sum(r['raw_output']!=r['expected_raw_output'] for r in historic_smoke),len(historic_smoke))},'arms':{}}
    for arm in (A,B):
        lookup,smoke=load_arm(arm,ids,refs)
        armdata={'strategies':{},'smoke':smoke,'same_inputs_n':len(same),'resource_config':read(X/arm/'run_config.json')}
        for name,rows in lookup.items():
            data=by_strategy(name,rows,ids)
            data['same_inputs_vs_task12_r2']=comparison(rows,r2,same)
            if name==R3:data['all_vs_try2']=comparison(rows,r3,rows)
            if name==R2:data['all_vs_task12_r2']=comparison(rows,r2,rows)
            armdata['strategies'][name]=data
        if arm==B:
            armdata['both_within']=inconsistent((ids[s][k],r['raw_output']) for s,v in lookup.items() for k,r in v.items())
            armdata['paired_r2_minus_r3']=paired(lookup[R2],lookup[R3])
        result['arms'][arm]=armdata
    result['p_a']=result['arms'][A]['strategies'][R3]['within']
    result['p_b']=result['arms'][B]['strategies'][R3]['within']
    return result

"""Render the registered descriptive diagnostics; all score fields are fractions."""
def fmt_ratio(r):
    return f"{r['count']}/{r['n']}（{100*r['rate']:.3f}%）" if r['n'] else '0/0'
def fmt_avg(r):
    return f"{r['mean']:.3f}（n={r['n']}）" if r['n'] else '—（n=0）'
def fs(s):
    return f"{s['f1']*100:.3f}% / {s['em']*100:.3f}%（n={s['n']}）" if s['n'] else '—（n=0）'
def tab(head,rows):
    return '\n'.join('| '+' | '.join(str(x).replace('|',' / ') for x in row)+' |' for row in [head,['---']*len(head),*rows])
def render(r):
    armdata=r['arms'];hist=r['references']; base=hist['same_inputs_try2_vs_task12']
    series=[('a · r3',armdata[A]['strategies'][R3]),('b · r2',armdata[B]['strategies'][R2]),('b · r3',armdata[B]['strategies'][R3])]
    blocks=['# 任务15　Qwen3-8B 留出集读者一致性诊断',
            f"对输入 token 序列逐个相等且出现至少两次的组，臂 a 的 r3 有 {fmt_ratio(r['p_a'])} 组出现多种原始输出，臂 b 的 r3 为 {fmt_ratio(r['p_b'])}。统计保留大小写、空白与标点；每组只计一次。",
            '## 同次运行与跨运行的一致性',
            tab(['运行与策略','不一致组/重复组'],[[n,fmt_ratio(v['within'])] for n,v in series]+[['b · 两策略合并',fmt_ratio(armdata[B]['both_within'])],['历史 try2 · r3',fmt_ratio(hist['try2']['within_token'])],['历史任务12 · 全策略',fmt_ratio(hist['task12']['within_token'])]]),
            '两种策略同题同格、提示 token 序列完全相同的子集有 n=2,116 题次。下表的历史对照为任务12 的 r2，对齐后逐条比较原始输出。',
            tab(['新运行','raw 不同','新运行 F1 / EM','历史 r2 F1 / EM'],[[n,fmt_ratio(v['same_inputs_vs_task12_r2']['raw_differences']),fs(v['same_inputs_vs_task12_r2']['left_scores']),fs(v['same_inputs_vs_task12_r2']['right_scores'])] for n,v in series]+[['历史 try2',fmt_ratio(base['raw_differences']),fs(base['left_scores']),fs(base['right_scores'])]]),
            '全量按题目与格配对，输出长度单位为 token。差异子集的长度均值只取 raw 输出不同的题次。',
            tab(['新运行/历史对照','raw 不同','差异题新/旧均长','全部题新/旧均长'],[[n+' / '+label,fmt_ratio(v[k]['raw_differences']),fmt_avg(v[k]['length_changed']['left'])+' / '+fmt_avg(v[k]['length_changed']['right']),fmt_avg(v[k]['length_all']['left'])+' / '+fmt_avg(v[k]['length_all']['right'])] for n,v,k,label in [(series[0][0],series[0][1],'all_vs_try2','try2'),(series[1][0],series[1][1],'all_vs_task12_r2','任务12 r2'),(series[2][0],series[2][1],'all_vs_try2','try2')]]),
            '## 问答分数与输出长度',
            tab(['范围','a · r3 F1 / EM','b · r2 F1 / EM','b · r3 F1 / EM'],[['合并',*[fs(v['pooled']) for n,v in series]]]+[[c,*[fs(v['cells'][c]) for n,v in series]] for c in CELLS]+[[label,*[fs(v['by_evidence'][k]) for n,v in series]] for k,label in [('complete','该题证据齐全'),('incomplete','该题证据有缺失')]]),
            tab(['运行','平均输出 token','至少 20 token','达到 48 token'],[[n,fmt_avg(v['mean_tokens']),fmt_ratio(v['at_least_20']),fmt_ratio(v['at_48'])] for n,v in [(n,d['shape']) for n,d in series]+[('历史 try2',hist['try2']['shape']),('历史任务12 r2',hist['task12']['shape'])]]),
            '## 同批配对与资源']
    p=armdata[B]['paired_r2_minus_r3'];counts=p['f1_counts']
    blocks.append(f"臂 b 同批 r2 减 r3，F1 差 {p['f1_difference_pp']:+.3f} 个百分点，EM 差 {p['em_difference_pp']:+.3f} 个百分点（各 n={p['n']:,}）。F1 的 r2 高 / 平 / r3 高为 {counts['left_higher']} / {counts['equal']} / {counts['right_higher']}（n={p['n']:,}）。")
    blocks.append(tab(['臂','实际 disable_cascade_attn','设置来源','小样 raw 不同','耗时（秒）','峰值显存（MiB）'],[[lab,str(k['resource_config']['effective_disable_cascade_attn']),k['resource_config']['effective_disable_cascade_attn_source'],fmt_ratio(k['smoke']),f"{k['resource_config']['elapsed_seconds']:.1f}（n=1 次）",f"{k['resource_config']['peak_gpu1_mib']}（n=1 次）"] for lab,k in armdata.items()]))
    blocks.append(f"任务03 的一致性小样在两臂均单独成批，任务12 的历史差异为 {fmt_ratio(hist['task12_smoke'])}。a 按每个任务原顺序提交 24 个请求（n=249 批）；b 每批先放 r2 再放 r3 共 48 条（n=249 批）。")
    blocks.append("输入溯源核验了两层哈希。任务书给出的 5f584e73… 是 try2 输入清单 preflight.json 的 SHA-256；其中登记的 requests.jsonl 的 SHA-256 是 01d7bf55…（n=5,976 条请求）。两份文件均与历史签名一致。r2 由任务12 构造函数重建，逐条核对历史提示和 token（n=5,976）。")
    return '\n\n'.join(blocks)+'\n'
def main():
    r=run()
    for arm,z in r['arms'].items():
        require(all(v['pooled']['n']==5976 and all(c['n']==996 for c in v['cells'].values()) and sum(c['n'] for c in v['by_evidence'].values())==5976 for v in z['strategies'].values()),'score denominators')
    md=render(r)
    targets={SMALL/'reader_diag_results.json':encode(r),SMALL/'reader_diag_report.md':md.encode()}
    require(all(not p.exists() for p in targets),'refuse to overwrite published report')
    for p,payload in targets.items():write_once(p,payload)
    print('p_a',fmt_ratio(r['p_a']),'p_b',fmt_ratio(r['p_b']))
if __name__=='__main__':main()

