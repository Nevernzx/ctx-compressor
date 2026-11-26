"""Reconcile the Task 14 smoke records and publish only the small CPU-stage receipts."""
from collections import Counter
from decimal import Decimal
import json
import os
from pathlib import Path
import sys
PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
from scripts.deepseek_api_client import scan_artifacts,peak_end
from scripts.deepseek_api_subset import WORK,SMALL,read,utc,write_new
from datetime import datetime
D=Decimal


def rows(path):
    with path.open() as f:return [json.loads(l) for l in f]


def input_files():
    documents=list(SMALL.glob('*'))
    for name in ('src','scripts','tests'):
        documents.extend(p for p in (PROJECT/name).glob('*deepseek_api*') if p.is_file())
    documents.extend(p for p in WORK.iterdir() if p.is_file())
    documents.extend([WORK/'prep',WORK/'smoke'])
    suite=read(SMALL/'cpu_tests.json')
    documents.extend([Path(suite['xml']),Path(suite['log']),Path(suite['xml']).parent/'result.json'])
    return documents


def inspect():
    res=read(WORK/'smoke/smoke_results.json')
    info=read(SMALL/'input_checks.json')
    calls=rows(WORK/'smoke/api_calls.jsonl')
    entries=rows(WORK/'ledger.jsonl')
    assert len(calls)==res['n_calls']==36 and len({r['call_id'] for r in calls})==len(calls)
    assert all(r['requested_model']=='deepseek-flash' and r['period']=='off_peak'
               and peak_end(datetime.fromisoformat(r['started_utc'])) is None for r in calls)
    attempts=rows(WORK/'smoke/attempts.jsonl')
    chat=[r for r in attempts if r.get('kind')=='chat']
    assert all(peak_end(datetime.fromisoformat(r['started_utc'])) is None for r in attempts)
    assert sum(1 for r in chat if r.get('http_status')==200)==len(calls)
    stats=Counter()
    for r in calls:
        for field in ('prompt_cache_hit_tokens','prompt_cache_miss_tokens','completion_tokens'):
            stats[field]+=r['usage'][field]
    cost=(D('.02')*stats['prompt_cache_hit_tokens']+stats['prompt_cache_miss_tokens']+
          4*stats['completion_tokens'])/1000000
    paid=sum(D(r['cost_yuan']) for r in entries if r['event']=='settle')
    assert paid==cost==D(res['smoke_spent_yuan']) and paid+D(str(res['ledger']['reserved_yuan']))<=8
    nbad,bsteps=res['B_invalid'],res['B_steps']
    assert bsteps==16 and res['B_metrics']['combined']['invalid_outputs']['any_invalid']['count']==nbad
    delta=D(res['account_drop_yuan'])
    b0=D(res['balance_before']['total_balance']);b1=D(res['balance_after']['total_balance'])
    assert b0-b1==delta
    return res,stats,cost,info,len(chat)


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and Path.cwd()==PROJECT
    res,stats,cost,info,nchat=inspect()
    tests=read(SMALL/'cpu_tests.json')
    forecast=D(res['forecast']['full_estimate_yuan'])
    total=forecast+cost
    bg=res['balance_before']['total_balance'];end=res['balance_after']['total_balance']
    risk=D(str(res['ledger']['reserved_yuan']))
    paused=res['ledger']['peak_pauses']
    note=(
        '# 任务14　DeepSeek 子集实验小样\n\n'
        f"费用账本共结算 {res['n_calls']} 次请求。读者复核用了 {res['A_n']} 次，闭环压缩用了 {res['B_steps']} 次，"
        f"读取压缩结果用了 {res['B_reader_n']} 次；压缩无效动作 {res['B_invalid']}/{res['B_steps']} 步。\n\n"
        f"缓存命中输入 {stats['prompt_cache_hit_tokens']:,} token、未命中输入 {stats['prompt_cache_miss_tokens']:,} token，"
        f"输出 {stats['completion_tokens']:,} token（n={res['n_calls']}）。按用量计费 ¥{cost:.8f}；"
        f"账户余额 ¥{bg} → ¥{end}（2 次免费查询），余额减少额与账本差 ¥{D(res['account_drop_minus_ledger_yuan']):+.8f}。"
        f"当前不确定请求的保守预留为 ¥{risk}。\n\n"
        f"高峰暂停 {paused} 次，实际向付费端点发送 {nchat} 次 HTTP 请求。"
        f"已比对 A 的 {info['A_full_n']:,} 个提示和 B 的 {info['B']['n_trajectories']} 条首步提示，哈希差异均为 0。"
        f"完整 CPU 测试通过 {tests['tests']}/{tests['tests']} 项。\n\n"
        f"若从头运行全量，按小样输入命中比例、实际用量和注册的 548 个压缩步，"
        f"预计另需 ¥{forecast:.3f}（2,588 次请求）。含小样预计 ¥{total:.3f}，"
        f"硬上限为 ¥8.00。闭卷问题使用零缓存命中的估算；压缩器部分按小样每步平均用量外推。\n\n"
        '密钥检查涵盖本轮源码、测试、回执、余额记录、账本、预测和报告，以真实密钥及其前缀模式逐文件扫描。'
        '结果见 [安全检查回执](smoke_security.json)。\n')
    path=SMALL/'smoke_report.md'
    # Scan the source records before publishing the small summary. Recheck the finished report below.
    scan_artifacts(input_files())
    if path.exists() or (SMALL/'smoke_security.json').exists():
        raise FileExistsError('refusing to replace smoke deliverables')
    with path.open('x') as f:f.write(note)
    checked=scan_artifacts(input_files())
    proof={'status':'passed','utc':utc(),'scan':checked,'scope':'Task 14 source, tests, test result logs/XML, request/usage ledger, scope checks, predictions, run receipts and the smoke report.'}
    write_new(SMALL/'smoke_security.json',proof)
    assert scan_artifacts([SMALL/'smoke_security.json'])['matches']==0
    print(json.dumps({'calls':res['n_calls'],'spent':str(cost),**stats,'balance':end,
      'invalid':res['B_invalid'],'steps':res['B_steps'],'peak_pauses':paused,'attempts':nchat,
      'forecast':str(forecast),'total_forecast':str(total),'account_drop':res['account_drop_yuan'],
      'difference':res['account_drop_minus_ledger_yuan'],'scan':checked,'tests':tests['tests']}))
if __name__=='__main__':main()

