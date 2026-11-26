"""Freeze the verified Task 19 model and archival references without selecting new settings."""
from pathlib import Path
import os
import sys
from concurrent.futures import ThreadPoolExecutor

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
from scripts import test_v1_data as data
from src import reader_heldout_inputs as reuse
from src.reranker_v1 import MODEL as RERANKER_TOK, need, read, record, require_new, verify

E, OUT = data.E, data.SMALL
T11=PROJECT/"eval/grpo_r2"
T17=PROJECT/"eval/reranker_v1"
T18=PROJECT/"eval/grpo_r4"
BASE=E/"grpo_r2/merged_start"
ADAPTERS={"r2":E/"grpo_r2/train/checkpoint-125","r4":E/"grpo_r4/train/checkpoint-500"}
R4_TRACE=E/"eval_rollouts_grpo_r4/checkpoint-500/steps.jsonl"
PRE=OUT/"model_preflight.json"


def key(p):
    return str(p.relative_to(PROJECT)) if p.is_relative_to(PROJECT) else str(p)


def checked_models():
    global_sums=reuse.sums(PROJECT/"SHA256SUMS")
    signed={}
    manifest_records={}
    for n,folder in ((11,T11),(17,T17),(18,T18)):
        m=folder/f"TASK{n}_SHA256SUMS"
        verify(m,global_sums[key(m)])
        signed[n]=reuse.sums(m)
        manifest_records[str(n)]=record(m)
    requested={}
    def claim(p,n):
        need(p.resolve()==p and p.is_file(), f"missing or redirected signed path: {p}")
        digest=signed[n][key(p)]
        need(p not in requested or requested[p]==digest, f"conflicting manifest hashes: {p}")
        requested[p]=digest
        return {"path":str(p),"sha256":digest,"bytes":p.stat().st_size}

    m17=read(T17/"model_card.json")
    tr17=read(T17/"training.json")
    tau=read(T17/"tau_selection.json")
    r4=read(T18/"selected_checkpoint.json")
    r2=read(T11/"selected_checkpoint.json")
    claim(T17/"model_card.json",17)
    claim(T17/"training.json",17)
    claim(T17/"tau_selection.json",17)
    claim(T18/"selected_checkpoint.json",18)
    claim(T11/"selected_checkpoint.json",11)
    rr=Path(tr17["chosen"]["checkpoint"])
    need(rr==E/"reranker_v1/train/checkpoint-5300" and tau["tau"]==.5 and tau["n_tasks"]==100
         and r4["strategy"]=="checkpoint-500" and Path(r4["path"])==ADAPTERS["r4"]
         and r2["strategy"]=="checkpoint-125" and Path(r2["path"])==ADAPTERS["r2"],
         "frozen selected checkpoint or tau differs")
    card_files=[claim(RERANKER_TOK/item["file"],17) for item in m17["files"]]
    need(all(item["sha256"]==meta["sha256"]==ref["sha256"] for item,meta,ref in
             zip(card_files,m17["files"],card_files,strict=True)), "reranker original snapshot signatures differ")
    rr_models=[claim(p,17) for p in sorted(rr.iterdir()) if p.is_file()]
    need(any(a["path"]==str(rr/"model.safetensors") and a["sha256"]==tr17["chosen"]["weights"]["sha256"]
             for a in rr_models),"selected reranker signature differs")
    base_files=[claim(p,11) for p in sorted(BASE.iterdir()) if p.is_file()]
    need({"config.json","model.safetensors","tokenizer.json","tokenizer_config.json","merge_record.json"}
         <= {Path(x["path"]).name for x in base_files},"merged starting model incomplete")
    merge=read(BASE/"merge_record.json")
    start_ref=claim(T11/"start_model.json",11)
    start=read(Path(start_ref["path"]))
    need(start["strategy"]=="checkpoint-3000" and start["version"]=="v3"
         and merge["adapter_sha256"]==start["adapter_sha256"],"Task 11 start does not identify SFT v3 ckpt-3000")
    adapter_refs={}
    for arm, d in ADAPTERS.items():
        n=11 if arm=="r2" else 18
        adapter_refs[arm]=[claim(d/f,n) for f in ("adapter_config.json","adapter_model.safetensors")]
        wanted=r2 if arm=="r2" else r4
        need(adapter_refs[arm][1]["sha256"]==wanted["adapter_sha256"]
             and read(d/"adapter_config.json")["base_model_name_or_path"]==str(BASE),f"frozen adapter differs: {arm}")
    r4_engine=claim(T18/"checkpoint-500/run.json",18)
    f=read(Path(r4_engine["path"]))
    params=f["engine_parameters"]
    need(f["greedy"] is True and f["temperature"]==0 and f["max_tokens"]==256 and f["model"]==str(BASE)
         and params["model"]==params["tokenizer"]==str(BASE) and params["enforce_eager"] and params["enable_lora"],
         "Task 18 compressor engine settings differ")
    reader_file=PROJECT/"eval/reader/run_config.json"
    verify(reader_file,global_sums[key(reader_file)])
    rc=read(reader_file)
    need(rc["generation"]["max_tokens"]==48 and rc["generation"]["seed"]==20261117
         and rc["generation"]["temperature"]==0 and rc["generation"]["enable_thinking"] is False,
         "frozen reader protocol differs")
    g0_trace=claim(R4_TRACE,18)
    r4_results=claim(T18/"checkpoint-500/checkpoint-500_results.json",18)
    expected=read(Path(r4_results["path"]))["steps_artifact"]
    need(expected["path"]==g0_trace["path"] and expected["sha256"]==g0_trace["sha256"],"G0 reference differs")

    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(lambda item:verify(*item),requested.items()))
    return {
        "status":"passed","manifests":manifest_records,"base":{"path":str(BASE),"strategy":"SFT v3 checkpoint-3000","files":base_files},
        "adapters":{name:{"path":str(ADAPTERS[name]),"files":files} for name,files in adapter_refs.items()},
        "reranker":{"path":str(rr),"tokenizer":str(RERANKER_TOK),"model_files":rr_models,
                    "tokenizer_snapshot":card_files,"template":tr17["template"],"tau":tau["tau"],
                    "checkpoint":tr17["chosen"]["step"]},
        "compressor":{"engine":params,"sampling":{"temperature":0,"max_tokens":256,"seed":f["seed"]},
                      "frozen_source":r4_engine},
        "reader":{"frozen_source":record(reader_file),"engine":{**rc["parameters"],"disable_cascade_attn":True},
                  "generation":rc["generation"],"packages":rc["package_versions"]},
        "G0":{"ids":[0,1,2],"sampling_rule":"The three smallest archived held-out task IDs, fixed before replay.",
              "threshold":.05,"trace":g0_trace},
    }


def load():
    frozen=read(PRE)
    need(frozen["status"]=="passed","model preflight did not pass")
    return frozen


if __name__=="__main__":
    need(os.environ.get("CUDA_VISIBLE_DEVICES")=="", "model checks run on CPU")
    require_new(PRE)
    reuse.write_new(PRE,checked_models())
    print("Frozen Task 11/17/18 manifests, adapters, merged start and RR model verified.",flush=True)
