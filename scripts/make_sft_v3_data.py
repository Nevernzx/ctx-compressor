"""Build Task 10 targets from policy states and freeze the registered 20k + 10k sample."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import random
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from scripts import make_sft_data as sft
from scripts import rollout_dagger_v3 as rollout
from scripts.hedge_action import candidate_ids, ranked_confidence
from scripts.sample_sft_subset import metadata_identity, require, write_json_exclusive
from scripts.rollout_model import sha256, file_record
from src.build_traj import oracle_action, render_target
from src.heldout_eval import apply_model_output
from src.model_inference import compact, inference_prompt
from src.sft_protocol import encode_sft_record, length_distribution, make_messages, restore_state_keys

DATA = rollout.EXTERNAL / "data/sft_v3"
OUT = rollout.SMALL
MAX_LENGTH = 8392
SAMPLE_SEED = 20261122
MODES = ("reselect","append")  # Task 05/06 mode order for the single seeded generator
REF = PROJECT / "eval/sft_run_v2/sample_manifest.json"
REF_SHA = "a03db13d44a87dffcfab5d883ab16ee1acc1126a715f8449e6ad3e817c5f6c91"


def build_target(state, hedge):
    # Task 04/06 stores oracle_action(state) as sft_target, then uses render_target.
    base = oracle_action(state)
    base_result = apply_model_output(state,render_target({**state,"sft_target":base}))
    require(base_result["action_valid"], "Invalid base target")
    scores = {int(k):v for k,v in hedge["scores"].items()}
    ranked, p1 = ranked_confidence(candidate_ids(state),scores)
    require(hedge["tau"] == rollout.TAU and p1 == hedge["p1"], "Recorded confidence/threshold differs from scores")
    extra = next((i for i in ranked if i not in base["add"] and i not in state["kept_ids"]),None)
    fits = extra is not None and base_result["used_after"] + state["ntok"][extra] <= state["budget"]
    should_hedge = p1 is not None and p1 < rollout.TAU and fits
    target = {"add":list(base["add"]),"drop":list(base["drop"])}
    if should_hedge:
        target["add"].append(extra)
    result = apply_model_output(state,render_target({**state,"sft_target":target}))
    require(result["action_valid"] and result["used_after"] <= state["budget"], "Invalid or over-budget target")
    require((set(base_result["keep_after"]) & set(state["gold"])).issubset(result["keep_after"]), "Lost oracle-kept gold")
    actual_extras = set(result["keep_after"]) - set(base_result["keep_after"])
    require(actual_extras == ({extra} if should_hedge else set()) and not actual_extras.intersection(state["gold"]),
            "Extra must be non-gold and present iff the hedge condition holds")
    return target, result


def encode(row, template, audit):
    state = restore_state_keys(row["state"])
    require(row["split"]=="train" and row["state_id"] == f"{row['trajectory_id']}:{row['step']}"
            and row["mode"] == state["mode"] and row["budget"] == state["budget"], "Row identity mismatch")
    target, result = build_target(state, row["hedge"])
    messages = make_messages({**state,"sft_target":target})
    user_text = messages[0]["content"]
    expected = row["generation_metadata"]
    require(user_text == row["user_text"]
            and hashlib.sha256(user_text.encode()).hexdigest() == expected["user_text_sha256"],
            "Re-rendered policy user text differs")
    encoded = encode_sft_record(template,messages)
    if audit:
        prompt = inference_prompt(template.tokenizer,state)
        ids = prompt["prompt_token_ids"]
        require(ids == expected["prompt_token_ids"] and prompt["prompt_sha256"] == expected["prompt_sha256"]
                and encoded["input_ids"][:len(ids)] == ids, "Training, re-encoded and live C prompt IDs differ")
    meta = {"task":row["task"], "step":row["step"], "mode":row["mode"], "tier":row["budget_tier"], "split":"train",
            **{field:encoded[field] for field in sft.LENGTH_FIELDS}, "target_add_empty":not target["add"],
            "target_drop_nonempty":bool(target["drop"]),"budget_binding":not set(state["gold"]).issubset(result["keep_after"]),
            "state_id":row["state_id"],"source":"dagger_v3"}
    older = row["step"] >= 1 and row["mode"]=="reselect"
    counts = Counter(n=1, two_adds=int(len(target["add"])==2), nonempty_drop=int(bool(target["drop"])),
                     drops=len(target["drop"]), reselect_step1=int(older),
                     nongold_in_kept=int(older and bool(set(state["kept_ids"])-set(state["gold"]))),
                     overlength=int(meta["total_tokens"]>MAX_LENGTH), ids_compared=int(audit))
    return {"messages":messages,"meta":meta},counts


def ratio(num,n):
    return {"count":num,"n":n,"rate":num/n if n else None}


def statistics(c,hist):
    n = c["n"]
    return {"n_states":n,"two_adds":ratio(c["two_adds"],n),
            "nongold_in_kept_reselect_step_ge_1":ratio(c["nongold_in_kept"],c["reselect_step1"]),
            "nonempty_drop":ratio(c["nonempty_drop"],n), "mean_drops":{"sum":c["drops"],"n":n,"mean":c["drops"]/n},
            "full_training_token_length":length_distribution(hist),
            "overlength":ratio(c["overlength"],n),"user_text_checked":n,"prompt_ids_checked":c["ids_compared"]}


def select_ids(by_mode, per_mode=5000, seed=SAMPLE_SEED):
    rng = random.Random(seed)
    require(set(by_mode) == set(MODES), "Missing mode pool")
    chosen = []
    for mode in MODES:
        pool = by_mode[mode]
        require(len(pool) >= per_mode and len(pool)==len(set(pool)), "Insufficient or duplicate DAgger states")
        chosen.extend(rng.sample(pool,per_mode))
    require(len(chosen)==len(set(chosen))==2*per_mode, "DAgger IDs collide")
    return chosen


def legacy_rows(out):
    require(sha256(REF)==REF_SHA,"Task 06 sample receipt changed")
    old = rollout.load(REF)
    source = old["sources"]["train"]
    files = old["files"]
    for key, expected in (("train",20000),("eval_loss",1000),("sample_ids",None)):
        rec = files[key]
        require(sha256(rec["path"])==rec["sha256"],f"Changed Task 06 {key}")
        if expected is not None:
            require(rec["lines"]==expected, "Wrong Task 06 subset count")
    ids = rollout.load(Path(files["sample_ids"]["path"]))
    require(len(ids["train"])==20000 and len(ids["heldout_eval_loss"])==1000, "Frozen sample ID count mismatch")
    by_line = {r["source_line_1based"]:r for r in ids["train"]}
    require(len(by_line)==20000, "Duplicate frozen source line")
    seen = set()
    digest = hashlib.sha256()
    with Path(source["path"]).open("rb") as full, Path(files["train"]["path"]).open("rb") as subset:
        for pos,line in enumerate(full,1):
            digest.update(line)
            if pos in by_line:
                item = json.loads(line)
                key = metadata_identity(item)
                require(line == subset.readline() and key == metadata_identity(by_line[pos])
                        and key not in seen and item["meta"]["split"]=="train", "Frozen ID or source bytes differ")
                seen.add(key)
                out.write(line)
        require(len(seen)==20000 and not subset.readline() and pos == source["lines"] and digest.hexdigest()==source["sha256"],
                "Frozen 20k/source validation incomplete")
    with Path(files["eval_loss"]["path"]).open("rb") as stream:
        for entry in ids["heldout_eval_loss"]:
            row = json.loads(stream.readline())
            require(metadata_identity(entry)==metadata_identity(row) and row["meta"]["split"]=="heldout","Wrong evaluation-loss ID")
        require(not stream.readline(),"Extra evaluation rows")
    return files


def main(smoke):
    require(PROJECT == Path("/root/workplace/ctx-compressor") and os.environ.get("CUDA_VISIBLE_DEVICES")==""
            and Path(sys.prefix).resolve()==rollout.EXTERNAL/"envs/train", "Use the 5880 CPU-only train environment")
    series = rollout.SMOKE if smoke else rollout.SUMMARY
    summary = rollout.load(series)
    n = summary["n_steps"]
    steps = Path(summary["steps_file"]["path"])
    require(steps == rollout.ROOT/("smoke" if smoke else "rollouts")/"steps.jsonl" and summary["status"]=="complete"
            and ((smoke and summary["n_tasks"]==2) or (not smoke and n==22158 and summary["n_tasks"]==400))
            and sha256(steps)==summary["steps_file"]["sha256"], "C source validation failed")
    folder = rollout.ROOT/"smoke_targets" if smoke else DATA
    folder.mkdir(parents=True,exist_ok=False)
    receipt = OUT/("dagger_smoke_targets.json" if smoke else "sample_manifest.json")
    require(not receipt.exists(),"D receipt exists")
    sft.initialize_worker(rollout.EXTERNAL/"models/Qwen3-1.7B")
    template = sft.WORKER_TEMPLATE
    eligible = folder/"dagger_eligible.jsonl"
    ids, counts, hist = defaultdict(list),defaultdict(Counter),defaultdict(Counter)
    seen, num = set(), 0
    with eligible.open("xb") as pool:
        for nrow,row in enumerate(rollout.lines(steps)):
            state_id = row["state_id"]
            require(state_id not in seen and row["task"] in summary["task_ids"], "Duplicate state or unregistered task")
            seen.add(state_id)
            # Every row in smoke; systematic 2% of full input, with actual Swift prefix comparison.
            new,c = encode(row,template,smoke or nrow%50==0)
            cell = row["mode"]+"|"+row["budget_tier"]
            length = new["meta"]["total_tokens"]
            counts[cell].update(c)
            hist[cell][length] += 1
            if length <= MAX_LENGTH:
                pool.write((compact(new)+"\n").encode())
                ids[row["mode"]].append(state_id)
                num += 1
            if (nrow+1)%1000 == 0:
                print(f"targets checked={nrow+1}/{n}",flush=True)
    require(len(seen)==n and len(counts)==6 and dict((k,c["n"]) for k,c in counts.items())==summary["cells"],
            "C/D task grid mismatch")
    over = n-num
    stats = {k:statistics(c,hist[k]) for k,c in sorted(counts.items())}
    facts = {"status":"complete","n_states":n,"n_eligible":num,"overlength":ratio(over,n),"max_length":MAX_LENGTH,
             "cells":stats,"c_receipt":file_record(series),"user_text_checked":n,
             "prompt_ids_checked":sum(c["ids_compared"] for c in counts.values()),"eligible_file":{**file_record(eligible),"lines":num}}
    require(facts["prompt_ids_checked"]*100 >= n, "Prompt identity coverage under 1%")
    if over*100>n:
        write_json_exclusive(folder/"overlength_failure.json",{**facts,"status":"failed"})
        raise ValueError(f"{over}/{n} rows exceed {MAX_LENGTH}, above 1%")
    if smoke:
        write_json_exclusive(receipt,facts)
        print(f"smoke targets passed {n}/{n}, overlength {over}/{n}",flush=True)
        return
    picked = select_ids(ids)
    selected = set(picked)
    train = folder/"train_30k.jsonl"
    matched = set()
    with train.open("xb") as out:
        old = legacy_rows(out)
        with eligible.open("rb") as pool:
            for line in pool:
                row = json.loads(line)
                state_id = row["meta"]["state_id"]
                if state_id in selected:
                    require(state_id not in matched, "Selected DAgger state duplicated")
                    out.write(line)
                    matched.add(state_id)
    require(matched==selected and len(matched)==10000,"DAgger sample mismatch")
    sample = write_json_exclusive(folder/"sample_ids.json",{"seed":SAMPLE_SEED,"mode_order":MODES,
        "source_order":"C source order within each mode, one seeded RNG","frozen_v2_sample_ids":old["sample_ids"],
        "dagger_state_ids":picked})
    files={"train":{**file_record(train),"lines":30000}, "eval_loss":old["eval_loss"],"sample_ids":sample}
    with train.open("rb") as stream:
        require(sum(1 for _ in stream)==30000,"Train must contain 30000 rows")
    facts.update(seed=SAMPLE_SEED,files=files,frozen_v2_train=old["train"],frozen_v2_manifest=file_record(REF),
                 frozen_rows_source_id_and_bytes_checked=20000,fixed_eval_loss_ids_checked=1000,
                 selected_by_mode={m:5000 for m in MODES})
    write_json_exclusive(receipt,facts)
    print(f"{receipt} train rows=30000 eval rows=1000",flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke",action="store_true")
    main(parser.parse_args().smoke)

