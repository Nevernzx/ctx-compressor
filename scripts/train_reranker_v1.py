"""Task 17 full-parameter BF16 reranker training and reusable score caches."""
import argparse
from collections import defaultdict
from importlib.metadata import version
import json
from math import ceil, fsum, isfinite
import os
from pathlib import Path
import random
import sys
import time

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from scripts import reranker_v1_data as data
from src.reader_heldout_inputs import rows, write_new
from src.reranker_v1 import (BATCH, MODEL, SEED, SMALL, WORK, average_precision, choose_checkpoint,
                            logit_difference, need, probability, read, record, require_new,
                            signed_model, verify)

MICROBATCH = 8
EVAL_MICROBATCH = 64
LR = 1e-5
PROBE_STEPS = 10
PROBE_WARMUP = 2
CAP_SECONDS = 3 * 3600
WEIGHT_DECAY = .01


def load_model(path, dtype):
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(str(path), local_files_only=True, trust_remote_code=False,
                                              dtype=dtype, attn_implementation="sdpa").to("cuda")


def tensor_batch(encoded, label_tokens, device):
    import torch
    width = max(map(len, encoded))
    n = len(encoded)
    left = np.full((n,width),label_tokens["pad_token_id"],dtype=np.int64)
    mask = np.zeros((n,width),dtype=np.int64)
    for i, ids in enumerate(encoded):
        left[i,-len(ids):] = ids
        mask[i,-len(ids):] = 1
    return torch.from_numpy(left).to(device), torch.from_numpy(mask).to(device)


def score_pairs(model, encoded, label_tokens, device="cuda"):
    """Read the same two logits as the full head, avoiding dense logits for every token."""
    import torch
    import torch.nn.functional as F
    ids, mask = tensor_batch(encoded,label_tokens,device)
    last = model.model(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state[:, -1, :]
    head = model.lm_head
    label_ids = [label_tokens["yes_token_id"], label_tokens["no_token_id"]]
    bias = head.bias[label_ids] if head.bias is not None else None
    values = F.linear(last, head.weight[label_ids], bias)
    return logit_difference(values.float(), 0, 1)


def order_for(description,ratio):
    entry = description["partitions"]["train"]["negative_ratios"][str(ratio)]
    path = Path(entry["file"]["path"])
    verify(path,entry["file"]["sha256"])
    order = np.load(path,allow_pickle=False)
    need(len(order)==entry["n"],"sample order differs")
    return order


def set_seeds(torch):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)


def optimizer_for(model, updates):
    import torch
    from transformers import get_cosine_schedule_with_warmup
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, fused=True)
    warmup = ceil(updates * .03)
    scheduler = get_cosine_schedule_with_warmup(optimizer,num_warmup_steps=warmup,num_training_steps=updates)
    return optimizer,scheduler,warmup


def update(model, optimizer, scheduler, bank, examples, template):
    import torch
    import torch.nn.functional as F
    genuine = len(examples)
    # Keep the registered 64 physical pairs in a short final batch. Zero-weight padding keeps
    # every sampled pair exactly once in the epoch loss, including the final genuine examples.
    physical = list(examples) + [examples[-1]] * (BATCH - genuine)
    optimizer.zero_grad(set_to_none=True)
    loss_sum = torch.zeros((), dtype=torch.float32, device="cuda")
    for offset in range(0,BATCH,MICROBATCH):
        block = physical[offset:offset+MICROBATCH]
        encoded = [bank.get(i) for i in block]
        labels = torch.tensor([bank.index[i]["label"] for i in block],dtype=torch.float32,device="cuda")
        mask = torch.arange(offset,offset+len(block),device="cuda") < genuine
        with torch.autocast(device_type="cuda",dtype=torch.bfloat16):
            logits = score_pairs(model,encoded,template)
            loss = (F.binary_cross_entropy_with_logits(logits,labels,reduction="none") * mask).sum() / genuine
        loss.backward()
        loss_sum += loss.detach()
    loss_number = float(loss_sum.item())
    if not isfinite(loss_number):
        raise FloatingPointError("nonfinite pair BCE")
    optimizer.step()
    scheduler.step()
    return loss_number


def score_bank(model,bank,template,output,picked=None):
    import torch
    order = [i for i,row in enumerate(bank.index) if picked is None or row["task"] in picked]
    summaries = defaultdict(list)
    started = time.monotonic()
    truncation = 0
    with output.open("x",encoding="utf-8") as stream, torch.inference_mode():
        for offset in range(0,len(order),EVAL_MICROBATCH):
            block = order[offset:offset+EVAL_MICROBATCH]
            logits = score_pairs(model, [bank.get(i) for i in block],template).tolist()
            for i,score in zip(block,logits,strict=True):
                item = bank.index[i]
                task, pid, y = (item[k] for k in ("task","pid","label"))
                summaries[task].append((score,y))
                truncation += item["truncated"]
                stream.write(data.compact({"task":task,"pid":pid,"label":y,"score":score,"prob":probability(score)}) + "\n")
    aps = {str(task):average_precision(values) for task,values in summaries.items()}
    return {"mAP":fsum(aps.values()) / len(aps),"AP_by_task":aps,"n_tasks":len(aps),"n_pairs":len(order),
            "n_truncated":truncation,"elapsed_seconds":time.monotonic()-started,"file":record(output)}


def select_val100(source,output,ids):
    wanted = set(ids)
    result = data.dump_rows(output,(row for row in rows(source) if row["task"] in wanted))
    need(result["n"]>0,"empty val100 scores")
    return result


def clear_cuda():
    import gc
    import torch
    gc.collect()
    torch.cuda.empty_cache()


def smoke(bank, samples, template, optimizer_steps):
    import torch
    set_seeds(torch)
    initial = load_model(MODEL,torch.float32)
    initial.train()
    optim, scheduler, _ = optimizer_for(initial,optimizer_steps)
    timings = []
    losses = []
    for step in range(PROBE_WARMUP+PROBE_STEPS):
        begin = step*BATCH
        batch = samples[begin:begin+BATCH]
        torch.cuda.synchronize()
        started = time.monotonic()
        losses.append(update(initial,optim,scheduler,bank,batch,template))
        torch.cuda.synchronize()
        elapsed = time.monotonic()-started
        if step >= PROBE_WARMUP:
            timings.append(elapsed)
    del initial,optim,scheduler
    clear_cuda()
    return {"measured_updates":len(timings),"warmup_updates":PROBE_WARMUP,"seconds":timings,
            "losses":losses,"seconds_per_update":fsum(timings)/len(timings),
            "reloaded_snapshot_before_formal_fit":True}


def run():
    need(Path.cwd()==PROJECT and os.environ.get("CUDA_VISIBLE_DEVICES")=="1" and
         os.environ.get("CUDA_DEVICE_ORDER")=="PCI_BUS_ID","expected admitted GPU 1 only")
    gate=read(WORK/"stages/S1/receipt_gate.json")
    need(gate["gpu"]["index"]==1,"S1 GPU gate missing")
    import torch
    need(torch.cuda.device_count()==1 and torch.cuda.is_bf16_supported(),"GPU 1 BF16 is unavailable")
    cuda = torch.device("cuda",0)  # Local CUDA ordinal 0 is physical GPU 1.
    torch.cuda.set_device(cuda)
    directory=WORK/"train"
    scores=WORK/"scores"
    require_new(directory,scores,SMALL/"training.json")
    directory.mkdir()
    scores.mkdir()
    card = signed_model()
    proof = data.verify_split()
    verify(data.TOKENS,data.checksum(SMALL/"tokenization.json"))
    desc = read(data.TOKENS)
    template = desc["template"]
    banks = {key:data.Bank(key,desc) for key in data.PARTITIONS}
    samples = {r:order_for(desc,r) for r in (7,3)}
    steps = {r:ceil(len(v)/BATCH) for r,v in samples.items()}
    probe = smoke(banks["train"],samples[7],template,steps[7])
    estimates = {r:count*probe["seconds_per_update"] for r,count in steps.items()}
    ratio = 3 if estimates[7]>CAP_SECONDS else 7
    order = samples[ratio]
    write_new(directory/"throughput_smoke.json", {**probe,"estimated_fit_seconds":estimates,
              "limit_seconds":CAP_SECONDS,"chosen_negative_ratio":ratio})
    zero = load_model(MODEL,torch.bfloat16).eval()
    val = banks["train-val"]
    zero_val = score_bank(zero,val,template,directory/"zero_shot_val_200.jsonl")
    zero_held = score_bank(zero,banks["heldout"],template,scores/"zero_shot_heldout.jsonl")
    del zero
    clear_cuda()
    set_seeds(torch)
    network = load_model(MODEL,torch.float32)
    need(all(p.requires_grad for p in network.parameters()),"full-parameter fitting required")
    params = sum(w.numel() for w in network.parameters())
    network.train()
    optim,scheduler,warmup = optimizer_for(network,steps[ratio])
    checkpoints = []
    seen = 0
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    with (directory/"optimization.jsonl").open("x",encoding="utf-8",buffering=1) as log:
        for step in range(1,steps[ratio]+1):
            offset = (step-1)*BATCH
            examples = order[offset:offset+BATCH]
            begin = time.monotonic()
            loss = update(network,optim,scheduler,banks["train"],examples,template)
            seen += len(examples)
            log.write(data.compact({"step":step,"pairs":len(examples),"physical_pairs":BATCH,
                                    "BCE":loss,"lr_next":scheduler.get_last_lr()[0],
                                    "seconds":time.monotonic()-begin})+"\n")
            if step%100==0 or step==steps[ratio]:
                print(f"train step={step}/{steps[ratio]} pairs={seen} loss={loss:.6f}",flush=True)
            if step%1000!=0 and step!=steps[ratio]:
                continue
            checkpoint = directory / f"checkpoint-{step}"
            require_new(checkpoint)
            network.save_pretrained(checkpoint,safe_serialization=True,max_shard_size="10GB")
            candidate=load_model(checkpoint,torch.bfloat16).eval()
            checked=score_bank(candidate,val,template,checkpoint/"validation_scores.jsonl")
            checkpoints.append({"step":step,"checkpoint":str(checkpoint),**checked,
                                "weights":record(checkpoint/"model.safetensors")})
            del candidate
            clear_cuda()
            print(f"validation step={step} mAP={checked['mAP']:.8f}",flush=True)
    fit_seconds = time.monotonic()-started
    peak = int(torch.cuda.max_memory_allocated())
    del network,optim,scheduler
    clear_cuda()
    need(seen==len(order) and zero_val["n_tasks"]==200 and all(c["n_tasks"]==200 for c in checkpoints),
         "training or train-val AP is incomplete")
    chosen = choose_checkpoint(checkpoints)
    selected = load_model(Path(chosen["checkpoint"]),torch.bfloat16).eval()
    held = score_bank(selected,banks["heldout"],template,scores/"selected_heldout.jsonl")
    val100 = set(proof["val100_tasks"])
    val100_cache = select_val100(Path(chosen["file"]["path"]),scores/"selected_val100.jsonl",val100)
    need(held["n_tasks"]==zero_held["n_tasks"]==249 and
         {r["task"] for r in rows(scores/"selected_val100.jsonl")}==val100,"saved score cache coverage")
    payload = {"status":"complete","model_source":card["source"],"revision":card["revision"],
               "training_engine":{"packages":{k:version(k) for k in ("torch","transformers","safetensors")},
                                  "gpu":1,"sdpa":True,"seed":SEED,"microbatch":MICROBATCH,
                                  "pairs_per_optimizer_step":BATCH,"epoch":1,"learning_rate":LR,
                                  "schedule":"cosine","warmup_fraction":.03,"warmup_steps":warmup,
                                  "weight_decay":WEIGHT_DECAY,"optimizer":"AdamW fused","dtype":"BF16 autocast, FP32 master parameters",
                                  "trainable_parameters":params,"loss":"per-pair BCE on yes-minus-no logits",
                                  "last_step_real_pairs":len(order)-(steps[ratio]-1)*BATCH,
                                  "last_step_padding":"extra forward rows have zero objective weight"},
               "template":template,"negative_ratio":ratio,"n_train_pairs":seen,"n_optimizer_steps":steps[ratio],
               "n_truncated_train_pairs":desc["partitions"]["train"]["negative_ratios"][str(ratio)]["truncated"],
               "throughput_probe":read(directory/"throughput_smoke.json"),"fit_wall_seconds_including_checkpoints":fit_seconds,
               "peak_pytorch_gpu_allocation_bytes":peak,
               "zero_shot_validation":zero_val,"checkpoints":checkpoints,"chosen":chosen,
               "score_cache":{"selected_val100":val100_cache,"selected_heldout":held["file"],
                              "zero_shot_heldout":zero_held["file"]}}
    write_new(SMALL/"training.json",payload)
    print(f"S1 complete: checkpoint={chosen['step']} mAP={chosen['mAP']:.8f}",flush=True)


if __name__ == "__main__":
    run()

