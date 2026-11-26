"""Acquire the three registered Task 20 assets on 5880 with hash-checked curl transfers."""
import argparse
from hashlib import sha256
from html.parser import HTMLParser
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import time
from urllib.parse import parse_qs, urlencode, urljoin, urlsplit
import zipfile

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from src.reader_heldout_inputs import EXTERNAL, checksum, need, read, verify, write_new

WORK = EXTERNAL/"agent_v1"
SMALL = PROJECT/"eval/agent_v1"
GRPO = EXTERNAL/"envs/grpo/bin/python"
VENDOR = EXTERNAL/"vendor"
MODELS = {
    "llmlingua2": ("microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
                  "50f9b93891f8e7d4e578d89a821325f93f4b89bd", "llmlingua-2-xlm-roberta-large-meetingbank",
                  ("README.md","configuration.json","config.json","tokenizer_config.json","special_tokens_map.json","tokenizer.json","model.safetensors")),
    "open_provence": ("hotchpotch/open-provence-reranker-v1-gte-modernbert-base",
                      "539126c60a0f858d5329d7ccfce5966075f874ee", "open-provence-en-gte",
                      ("README.md","config.json","configuration.json","modeling_open_provence_standalone.py",
                       "tokenizer_config.json","special_tokens_map.json","tokenizer.json","model.safetensors")),
}
ALIASES = ("https://mirrors.aliyun.com/pypi/simple/llmlingua/", "https://pypi.tuna.tsinghua.edu.cn/simple/llmlingua/")
CURL = ("/usr/bin/curl","--noproxy","*","-fsSL","--connect-timeout","25")


class WheelLinks(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, items):
        attrs = dict(items)
        if tag == "a" and "href" in attrs and "data-yanked" not in attrs:
            self.links.append(attrs["href"])


def get(url):
    return subprocess.check_output([*CURL,"--max-time","90",url])


def fetch(url, path, digest, n=None):
    path.parent.mkdir(exist_ok=True, parents=True)
    need(not os.path.lexists(path) and path.resolve()==path, f"existing target: {path}")
    part=path.with_name(path.name+".partial")
    need(part.resolve()==part, f"redirected partial: {part}")
    if not part.exists():
        with part.open("xb"):
            pass
    if n is None or part.stat().st_size < n:
        subprocess.run([*CURL,"-C","-","--retry","1000","--retry-delay","60","--retry-all-errors",
                        "--speed-limit","1000","--speed-time","120","-o",str(part),url],check=True)
    need(n is None or n==part.stat().st_size,f"wrong content length: {part}")
    verify(part,digest)
    part.rename(path)
    return {"path":str(path),"bytes":path.stat().st_size,"sha256":digest}


def wheel():
    chosen = None
    for url in ALIASES:
        try:
            parser = WheelLinks()
            parser.feed(get(url).decode())
            options=[]
            for href in parser.links:
                p=urlsplit(urljoin(url,href))
                match=re.fullmatch(r"llmlingua-(\d+(?:\.\d+)*)-py3-none-any\.whl",p.path.rsplit("/",1)[-1])
                sign=parse_qs(p.fragment).get("sha256",[])
                if match and len(sign)==1:
                    options.append((tuple(map(int,match[1].split("."))),p,sign[0],match[1]))
            if options:
                chosen=max(options,key=lambda x:x[0])
                break
        except (OSError, subprocess.CalledProcessError):
            continue
    need(chosen is not None,"NEED: LLMLingua wheel unavailable from Aliyun or TUNA")
    _, parsed, digest, version = chosen
    path=WORK/"acquisition"/parsed.path.rsplit("/",1)[-1]
    transfer=fetch(parsed._replace(fragment="").geturl(),path,digest)
    unpack=VENDOR/f"llmlingua-{version}"
    unpack.mkdir(parents=True)
    extracted=[]
    with zipfile.ZipFile(path) as z:
        for f in z.infolist():
            rel=PurePosixPath(f.filename)
            need(rel.parts and not rel.is_absolute() and not {"..",".git"}.intersection(rel.parts)
                 and (f.external_attr>>16)&0o170000!=0o120000,"unsafe wheel entry")
            if not f.is_dir():
                target=unpack.joinpath(*rel.parts)
                target.parent.mkdir(parents=True,exist_ok=True)
                data=z.read(f)
                with target.open("xb") as out:
                    out.write(data)
                extracted.append({"path":str(target),"bytes":len(data),"sha256":sha256(data).hexdigest()})
    env=dict(os.environ,CUDA_VISIBLE_DEVICES="",PYTHONDONTWRITEBYTECODE="1",HF_HUB_OFFLINE="1",
             TRANSFORMERS_OFFLINE="1",PYTHONPATH=str(unpack),NLTK_DATA=str(EXTERNAL/"nltk_data"))
    check = "import importlib.metadata as m; from llmlingua import PromptCompressor; print(m.version('llmlingua'))"
    acquired=subprocess.run([str(GRPO),"-B","-c",check],cwd=PROJECT,env=env,text=True,capture_output=True)
    if acquired.returncode:
        print(acquired.stderr,file=sys.stderr)
        raise RuntimeError("NEED: importing LLMLingua wheel failed in the existing GRPO environment")
    need(acquired.stdout.strip().splitlines()[-1]==version,"loaded a different LLMLingua version")
    write_new(WORK/"acquisition/wheel.json",{"status":"ready","revision":version,"license":"MIT",
               "source":url,"vendor":str(unpack),"wheel":transfer,"files":extracted})
    print("LLMLingua",version,"CPU import passed",flush=True)


def model(key):
    slug, revision, dirname, names=MODELS[key]
    api=f"https://modelscope.cn/api/v1/models/{slug}"
    tree_url=api+"/repo/files?"+urlencode({"Revision":revision,"Recursive":"true"})
    metadata=json.loads(get(tree_url))
    need(metadata["Code"]==200,f"ModelScope revision unavailable: {slug}")
    files={f["Path"]:f for f in metadata["Data"]["Files"] if f["Type"]=="blob"}
    target=EXTERNAL/"models"/dirname
    receipt=WORK/f"acquisition/{key}.json"
    need(not receipt.exists() and target.resolve()==target, "model receipt already exists or destination redirected")
    target.mkdir(exist_ok=True)
    selected=[]
    for name in names:
        item=files[name]
        path=target/name
        url=api+"/repo?"+urlencode({"Revision":revision,"FilePath":name})
        if path.exists():
            need(path.resolve()==path and path.stat().st_size==item["Size"],f"existing file differs: {path}")
            verify(path,item["Sha256"])
            selected.append({"path":str(path),"bytes":item["Size"],"sha256":item["Sha256"]})
        else:
            print("fetch",key,name,item["Size"],flush=True)
            selected.append(fetch(url,path,item["Sha256"],item["Size"]))
    write_new(receipt,{"status":"ready","model":slug,"revision":revision,"license":"MIT","repository_api":tree_url,
                       "directory":str(target),"files":selected,
                       "description":"third-party open-source Provence reproduction" if key=="open_provence" else "LLMLingua-2"})
    print("ready",key,revision,flush=True)


def launcher():
    need(os.environ.get("CUDA_VISIBLE_DEVICES")=="","downloads must hide CUDA")
    a=WORK/"acquisition"
    a.mkdir()
    children={}
    for job in ("wheel",*MODELS):
        with (a/f"{job}.log").open("xb") as log:
            proc=subprocess.Popen([sys.executable,"-B","-u",str(Path(__file__).resolve()),job],cwd=PROJECT,
                 env=dict(os.environ,CUDA_VISIBLE_DEVICES="",PYTHONDONTWRITEBYTECODE="1"),stdin=subprocess.DEVNULL,
                 stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
        children[job]=proc.pid
    write_new(a/"pids.json",children)
    print(children)


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("job",choices=("launch","wheel",*MODELS))
    args=cli.parse_args()
    (launcher if args.job=="launch" else wheel if args.job=="wheel" else lambda:model(args.job))()
