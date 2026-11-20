#!/usr/bin/env python3
"""Prompt (prefix) caching benchmark for SGLang on Amazon SageMaker AI real-time endpoints.

Uses the AWS SGLang Deep Learning Container (SageMaker variant). Everything runs through the
SageMaker runtime API (InvokeEndpointWithResponseStream), so the numbers include the SageMaker
request path exactly as an application would see it.

Sub-commands
    deploy     Create a SageMaker model + endpoint config + endpoint (cache on or off).
    bench      Run the experiments against one endpoint and write results/<label>_*.csv.
    kvpool     Read the KV-pool size SGLang logs at startup (CloudWatch Logs).
    analyze    Build tables, figures and the cost / Bedrock break-even view from results/.
    cleanup    Delete every endpoint/config/model this script created (tag project=prompt-cache-blog).
    remote     Run `bench` inside a SageMaker Processing job in the endpoint's Region
               (keeps WAN latency out of TTFT when your laptop is far from the Region).

Design choices (each one maps to a review comment on the blog):
  * Prefix lengths are in TOKENS, built with the model's tokenizer, so a "2,048-token prefix" is
    exactly 2,048 tokens of shared prompt (BOS + system prompt + user-turn tag).
  * The shared prefix is the system prompt; the user query changes on every request.
  * A cold request is forced without /flush_cache (not reachable through SageMaker) by putting a
    random nonce at the start of the system prompt, so nothing in the radix tree matches.
  * We record TTFT *and* end-to-end latency (prefill + decode) for every request.
  * A closed-loop load test at several concurrency levels measures sustained throughput with the
    cache on and off, which is what drives instance-hour cost on SageMaker.
  * An eviction test cycles through more distinct long prefixes than the KV pool can hold.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import random
import re
import statistics
import sys
import threading
import time
import uuid

import boto3
from botocore.config import Config

# --------------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------------
REGION = os.environ.get("AWS_REGION") or boto3.session.Session().region_name or "us-east-1"
IMAGE_TAG = "server-sagemaker-cuda-v1.4.1"           # AWS SGLang DLC, SageMaker variant (pinned)
IMAGE_URI = f"763104351884.dkr.ecr.{REGION}.amazonaws.com/sglang:{IMAGE_TAG}"
INFERENCE_AMI = "al2023-ami-sagemaker-inference-gpu-4-1"  # required for the CUDA 13 DLC
INSTANCE_TYPE = "ml.g5.xlarge"                       # 1x NVIDIA A10G, 24 GB
INSTANCE_PRICE_PER_HOUR = 1.408                      # ml.g5.xlarge hosting, us-east-1, on-demand
TAG = {"Key": "project", "Value": "prompt-cache-blog"}

MODELS = {
    "1.5b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "7b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
}
MEM_FRACTION_STATIC = "0.85"
CONTEXT_LENGTH = "16384"

# Experiment knobs
SWEEP_PREFIX_TOKENS = [512, 1024, 2048, 4096, 8192]
SWEEP_TRIALS = 5            # independent prefixes per length (each gives 1 cold request)
SWEEP_WARM_PER_TRIAL = 4    # warm requests per trial, each with a different user query
SWEEP_MAX_TOKENS = 32       # short decode, so TTFT isolates prefill

LOAD_PREFIX_TOKENS = 2048
LOAD_NUM_PREFIXES = 4       # e.g. four agents / tenants, each with its own system prompt
LOAD_MAX_TOKENS = 128       # a realistic short answer / tool call
LOAD_CONCURRENCY = [1, 4, 8, 16, 32]
LOAD_REQUESTS_PER_WORKER = 6
LOAD_MIN_REQUESTS = 24

EVICT_PREFIX_TOKENS = 8192
EVICT_NUM_PREFIXES = [4, 6, 8, 10, 12, 16, 24]
EVICT_PASSES = 3

TOKENIZER_REPO = MODELS["1.5b"]  # 1.5B and 7B ship an identical tokenizer
BOS = "<｜begin▁of▁sentence｜>"
USER_TAG = "<｜User｜>"

RESULTS_DIR = os.environ.get("RESULTS_DIR", "results")

_cfg = Config(region_name=REGION, max_pool_connections=64,
              retries={"max_attempts": 2, "mode": "standard"}, read_timeout=120)


def sm_client():
    return boto3.client("sagemaker", config=_cfg)


def rt_client():
    return boto3.client("sagemaker-runtime", config=_cfg)


# --------------------------------------------------------------------------------------------
# Deployment
# --------------------------------------------------------------------------------------------
def endpoint_name(size: str, cache: bool, suffix: str) -> str:
    return f"pc-{size.replace('.', '')}-{'cache' if cache else 'nocache'}-{suffix}"


def deploy(size: str, cache: bool, role: str, suffix: str, wait: bool = True) -> str:
    """Create model, endpoint config and endpoint for one model with the cache on or off."""
    sm = sm_client()
    name = endpoint_name(size, cache, suffix)
    env = {
        "SM_SGLANG_MODEL_PATH": MODELS[size],
        "SM_SGLANG_MEM_FRACTION_STATIC": MEM_FRACTION_STATIC,
        "SM_SGLANG_CONTEXT_LENGTH": CONTEXT_LENGTH,
        # Report cached prompt tokens in usage.prompt_tokens_details.cached_tokens
        "SM_SGLANG_ENABLE_CACHE_REPORT": "true",
    }
    if not cache:
        env["SM_SGLANG_DISABLE_RADIX_CACHE"] = "true"   # negative control
    sm.create_model(ModelName=name, ExecutionRoleArn=role, Tags=[TAG],
                    PrimaryContainer={"Image": IMAGE_URI, "Environment": env})
    sm.create_endpoint_config(
        EndpointConfigName=name, Tags=[TAG],
        ProductionVariants=[{
            "VariantName": "AllTraffic", "ModelName": name,
            "InstanceType": INSTANCE_TYPE, "InitialInstanceCount": 1,
            "InferenceAmiVersion": INFERENCE_AMI,
            "ContainerStartupHealthCheckTimeoutInSeconds": 1200,
        }])
    sm.create_endpoint(EndpointName=name, EndpointConfigName=name, Tags=[TAG])
    print("creating", name, flush=True)
    if wait:
        sm.get_waiter("endpoint_in_service").wait(
            EndpointName=name, WaiterConfig={"Delay": 30, "MaxAttempts": 80})
        print("InService:", name, flush=True)
    return name


def cleanup(prefix: str = "pc-") -> None:
    """Delete endpoints, configs and models created by this script (name prefix + project tag)."""
    sm = sm_client()

    def tagged(arn):
        tags = sm.list_tags(ResourceArn=arn)["Tags"]
        return any(t["Key"] == TAG["Key"] and t["Value"] == TAG["Value"] for t in tags)

    for ep in sm.list_endpoints(NameContains=prefix, MaxResults=100)["Endpoints"]:
        if ep["EndpointName"].startswith(prefix) and tagged(ep["EndpointArn"]):
            print("delete endpoint", ep["EndpointName"]); sm.delete_endpoint(EndpointName=ep["EndpointName"])
    for c in sm.list_endpoint_configs(NameContains=prefix, MaxResults=100)["EndpointConfigs"]:
        if c["EndpointConfigName"].startswith(prefix) and tagged(c["EndpointConfigArn"]):
            print("delete config", c["EndpointConfigName"])
            sm.delete_endpoint_config(EndpointConfigName=c["EndpointConfigName"])
    for m in sm.list_models(NameContains=prefix, MaxResults=100)["Models"]:
        if m["ModelName"].startswith(prefix) and tagged(m["ModelArn"]):
            print("delete model", m["ModelName"]); sm.delete_model(ModelName=m["ModelName"])


# --------------------------------------------------------------------------------------------
# KV pool size (comment: "how much space was left for the KV cache?")
# --------------------------------------------------------------------------------------------
def kv_pool(endpoint: str) -> dict:
    logs = boto3.client("logs", config=_cfg)
    group = f"/aws/sagemaker/Endpoints/{endpoint}"
    out = {"endpoint": endpoint}
    pats = {
        "kv_tokens": r"KV Cache is allocated\..*?#tokens: (\d+)",
        "kv_k_gb": r"K size: ([\d.]+) GB",
        "kv_v_gb": r"V size: ([\d.]+) GB",
        "max_total_num_tokens": r"max_total_num_tokens=(\d+)",
        "max_running_requests": r"max_running_requests=(\d+)",
        "weights_gb": r"Load weight end\..*?mem usage=([\d.]+) GB",
        "avail_mem_gb": r"Load weight end\..*?avail mem=([\d.]+) GB",
        "radix_eviction_policy": r"radix_eviction_policy'?[=:] ?'(\w+)'",
        "disable_radix_cache": r"disable_radix_cache'?[=:] ?(\w+)",
        "page_size": r"'?page_size'?[=:] ?(\d+)",
        "attention_backend": r"'?attention_backend'?[=:] ?'(\w+)'",
    }
    for s in logs.describe_log_streams(logGroupName=group)["logStreams"]:
        kw = {"logGroupName": group, "logStreamName": s["logStreamName"], "startFromHead": True}
        while True:
            r = logs.get_log_events(**kw)
            for e in r["events"]:
                for k, p in pats.items():
                    if k not in out:
                        m = re.search(p, e["message"])
                        if m:
                            out[k] = m.group(1)
            if not r["events"] or r["nextForwardToken"] == kw.get("nextToken"):
                break
            kw["nextToken"] = r["nextForwardToken"]
    return out


# --------------------------------------------------------------------------------------------
# Token-exact prompt construction
# --------------------------------------------------------------------------------------------
_tok = None


def tokenizer():
    global _tok
    if _tok is None:
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer
        _tok = Tokenizer.from_file(hf_hub_download(TOKENIZER_REPO, "tokenizer.json"))
    return _tok


def _ntok(text: str) -> int:
    return len(tokenizer().encode(text, add_special_tokens=False).ids)


def _system_corpus(seed: int = 7) -> str:
    """A long, deterministic, agent-style system prompt: policy, tool schemas, few-shot examples."""
    rng = random.Random(seed)
    topics = ["refunds", "chargebacks", "KYC refresh", "card replacement", "loan prepayment",
              "address change", "fraud alerts", "fee reversal", "statement disputes", "wire limits",
              "account closure", "nominee update", "standing instructions", "FX conversion",
              "credit limit increase", "EMI conversion", "cheque book requests", "tax certificates"]
    verbs = ["verify", "confirm", "escalate", "log", "summarise", "cite", "reject", "approve"]
    parts = ["You are FinAssist, an enterprise customer-service agent for a retail bank. "
             "Follow every policy below exactly, call tools only with valid JSON arguments, "
             "cite the policy section you relied on, and never reveal internal notes.\n"]
    sec = 1
    while len(parts) < 1200:
        t = rng.choice(topics)
        parts.append(f"\n## Policy {sec}: {t.title()}\n")
        for i in range(1, 6):
            parts.append(f"{sec}.{i} Agents must {rng.choice(verbs)} the customer's request about {t} "
                         f"within {rng.randint(1, 72)} hours, check limit band {rng.choice('ABCDE')}"
                         f"{rng.randint(1, 9)}, and record reason code R{rng.randint(100, 999)}.\n")
        parts.append(json.dumps({"tool": f"{t.replace(' ', '_')}_lookup_{sec}",
                                 "description": f"Look up {t} records",
                                 "parameters": {"customer_id": "string", "case_id": "string",
                                                "amount": "number", "channel": "string"}}) + "\n")
        parts.append(f"Example {sec}: Q: Can you help with {t}? A: I will {rng.choice(verbs)} "
                     f"this under Policy {sec}.{rng.randint(1, 5)} and open case "
                     f"C-{rng.randint(10000, 99999)}.\n")
        sec += 1
    return "".join(parts)


_CORPUS_IDS = None


def build_system_prompt(prefix_tokens: int, nonce: str) -> str:
    """System prompt such that BOS + system + '<｜User｜>' is exactly `prefix_tokens` tokens."""
    global _CORPUS_IDS
    tok = tokenizer()
    if _CORPUS_IDS is None:
        _CORPUS_IDS = tok.encode(_system_corpus(), add_special_tokens=False).ids
    head = f"[deployment {nonce}]\n"
    budget = prefix_tokens - 2 - _ntok(head)          # 2 = BOS + <｜User｜>
    n = budget
    for _ in range(20):                               # adjust for decode/re-encode drift
        body = tok.decode(_CORPUS_IDS[:n])
        got = _ntok(head + body)
        if got == prefix_tokens - 2:
            return head + body
        n += (prefix_tokens - 2) - got
    return head + body


_Q_SUBJ = ["my credit card", "a wire transfer", "my home loan", "a disputed charge", "my savings account",
           "an FX payment", "my debit card", "a standing instruction", "my tax certificate", "an EMI"]
_Q_ASK = ["Why was {s} declined yesterday", "How do I raise the limit on {s}",
          "Can you reverse the fee on {s}", "What documents do I need for {s}",
          "How long will {s} take to process", "Is there a penalty for closing {s} early",
          "Who approves changes to {s}", "Can I split {s} into instalments"]


def make_query(rng: random.Random) -> str:
    s = rng.choice(_Q_SUBJ)
    return (rng.choice(_Q_ASK).format(s=s) + f"? My customer ID is {rng.randint(10**7, 10**8 - 1)} "
            f"and the reference is {uuid.UUID(int=rng.getrandbits(128)).hex[:12]}. Please answer briefly.")


# --------------------------------------------------------------------------------------------
# Streaming client
# --------------------------------------------------------------------------------------------
def invoke_stream(rt, endpoint: str, model_id: str, system: str, user: str, max_tokens: int,
                  attempts: int = 2) -> dict:
    """One streaming chat completion; returns TTFT, end-to-end latency and token usage.
    Failed requests are retried once and, if still failing, recorded with error set."""
    err = None
    for _ in range(attempts):
        try:
            return _invoke_stream(rt, endpoint, model_id, system, user, max_tokens)
        except Exception as e:  # noqa: BLE001 - keep the benchmark running, record the failure
            err = f"{type(e).__name__}: {str(e)[:200]}"
            print("  request failed:", err, flush=True)
    return {"ttft_ms": None, "e2e_ms": None, "prompt_tokens": None, "cached_tokens": None,
            "completion_tokens": None, "chunks": 0, "error": err}


def _invoke_stream(rt, endpoint, model_id, system, user, max_tokens):
    body = {
        "model": model_id,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "max_tokens": max_tokens, "temperature": 0.0,
        "ignore_eos": True,                      # fixed decode length for every request
        "stream": True, "stream_options": {"include_usage": True},
    }
    t0 = time.perf_counter()
    resp = rt.invoke_endpoint_with_response_stream(
        EndpointName=endpoint, ContentType="application/json", Body=json.dumps(body))
    ttft, usage, buf, n_chunks = None, {}, b"", 0
    for ev in resp["Body"]:
        part = ev.get("PayloadPart")
        if not part:
            continue
        buf += part["Bytes"]
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            line = line.strip()
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                continue
            obj = json.loads(data)
            ch = obj.get("choices") or []
            if ch:
                d = ch[0].get("delta") or {}
                if ttft is None and (d.get("content") or d.get("reasoning_content")):
                    ttft = time.perf_counter() - t0
                n_chunks += 1
            if obj.get("usage"):
                usage = obj["usage"]
    e2e = time.perf_counter() - t0
    details = usage.get("prompt_tokens_details") or {}
    return {"ttft_ms": round(ttft * 1000, 1) if ttft else None, "e2e_ms": round(e2e * 1000, 1),
            "prompt_tokens": usage.get("prompt_tokens"),
            "cached_tokens": details.get("cached_tokens", 0) or 0,
            "completion_tokens": usage.get("completion_tokens"), "chunks": n_chunks}


# --------------------------------------------------------------------------------------------
# Experiments
# --------------------------------------------------------------------------------------------
def exp_sweep(rt, ep, model_id, rng, out_rows):
    """Prefix-length sweep. Each trial: fresh prefix -> 1 cold request, then warm requests that
    reuse the prefix with a DIFFERENT user query each time."""
    for n in SWEEP_PREFIX_TOKENS:
        for trial in range(SWEEP_TRIALS):
            system = build_system_prompt(n, uuid.uuid4().hex[:16])
            for k in range(1 + SWEEP_WARM_PER_TRIAL):
                r = invoke_stream(rt, ep, model_id, system, make_query(rng), SWEEP_MAX_TOKENS)
                r.update(exp="sweep", prefix_tokens=n, trial=trial, phase="cold" if k == 0 else "warm")
                out_rows.append(r)
            print(f"  sweep {n:>5} trial {trial}: cold {out_rows[-1 - SWEEP_WARM_PER_TRIAL]['ttft_ms']} ms"
                  f" -> warm {out_rows[-1]['ttft_ms']} ms (cached {out_rows[-1]['cached_tokens']}"
                  f"/{out_rows[-1]['prompt_tokens']})", flush=True)


def exp_load(rt, ep, model_id, rng, out_rows):
    """Closed-loop load test: C workers send back-to-back requests. Each request picks one of
    LOAD_NUM_PREFIXES system prompts at random and a fresh user query."""
    prefixes = [build_system_prompt(LOAD_PREFIX_TOKENS, uuid.uuid4().hex[:16])
                for _ in range(LOAD_NUM_PREFIXES)]
    for p in prefixes:   # steady state: each prefix has been seen once (no-op when cache is off)
        invoke_stream(rt, ep, model_id, p, make_query(rng), 1)
    lock = threading.Lock()
    for c in LOAD_CONCURRENCY:
        total = max(LOAD_MIN_REQUESTS, c * LOAD_REQUESTS_PER_WORKER)
        jobs = [(rng.choice(prefixes), make_query(rng)) for _ in range(total)]
        it = iter(jobs)
        rows = []

        def worker():
            while True:
                with lock:
                    job = next(it, None)
                if job is None:
                    return
                r = invoke_stream(rt, ep, model_id, job[0], job[1], LOAD_MAX_TOKENS)
                with lock:
                    rows.append(r)

        t0 = time.perf_counter()
        with cf.ThreadPoolExecutor(c) as ex:
            for f in [ex.submit(worker) for _ in range(c)]:
                f.result()
        wall = time.perf_counter() - t0
        for r in rows:
            r.update(exp="load", concurrency=c, wall_s=round(wall, 2), prefix_tokens=LOAD_PREFIX_TOKENS)
        out_rows.extend(rows)
        ok = [r for r in rows if r["ttft_ms"] is not None]
        ttfts = sorted(r["ttft_ms"] for r in ok)
        print(f"  load c={c:>2}: {len(ok) / wall:.2f} req/s, TTFT p50 {ttfts[len(ttfts) // 2]} ms, "
              f"hit {sum(r['cached_tokens'] for r in ok) / sum(r['prompt_tokens'] for r in ok):.1%}, "
              f"errors {len(rows) - len(ok)}", flush=True)


def exp_evict(rt, ep, model_id, rng, out_rows):
    """Round-robin over N distinct long prefixes. Once N x prefix exceeds the KV pool, LRU evicts
    each prefix before it comes round again and the hit rate collapses."""
    for n_pref in EVICT_NUM_PREFIXES:
        prefixes = [build_system_prompt(EVICT_PREFIX_TOKENS, uuid.uuid4().hex[:16]) for _ in range(n_pref)]
        for p_i in range(EVICT_PASSES):
            for i, p in enumerate(prefixes):
                r = invoke_stream(rt, ep, model_id, p, make_query(rng), 1)
                r.update(exp="evict", num_prefixes=n_pref, pass_idx=p_i, prefix_idx=i,
                         prefix_tokens=EVICT_PREFIX_TOKENS)
                out_rows.append(r)
        later = [r for r in out_rows if r["exp"] == "evict" and r["num_prefixes"] == n_pref
                 and r["pass_idx"] > 0 and r["prompt_tokens"]]
        print(f"  evict N={n_pref:>2}: hit {sum(r['cached_tokens'] for r in later) / sum(r['prompt_tokens'] for r in later):.1%}",
              flush=True)


def bench(endpoint: str, size: str, label: str, experiments: list[str]) -> None:
    import csv
    os.makedirs(RESULTS_DIR, exist_ok=True)
    rt, model_id, rng = rt_client(), MODELS[size], random.Random(42)
    print(f"== {label}: {endpoint} ({model_id})", flush=True)
    for _ in range(3):   # warm up CUDA graphs / JIT so the first measured request is not penalised
        invoke_stream(rt, endpoint, model_id, build_system_prompt(256, uuid.uuid4().hex[:16]), make_query(rng), 8)
    for exp in experiments:
        rows: list[dict] = []
        {"sweep": exp_sweep, "load": exp_load, "evict": exp_evict}[exp](rt, endpoint, model_id, rng, rows)
        keys = sorted({k for r in rows for k in r})
        path = f"{RESULTS_DIR}/{label}_{exp}.csv"
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
        print("wrote", path, flush=True)


# --------------------------------------------------------------------------------------------
# Remote execution (SageMaker Processing job in the endpoint's Region)
# --------------------------------------------------------------------------------------------
def remote(role: str, bucket: str, jobs: list[str]) -> str:
    """Run `bench` for each 'endpoint:size:label:exp1,exp2' spec in one Processing job."""
    s3 = boto3.client("s3", config=_cfg)
    run = time.strftime("%Y%m%d-%H%M%S")
    key = f"prompt-cache-blog/{run}"
    s3.upload_file(os.path.abspath(__file__), bucket, f"{key}/code/run_benchmark.py")
    lines = ["set -e", "pip install -q tokenizers==0.22.2 huggingface_hub==1.11.0",
             "cd /opt/ml/processing/code", "OUT=/opt/ml/processing/output", "set +e"]
    for j in jobs:
        ep, size, label, exps = j.split(":")
        lines.append(f"RESULTS_DIR=$OUT python run_benchmark.py bench --endpoint {ep} --size {size} "
                     f"--label {label} --experiments {exps} 2>&1 | tee $OUT/{label}.log &")
    lines.append("wait")
    s3.put_object(Bucket=bucket, Key=f"{key}/code/run.sh", Body="\n".join(lines).encode())
    sm = sm_client()
    name = f"pc-client-{run}"
    sm.create_processing_job(
        ProcessingJobName=name, RoleArn=role, Tags=[TAG],
        AppSpecification={
            "ImageUri": f"683313688378.dkr.ecr.{REGION}.amazonaws.com/sagemaker-scikit-learn:1.4-2-py312-cpu-py3",
            "ContainerEntrypoint": ["bash", "/opt/ml/processing/code/run.sh"]},
        Environment={"AWS_REGION": REGION, "PYTHONUNBUFFERED": "1"},
        ProcessingResources={"ClusterConfig": {"InstanceCount": 1, "InstanceType": "ml.m5.xlarge",
                                               "VolumeSizeInGB": 20}},
        ProcessingInputs=[{"InputName": "code", "S3Input": {
            "S3Uri": f"s3://{bucket}/{key}/code/", "LocalPath": "/opt/ml/processing/code",
            "S3DataType": "S3Prefix", "S3InputMode": "File"}}],
        ProcessingOutputConfig={"Outputs": [{"OutputName": "results", "S3Output": {
            "S3Uri": f"s3://{bucket}/{key}/results/", "LocalPath": "/opt/ml/processing/output",
            "S3UploadMode": "EndOfJob"}}]},
        StoppingCondition={"MaxRuntimeInSeconds": 4 * 3600})
    print("processing job:", name, f"-> s3://{bucket}/{key}/results/", flush=True)
    return name


# --------------------------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------------------------
BEDROCK = {  # us-east-1 on-demand, USD per 1M tokens (Amazon Bedrock pricing page, at time of writing)
    "Amazon Nova Micro": {"input": 0.035, "cache_read": 0.00875, "cache_write": 0.0, "output": 0.14},
    "Amazon Nova Lite": {"input": 0.06, "cache_read": 0.015, "cache_write": 0.0, "output": 0.24},
}


def _p(xs, q):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    k = (len(xs) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def analyze(results_dir: str = RESULTS_DIR) -> None:
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    R = results_dir
    sizes = [s for s in MODELS if os.path.exists(f"{R}/{s}-cache_sweep.csv")]
    label = {"1.5b": "DeepSeek-R1-Distill-Qwen-1.5B", "7b": "DeepSeek-R1-Distill-Qwen-7B"}

    # ---- Table 1 / Figure 1-2: prefix sweep, cache on (and off as control)
    rows = []
    for s in sizes:
        for mode in ["cache", "nocache"]:
            f = f"{R}/{s}-{mode}_sweep.csv"
            if not os.path.exists(f):
                continue
            d = pd.read_csv(f); d = d[d.ttft_ms.notna()]
            for n, g in d.groupby("prefix_tokens"):
                c, w = g[g.phase == "cold"], g[g.phase == "warm"]
                rows.append({
                    "model": s, "cache": mode == "cache", "prefix_tokens": n,
                    "prompt_tokens_median": int(g.prompt_tokens.median()),
                    "ttft_cold_ms": round(c.ttft_ms.median(), 1), "ttft_warm_ms": round(w.ttft_ms.median(), 1),
                    "ttft_warm_p90_ms": round(_p(w.ttft_ms, .9), 1),
                    "e2e_cold_ms": round(c.e2e_ms.median(), 1), "e2e_warm_ms": round(w.e2e_ms.median(), 1),
                    "cached_tokens_warm_median": int(w.cached_tokens.median()),
                    "cached_tokens_cold_max": int(c.cached_tokens.max()),
                    "warm_hit_rate_pct": round(100 * w.cached_tokens.sum() / w.prompt_tokens.sum(), 1),
                    "n_cold": len(c), "n_warm": len(w)})
    sw = pd.DataFrame(rows)
    sw["ttft_reduction_pct"] = (100 * (1 - sw.ttft_warm_ms / sw.ttft_cold_ms)).round(1)
    sw["e2e_reduction_pct"] = (100 * (1 - sw.e2e_warm_ms / sw.e2e_cold_ms)).round(1)
    sw.to_csv(f"{R}/table_sweep.csv", index=False)

    on = sw[sw.cache]
    fig, ax = plt.subplots(figsize=(7, 4.6))
    for s in sizes:
        g = on[on.model == s]
        ax.plot(g.prefix_tokens, g.ttft_reduction_pct, marker="o", label=f"{label[s]} — TTFT")
        ax.plot(g.prefix_tokens, g.e2e_reduction_pct, marker="s", ls="--", label=f"{label[s]} — end-to-end")
    ax.set_xscale("log", base=2); ax.set_xticks(SWEEP_PREFIX_TOKENS); ax.set_xticklabels(SWEEP_PREFIX_TOKENS)
    ax.set_xlabel("Shared prefix (tokens)"); ax.set_ylabel("Latency reduction on cache hit (%)")
    ax.set_title("Cache-hit benefit vs. shared-prefix length (ml.g5.xlarge)"); ax.grid(alpha=.3)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)
    fig.tight_layout(); fig.savefig(f"{R}/fig1_reduction_vs_prefix.png", dpi=150)

    fig, axs = plt.subplots(1, len(sizes), figsize=(6 * len(sizes), 4), squeeze=False)
    for ax, s in zip(axs[0], sizes):
        g = on[on.model == s]
        x = range(len(g)); w = .38
        ax.bar([i - w / 2 for i in x], g.ttft_cold_ms, w, label="cold (miss)")
        ax.bar([i + w / 2 for i in x], g.ttft_warm_ms, w, label="warm (hit, new query)")
        ax.set_xticks(list(x)); ax.set_xticklabels(g.prefix_tokens)
        ax.set_xlabel("Shared prefix (tokens)"); ax.set_ylabel("Median TTFT (ms)"); ax.set_title(label[s])
        ax.legend()
    fig.tight_layout(); fig.savefig(f"{R}/fig2_ttft_cold_warm.png", dpi=150)

    # ---- Table 3 / Figure 3: throughput under load, cache on vs off
    lrows = []
    for s in sizes:
        for mode in ["cache", "nocache"]:
            f = f"{R}/{s}-{mode}_load.csv"
            if not os.path.exists(f):
                continue
            d = pd.read_csv(f); d = d[d.ttft_ms.notna()]
            for c, g in d.groupby("concurrency"):
                rps = len(g) / g.wall_s.iloc[0]
                lrows.append({
                    "model": s, "cache": mode == "cache", "concurrency": c, "requests": len(g),
                    "req_per_s": round(rps, 3),
                    "output_tok_per_s": round(g.completion_tokens.sum() / g.wall_s.iloc[0], 1),
                    "ttft_p50_ms": round(_p(g.ttft_ms, .5), 1), "ttft_p95_ms": round(_p(g.ttft_ms, .95), 1),
                    "e2e_p50_ms": round(_p(g.e2e_ms, .5), 1), "e2e_p95_ms": round(_p(g.e2e_ms, .95), 1),
                    "hit_rate_pct": round(100 * g.cached_tokens.sum() / g.prompt_tokens.sum(), 1),
                    "prompt_tokens_mean": round(g.prompt_tokens.mean(), 1),
                    "usd_per_1M_req_at_full_util": round(INSTANCE_PRICE_PER_HOUR / (rps * 3600) * 1e6, 2)})
    ld = pd.DataFrame(lrows)
    ld.to_csv(f"{R}/table_load.csv", index=False)
    if len(ld):
        fig, axs = plt.subplots(1, 2, figsize=(12, 4))
        for s in sizes:
            for mode, ls in [(True, "-"), (False, "--")]:
                g = ld[(ld.model == s) & (ld.cache == mode)]
                if len(g):
                    lab = f"{label[s]} — cache {'on' if mode else 'off'}"
                    axs[0].plot(g.concurrency, g.req_per_s, marker="o", ls=ls, label=lab)
                    axs[1].plot(g.concurrency, g.ttft_p95_ms, marker="o", ls=ls, label=lab)
        axs[0].set_ylabel("Sustained throughput (requests/s)"); axs[1].set_ylabel("TTFT p95 (ms)")
        for ax in axs:
            ax.set_xlabel("Concurrent clients"); ax.set_xticks(LOAD_CONCURRENCY); ax.grid(alpha=.3)
        axs[1].set_yscale("log")
        h, l = axs[0].get_legend_handles_labels()
        fig.legend(h, l, loc="lower center", ncol=4, fontsize=8)
        fig.suptitle(f"Closed-loop load on one ml.g5.xlarge ({LOAD_PREFIX_TOKENS}-token shared prefix, "
                     f"{LOAD_NUM_PREFIXES} prefixes, new query per request, {LOAD_MAX_TOKENS} output tokens)",
                     fontsize=9)
        fig.tight_layout(rect=(0, 0.07, 1, 1)); fig.savefig(f"{R}/fig3_throughput.png", dpi=150)

    # ---- Eviction
    erows = []
    for s in sizes:
        f = f"{R}/{s}-cache_evict.csv"
        if os.path.exists(f):
            d = pd.read_csv(f); d = d[d.ttft_ms.notna()]
            for n, g in d[d.pass_idx > 0].groupby("num_prefixes"):
                erows.append({"model": s, "num_prefixes": n, "prefix_tokens": int(g.prefix_tokens.iloc[0]),
                              "working_set_tokens": int(n * g.prefix_tokens.iloc[0]),
                              "hit_rate_pct": round(100 * g.cached_tokens.sum() / g.prompt_tokens.sum(), 1),
                              "ttft_p50_ms": round(_p(g.ttft_ms, .5), 1)})
    ev = pd.DataFrame(erows)
    ev.to_csv(f"{R}/table_evict.csv", index=False)
    kv = {}
    if os.path.exists(f"{R}/kv_pool.json"):
        kv = json.load(open(f"{R}/kv_pool.json"))
    if len(ev):
        fig, ax = plt.subplots(figsize=(7, 4))
        for s, g in ev.groupby("model"):
            ax.plot(g.working_set_tokens / 1000, g.hit_rate_pct, marker="o", label=label[s])
            pool = kv.get(f"{s}-cache", {}).get("kv_tokens")
            if pool:
                ax.axvline(int(pool) / 1000, ls=":", c="gray")
                ax.text(int(pool) / 1000 - (95 if int(pool) > 300000 else 0), 70, f" {label[s].split('-')[-1]} KV pool\n {int(pool):,} tokens", fontsize=8)
        ax.set_xlabel(f"Working set = distinct prefixes x {EVICT_PREFIX_TOKENS} tokens (thousands)")
        ax.set_ylabel("Prefix-cache hit rate (%)"); ax.set_ylim(0, 105); ax.grid(alpha=.3)
        ax.legend(loc="upper center", bbox_to_anchor=(0.62, 0.35), fontsize=8)
        ax.set_title("Hit rate when distinct prefixes exceed the KV pool (round-robin, LRU)")
        fig.tight_layout(); fig.savefig(f"{R}/fig4_eviction.png", dpi=150)

    # ---- Cost from measured throughput (instance-hours / sustained requests)
    TTFT_P95_SLO_MS = 1000
    crows = []
    for s in sizes:
        for mode in [True, False]:
            g = ld[(ld.model == s) & (ld.cache == mode)] if len(ld) else ld
            if not len(g):
                continue
            c1 = g[g.concurrency == 1].iloc[0]
            peak = g.sort_values("req_per_s").iloc[-1]
            slo = g[g.ttft_p95_ms <= TTFT_P95_SLO_MS]
            slo = slo.sort_values("req_per_s").iloc[-1] if len(slo) else None
            crows.append({
                "model": s, "cache": mode,
                "c1_req_per_s": c1.req_per_s, "c1_usd_per_1M_req": c1.usd_per_1M_req_at_full_util,
                "peak_concurrency": int(peak.concurrency), "peak_req_per_s": peak.req_per_s,
                "peak_usd_per_1M_req": peak.usd_per_1M_req_at_full_util, "peak_ttft_p95_ms": peak.ttft_p95_ms,
                "slo_concurrency": int(slo.concurrency) if slo is not None else None,
                "slo_req_per_s": slo.req_per_s if slo is not None else None,
                "slo_usd_per_1M_req": slo.usd_per_1M_req_at_full_util if slo is not None else None})
    cost = pd.DataFrame(crows)
    cost.to_csv(f"{R}/table_cost.csv", index=False)

    # ---- Bedrock break-even (modelled; not a like-for-like benchmark)
    brow = []
    if len(ld):
        # user query + chat-template tokens actually sent, measured from the load test
        q_tokens = int(round(ld.prompt_tokens_mean.mean() - LOAD_PREFIX_TOKENS))
        for s in sizes:
            g = ld[(ld.model == s) & ld.cache]
            if not len(g):
                continue
            best = g.sort_values("req_per_s").iloc[-1]
            monthly = INSTANCE_PRICE_PER_HOUR * 730
            cap = best.req_per_s * 3600 * 730
            for bm, p in BEDROCK.items():
                for hit in [0.0, 0.8, 0.95]:
                    per_req = (LOAD_PREFIX_TOKENS * (hit * p["cache_read"] + (1 - hit) * (p["input"] + p["cache_write"]))
                               + q_tokens * p["input"] + LOAD_MAX_TOKENS * p["output"]) / 1e6
                    brow.append({"self_hosted_model": s, "bedrock_model": bm, "bedrock_cache_hit": hit,
                                 "request_shape": f"{LOAD_PREFIX_TOKENS}+{q_tokens} in / {LOAD_MAX_TOKENS} out",
                                 "bedrock_usd_per_1M_req": round(per_req * 1e6, 2),
                                 "selfhost_usd_per_month_1_instance": round(monthly, 0),
                                 "selfhost_peak_req_per_s": best.req_per_s,
                                 "selfhost_usd_per_1M_req_at_peak": round(monthly / cap * 1e6, 2),
                                 "breakeven_req_per_month": int(monthly / per_req),
                                 "breakeven_req_per_s": round(monthly / per_req / (730 * 3600), 2),
                                 "breakeven_utilisation_pct": round(100 * (monthly / per_req) / cap, 1)})
    pd.DataFrame(brow).to_csv(f"{R}/table_bedrock_breakeven.csv", index=False)

    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(sw.to_string(), "\n", ld.to_string(), "\n", cost.to_string(), "\n", ev.to_string(), "\n",
          pd.DataFrame(brow).to_string())


# --------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("deploy"); d.add_argument("--size", choices=MODELS, required=True)
    d.add_argument("--cache", choices=["on", "off"], default="on"); d.add_argument("--role", required=True)
    d.add_argument("--suffix", default=uuid.uuid4().hex[:6]); d.add_argument("--no-wait", action="store_true")
    b = sub.add_parser("bench"); b.add_argument("--endpoint", required=True)
    b.add_argument("--size", choices=MODELS, required=True); b.add_argument("--label", required=True)
    b.add_argument("--experiments", default="sweep,load")
    k = sub.add_parser("kvpool"); k.add_argument("--endpoint", nargs="+", required=True)
    k.add_argument("--labels", nargs="+", required=True)
    sub.add_parser("analyze")
    c = sub.add_parser("cleanup"); c.add_argument("--prefix", default="pc-")
    r = sub.add_parser("remote"); r.add_argument("--role", required=True); r.add_argument("--bucket", required=True)
    r.add_argument("--job", action="append", required=True, help="endpoint:size:label:exp1,exp2")
    a = ap.parse_args()
    if a.cmd == "deploy":
        deploy(a.size, a.cache == "on", a.role, a.suffix, wait=not a.no_wait)
    elif a.cmd == "bench":
        bench(a.endpoint, a.size, a.label, a.experiments.split(","))
    elif a.cmd == "kvpool":
        os.makedirs(RESULTS_DIR, exist_ok=True)
        res = {lab: kv_pool(ep) for ep, lab in zip(a.endpoint, a.labels)}
        json.dump(res, open(f"{RESULTS_DIR}/kv_pool.json", "w"), indent=2); print(json.dumps(res, indent=2))
    elif a.cmd == "analyze":
        analyze()
    elif a.cmd == "cleanup":
        cleanup(a.prefix)
    elif a.cmd == "remote":
        remote(a.role, a.bucket, a.job)


if __name__ == "__main__":
    main()
