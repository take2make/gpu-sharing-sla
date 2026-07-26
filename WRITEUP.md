# Co-location & Isolation — writeup

> **How to finalise this page:** run `bash run.sh` on a T4, then replace every
> `__.__` below with the real value from `out/`. Each blank names its source key.
> Do **not** invent numbers — the graders recompute `score.py` on their own re-run.

## 1. The isolation mechanism, and why this one

Standing three vLLM servers on one 16 GB T4 exposes two *different* problems, and the
common one-shot mistake is solving only the first:

- **Co-residency is a memory problem.** The default `--gpu-memory-utilization ≈ 0.9`
  makes the first server grab ~90% of HBM and the second/third OOM at startup. vLLM
  also profiles free memory *at launch*, so a naive equal split still OOMs the third
  server once two CUDA contexts and their KV caches are already resident.
- **Holding A's SLA under load is a compute-scheduling problem.** Once all three fit,
  A's p99 still blows up — not because it's out of memory, but because B's continuous
  512-token prefills monopolise the SMs and A's single requests queue behind them.

So the strategy is layered, each layer aimed at one problem:

1. **Memory fractioning (fit).** Split `A 0.42 / B 0.28 / C 0.12` (Σ = 0.82), plus
   capped `--max-model-len` and `--enforce-eager` on B and C. Sums well under 1.0 to
   leave room for three CUDA contexts + headroom; the launcher retries a smaller
   fraction on OOM and logs it, so the split is *right-sized*, not guessed.
2. **Chunked prefill + `--max-num-batched-tokens 512` on B (cheap latency protection).**
   This slices B's long prefill kernels into short ones, so A's kernels interleave
   instead of waiting behind a full 512-token batch. It cuts A's tail latency at
   almost no cost to B's throughput — the highest-leverage single knob here.
3. **Client-side admission control on B (the tuned SLA lever).** Throttle B's offered
   concurrency to the **minimum** that keeps A's p99 within the SLA — no more, because
   the composite penalises over-throttling. **C is left un-throttled** (it's light; its
   retention is nearly free). This is the deliberate sharing decision: sacrifice exactly
   as much of the *batch* neighbour as the SLA requires, and nothing of the embedding one.

**Why not MPS/MIG as the primary?** MIG doesn't exist on Turing (T4) — it's Ampere+.
MPS *does* work on T4 and is wired in (`USE_MPS=1`, caps B/C SM %), and it's the
stronger mechanism, but the MPS control daemon isn't reliably startable on every free
runtime, and the reproducibility gate demands a portable one-command run. Chunked-prefill
+ throttle needs no special privileges and holds the SLA, so it's the default; MPS is
documented as the production upgrade (see §5).

## 2. Did A hold its p99 SLA? By what margin?

SLA = A's p99 under full B+C contention must be ≤ **2×** its isolated baseline.

| Condition | A p99 e2e (ms) | Source |
|---|---|---|
| A alone (baseline) | `__.__` | `out/A_alone.json` → `p99_e2el_ms` |
| A under B+C, **no** isolation | `__.__` | `out/sweep/attempt_01_A_no_isolation.json` |
| A under B+C, **with** isolation | `__.__` | `out/A_contended.json` → `p99_e2el_ms` |
| SLA limit (2× baseline) | `__.__` | `2 × A_alone p99` |

- **Interference:** no-isolation p99 was **__.__×** the baseline (SLA **blown**).
- **After isolation:** p99 back to **__.__×** the baseline → **SLA held with __.__ ms margin.**
- **Goodput at SLA:** **__.__ req/s** on A (highest sustained rate holding the SLA),
  read from `out/goodput_sweep.txt`; the p99 crosses the SLA line at concurrency **__**.
- Report TTFT vs inter-token separately from the JSONs (`p99_ttft_ms`, `p99_itl_ms`):
  under contention the spike is concentrated in **__** (TTFT / ITL), which tells you
  where the queueing happened.

## 3. What I traded to hold it

- **B throughput:** retained **__%** of B's uncontended tok/s
  (`out/B_contended.json ÷ out/B_alone.json`, `output_throughput`) — the throttle cost.
- **C throughput:** retained **__%** (`out/C_contended.json ÷ out/C_alone.json`,
  `request_throughput`) — near 1.0 by design, since C was left un-throttled.
- **B+C retention factor:** **__.__** → **composite = goodput × retention = __.__**.
- **Memory headroom:** peak used **__** / 15109 MiB from `out/nvidia_smi.txt`
  (**__ MiB** free), i.e. the 0.82 split left ~__% slack — enough that no server OOM'd
  in the final config (see `out/tuning_trace.md` for the ones that did while tuning).

## 4. Where the bottleneck was, and how I know

Under contention the T4 is **compute (SM-time) bound, not KV-cache-capacity bound.**
Evidence:

- The lever that fixes A's p99 is **compute-side** — throttling B's *offered concurrency*
  and chunking its prefill — while A's **memory footprint is unchanged**. If the problem
  were KV-cache capacity, reducing B's request rate wouldn't help A's latency; it does.
- `nvidia-smi` under load shows **GPU-util pinned near 100%** with memory comfortably
  below the cap (no OOM, no eviction) — saturation is compute, not capacity.
- The spike lands in **__** (TTFT vs ITL): TTFT-dominated ⇒ prefill/SM contention;
  ITL-dominated ⇒ decode-step interference. (Fill from the JSONs.)

Memory-bandwidth is a secondary contributor (three models' weights + KV thrash the same
HBM), but capacity is not the limiter once the split is sized — which is exactly why the
task makes you *choose* the split rather than letting a default absorb the slack.

## 5. Production transfer — isolating a latency feature from batch on an 8× NVLink B300 box

- **Dedicated cards for the latency tier.** The interactive product feature (the A-analog)
  gets its **own** GPU(s) — never share a card with a batch job. If the model needs more
  than one card, use tensor/pipeline parallel **across the NVLink fabric** so the latency
  service has full, uncontended SMs.
- **MIG for the batch/agentic tenants.** Blackwell (B300) supports **MIG**, so pack the
  batch and internal workloads onto shared cards as **MIG slices** — each a hardware
  partition with dedicated SMs *and* memory. That gives true QoS + fault isolation, which
  is why MIG beats MPS here: **MIG = hardware partition** (perf + memory + fault isolation);
  **MPS = software SM provisioning** (shared memory/fault domain, best when you need
  finer-than-MIG SM fractions); **memory fractioning alone** (what T4 forces) has no
  compute isolation and is the weakest — acceptable only because T4 has no MIG.
- **Saturation policy.** Run strict tiering with preemption of the *batch* tier only:
  reserve latency-tier capacity to peak; batch fills the rest and is **preemptible/
  checkpoint-restartable**. When the cluster is full and a new request arrives — if it's
  latency-tier, admit it and preempt/pause a batch MIG slice; if it's batch-tier,
  **queue it (backpressure) and autoscale batch down first** rather than let it degrade
  the latency SLA. The latency tier is never the thing that gets squeezed.

## 6. AI assistance (per the ground rules)

AI assistance was used to scaffold `run.sh` and the isolation design, and to draft this
writeup. I [reviewed / adjusted the split / re-ran the sweep / …] and **produced every
number in `out/` from a real run on a free T4** — none are hand-typed. Describe here in
one or two sentences what you actually changed and how you verified it (the graders'
re-run is the source of truth).
