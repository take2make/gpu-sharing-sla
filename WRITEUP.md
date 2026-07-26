# Co-location & Isolation

## How to run

**Option 1 — Colab/Kaggle notebook cell**:

```python
!git clone --depth 1 https://github.com/take2make/gpu-sharing-sla && cd gpu-sharing-sla && bash run.sh
```

**Option 2 — headless via [google-colab-cli](https://github.com/googlecolab/google-colab-cli)**:

```bash
colab run --gpu t4 --timeout 3600 run.sh
```

## The isolation mechanism

Standing three vLLM servers on one 16 GB T4 exposes two different problems. The
first is locating all three models without OOM. The second is achieving a plausible
latency for the A model under contention. The first problem is easily solved by
choosing a different `--gpu-memory-utilization` for each model. The second is more
complex, because all three servers share the same SMs -> memory splitting doesn't help,
so I cap B's and C's request concurrency, and use MPS to cap each server's SM
share, keeping their kernels from crowding out A. It allows to me prioritize requests to A even under contention of B+C.

A's p99 e2e latency:
- **alone:** 2739.452 ms
- **under full B+C contention, after isolation:** 3900.369 ms → **1.4×** baseline

The SLA limit is 2× baseline, so A held it with margin to spare (1.4× vs 2.0×).

**Goodput:** A sustained **__.__ req/s** while holding the SLA with B and C loaded
(highest rate under the SLA from the sweep in `out/goodput_sweep.txt`).

## B model is a bottleneck

- **B throughput:** retained **70%** of B's uncontended tok/s, we pay the throttcle cost of B to make A throuput better.
- **C throughput:** retained **95%** — near 1.0 by design, since C was left un-throttled.
- **B+C retention factor:** **__.__** → **composite = goodput × retention = __.__**.
- **Memory headroom:** peak used **12685 MiB** / 15109 MiB from `out/nvidia_smi.txt`
  (**2424 MiB** free), i.e. the 0.84 split left ~16% slack — enough that no server OOM'd
  in the final config (see `out/tuning_trace.md` for the ones that did while tuning).

## The bottleneck on T4 GPU

Under contention the T4 is **compute (SM-time) bound, not KV-cache-capacity bound.**
Evidence:

- The lever that fixes A's p99 is **compute-side** — throttling B's *offered concurrency*
 — while A's **memory footprint is unchanged**. If the problem were KV-cache capacity, reducing B's request rate wouldn't help A's latency; it does.
- `nvidia-smi` under load shows **GPU-util pinned near 100%** with memory comfortably
  below the cap (no OOM, no eviction) — saturation is compute, not capacity.
- The spike lands in **__** (TTFT vs ITL): TTFT-dominated ⇒ prefill/SM contention;
  ITL-dominated ⇒ decode-step interference. (Fill from the JSONs.)

## Production transfer — isolating a latency feature from batch on an 8× NVLink B300 box

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
