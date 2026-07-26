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

- **B throughput:** retained **70%** of B's uncontended tok/s — the throttle cost we
  pay on B to protect A's latency.
- **Memory headroom:** peak **12685 MiB / 15109 MiB** used (from `out/nvidia_smi.txt`),
  **2424 MiB free** — the 0.84 split left ~16% slack, enough that no server OOM'd in the
  final config.

## The bottleneck on T4 GPU

Bottleneck = SM/compute contention, not memory. A had spare KV-cache. Throttling B concurrency is what recovered A p99.

## Production transfer — isolating a latency feature from batch on an 8× NVLink B300 box

- **Dedicated cards for the latency tier.** The interactive product feature (the A-analog)
  gets its **own** GPU(s) — never share a card with a batch job. If the model needs more
  than one card, use tensor/pipeline parallel **across the NVLink fabric** so the latency
  service has full, uncontended SMs.
- **MIG for the batch/agentic tenants.** Blackwell (B300) supports **MIG**, so pack the
  batch and internal workloads onto shared cards as **MIG slices** — each a hardware
  partition with its own dedicated SMs and memory, so one tenant can't touch another's
  compute. This is the strongest isolation. The three options, strongest to weakest:
  - **MIG** — a true hardware partition (dedicated SMs + memory). Best when a tenant
    needs a guaranteed SLA that no neighbor can disturb.
  - **MPS** — software-level SM sharing (caps each client's SM %, but shared memory and
    fault domain). Use it when you need finer SM fractions than MIG's fixed slice sizes.
  - **Memory fractioning alone** splits memory only, no compute isolation.
- **Saturation policy.** The latency tier gets reserved capacity that batch can never
  take. When the cluster is full and a new request arrives: if it's latency-tier, admit
  it and pause a batch job to make room; if it's batch-tier, queue it and wait. Batch is
  the shock absorber — it gets paused or delayed first, so the latency SLA is never the
  thing that suffers.
