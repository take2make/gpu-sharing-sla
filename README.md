# GPU Co-location & Isolation

Three workloads co-resident on one 16 GB T4, with an isolation strategy that holds the
interactive workload's latency SLA under contention:

- **A** — `unsloth/Llama-3.2-1B-Instruct` — latency-sensitive, must hold p99 SLA
- **B** — `Qwen/Qwen2.5-0.5B-Instruct` — batch/throughput noisy neighbor
- **C** — `BAAI/bge-small-en-v1.5` — embedding load

**SLA:** A's p99 e2e under full B+C contention stays within 2× its isolated baseline.
**Isolation:** GPU memory fractioning (0.41/0.31/0.09) + concurrency throttling on B/C.

## Run (Colab/Kaggle T4)

```python
!git clone --depth 1 https://github.com/take2make/gpu-sharing-sla && cd gpu-sharing-sla && bash run.sh
```

`run.sh` installs pinned deps, launches all three servers co-resident, prints the
`nvidia-smi` co-residency dump, runs the benchmarks into `out/`, and computes the
composite score.

## Results

| Metric | Value |
|---|---|
| A p99 alone | ~3161 ms |
| A p99 under contention | ~4146 ms (1.32×) — SLA held |
| A goodput at SLA | ~4.244 req/s |
| B / C retention | 1.00 / 1.00 |
| Composite | ~4.2373 |

See `out/tuning_trace.md` for the tuning trail and `WRITEUP.md` for analysis.