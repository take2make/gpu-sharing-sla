0.45/0.35/0.2 → C model OOM'd at startup (sum>0.9 + 3 CUDA ctxs).

0.4/0.3/0.1 → all three fit, A p99 1.2× but C had already finished its load and
freed the GPU during A bench ran, so the contention was understated.
Added a wait-for-GPU-saturation loop and raised --num-prompts so B and C keep
loading the GPU for the whole of A run.

0.4/0.3/0.1 → A p99 1.4×, B retained 0.57. SLA held with margin to spare, so I
loosened B throttle to recover its throughput.

0.4/0.33/0.08, A p99 1.1×, B+C retained 0.55

0.45/0.3/0.08 → A p99 1.35×, B retained 0.57, C retained 0.69.
SLA held with margin; B retention is low → give B more concurrency
to raise its throughput.

0.41/0.31/0.09, conc A12/B12/C8 → A p99 1.51×, B and C retained 0.67

0.41/0.31/0.09, conc A14/B12/C10 → A p99 1.43×, B retained 0.76, C retained 0.79,
SLA held with margin → room to raise A's concurrency for more goodput.

0.41/0.31/0.09, conc A15/B13/C11 → A p99 1.535×, B retained 0.74, C retained 0.81,
Balanced operating point: A holds the SLA with margin, B and C both productive.

0.41/0.31/0.09, conc A16/B14/C12 → A p99 1.13×, B retained 0.5, C retained 1.0,
A warm-up was polluting the baseline → increased --num-warmups and
re-measured. B retention dropped (0.5) at B14 — pushed B too hard relative to its share.

0.41/0.31/0.09 → tested MPS, but it degraded the baseline results (little improvement in B), so I kept the concurrency-based isolation.