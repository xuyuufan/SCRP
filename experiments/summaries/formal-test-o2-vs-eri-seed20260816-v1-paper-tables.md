# Phase 8 Formal Test Tables

Delta is RL minus ERI; negative values favor RL.

| Dataset | Algorithm | Mean relocations | Difference vs ERI | 95% CI |
|---|---|---:|---:|---:|
| DS1 | ERI | 8.914750 | 0 | [0, 0] |
| DS1 | RL O2 | 10.087917 | 1.173167 | [1.039160, 1.311917] |
| DS2 | ERI | 9.071750 | 0 | [0, 0] |
| DS2 | RL O2 | 9.914833 | 0.843083 | [0.757331, 0.930333] |

| Dataset | Mean delta (RL-ERI) | 95% CI | Wilcoxon p | Paired t p |
|---|---:|---:|---:|---:|
| DS1 | 1.173167 | [1.039160, 1.311917] | 5.86456e-26 | 2.16653e-20 |
| DS2 | 0.843083 | [0.757331, 0.930333] | 5.28067e-29 | 3.58994e-21 |
