# Phase 9 Post-test Analysis and Development-only Diagnostics

## Scope and safeguards

This analysis uses only committed Phase 7B training/validation summaries and
new fixed-subset train/validation diagnostics. It does not read Phase 8 raw
test rows, does not evaluate the test split, and does not train or save a new
formal model. The frozen Phase 8 conclusion is referenced only as the question
to explain: the episode-15000 O2 policy did not outperform ERI.

The development diagnostic uses seed 20260816 and one deterministic train and
validation base layout from each of the 48 parameter groups. The O1/O2
ablation uses the same 400-episode training schedule for both observations and
one validation scenario for each DS1/DS2 static variant. These small-sample
results are diagnostic, not a new paper-level comparison.

## 1. Training and validation curve audit

| Episode | DS1 validation | DS2 validation | Selection score | FGB version |
|---:|---:|---:|---:|---:|
| 2,500 | 10.7073 | 10.5260 | 10.6167 | 9 |
| 5,000 | 10.6483 | 10.4546 | 10.5515 | 12 |
| 7,500 | 10.6969 | 10.4779 | 10.5874 | 17 |
| 10,000 | 10.6115 | 10.3144 | 10.4629 | 18 |
| 12,500 | 10.5190 | 10.3604 | 10.4397 | 18 |
| 15,000 | 10.3906 | 10.1935 | **10.2921** | 19 |
| 17,500 | 10.4338 | 10.4075 | 10.4206 | 19 |
| 20,000 | 10.5731 | 10.4258 | 10.4995 | 21 |
| 22,500 | 10.3942 | 10.2910 | 10.3426 | 25 |
| 25,000 | 10.4548 | 10.2406 | 10.3477 | 27 |

Episode 15,000 is the genuine minimum on the frozen validation trajectory.
The score degrades by 0.1285 at 17,500 and 0.2074 at 20,000, then only partly
recovers. More training therefore did not solve the gap.

The 27 baseline refreshes are highly nonuniform: 18 occur by episode 8,160,
none occur from 8,161 through 14,123, and eight occur from 17,968 onward. Every
refresh decision is based on only four paired episodes. This n=4 test produces
discrete, noisy evidence and plausibly makes the learning target move too
quickly early and late while remaining stale in the middle.

There is no monotonic entropy collapse: compact-window entropy averages 1.128
in the first 2,500 episodes and 0.922 in the last 2,500, with only one of 250
windows below 0.5. In contrast, every compact window has mean pre-clip gradient
norm above the configured 0.5 threshold (overall mean 1.665), showing persistent
clipping. The fixed validation probe estimates decision-advantage variance at
44.684 across 994 decisions. The original compact trace contains only window
mean advantages, so it cannot retrospectively recover the full formal-run
advantage variance.

Sampling is not the main explanation: S-bucket coefficient of variation is
2.68%, the largest/smallest bucket ratio is 1.092, and DS1/DS2 counts are
12,616/12,384.

## 2. O1 versus O2 development ablation

| Observation | Train episodes | Validation episodes | Validation mean | DS1 | DS2 |
|---|---:|---:|---:|---:|---:|
| O1 | 400 | 96 | 14.6875 | 14.3542 | 15.0208 |
| O2 | 400 | 96 | 11.3958 | 11.2083 | 11.5833 |

O2 is better by 3.2917 relocations under this fixed small budget. Thus the
complete O2 representation is useful relative to O1. However, the two adapters
differ in more than just their order nodes, so this ablation alone cannot prove
that the trained policy uses the specific future revealed-order sequence.

## 3. ERI action imitation and error states

The episode-15000 policy was compared with ERI on 1,746 ERI-guided public
decision states from 192 train/validation episodes. The RL action was observed
but ERI controlled the trajectory, avoiding policy-dependent state-selection
bias between comparison groups.

- Exact RL/ERI action agreement: 29.67%.
- Mean ERI-score penalty of the RL action: 0.0576.
- ERI-score-equivalent action rate: 90.91% to 93.72% across split/dataset cells.
- Strictly worse ERI-score action rate: 6.28% to 9.09%.

The low exact agreement therefore mostly reflects alternative actions tied
under ERI, not 70% substantive errors. The smaller 6-9% set of strictly worse
public-precedence choices can nevertheless accumulate across an episode.

Notable development groups are:

- S=9 and S=10 exact agreement is only 24.18% and 22.40%.
- S=8 is similarly low at 22.03%.
- Fill 0.67 has a larger mean ERI-score penalty (0.0712) than fill 0.50
  (0.0338), even though exact agreement is higher.
- T=5/6 have larger score penalties (0.0637/0.0763) than T=3/4.
- DS2 batch size 5 has the largest batch-stratified score penalty, 0.0955.
- The common target-tier-0, one-blocker state has only 22.44% exact agreement;
  deeper/more-blocked cells are rarer and have larger but less stable penalties.

These patterns support targeted development diagnostics for high fill, large
T, large S, and larger DS2 batches, but the fixed subset is too small for a
paper-level subgroup claim.

## 4. Representation usage

Across 627 states with at least two future revealed containers:

- controlled future-order permutation mean total variation: 0.003995;
- greedy action change rate under permutation: 0.319%;
- mean stack-embedding RMS change: 0.004467;
- masking all order nodes mean total variation: 0.023617;
- greedy action change rate under order-node ablation: 0.957%;
- maximum effect of perturbing masked padding nodes: exactly 0.

The mask behaves correctly, but the formal O2 policy is almost invariant to
the specific future order. This reconciles the two observations above: the O2
feature system helps relative to O1, while the model makes little practical
use of the lossless revealed-order sequence that motivated O2.

## 5. Why the current RL policy trails ERI

The train/validation evidence supports four interacting causes:

1. The policy slightly beats its own frozen learned baseline on the fixed
   validation probe (mean policy-minus-FGB relocations -0.0938), but FGB is a
   self-referential target and supplies no direct pressure toward ERI-quality
   public-precedence decisions.
2. Specific future-order information barely changes logits or greedy actions,
   so the current mean-pooled Transformer/pointer path does not exploit O2's
   main informational advantage.
3. A small but persistent 6-9% of public states select a destination with a
   strictly worse ERI score, especially in difficult fill/T/batch regimes.
4. n=4 FGB refresh tests, high advantage dispersion, and persistent gradient
   clipping create an unstable optimization target. The post-15k validation
   degradation shows that simply extending the same run is unlikely to fix it.

## Recommendation: D — structural model/training modification

If exceeding ERI is a research goal, another unchanged or merely longer run is
not justified. A new follow-up experiment should be development-only and
pre-register the following train/validation-supported changes:

1. Add an explicit order-aware sequence/cross-attention path from the blocker
   query to revealed-order nodes instead of relying primarily on symmetric
   global pooling.
2. Add a public-state ERI imitation or score-ranking auxiliary objective on
   train states, treating all minimum-ERI destinations as equivalent targets
   rather than imitating only ERI's deterministic tie-break.
3. Replace n=4 immediate FGB refresh decisions with a predeclared larger paired
   development buffer/cadence and audit unclipped-gradient frequency.
4. Select the follow-up only by fixed validation relocation score, ERI-score
   error rate, order-sensitivity probes, and stability metrics.

This must be named a separate follow-up experiment. Phase 8 remains the final
test result for the current model, and the existing test split can no longer
be described as untouched for any future model comparison. If no structural
follow-up is desired, the scientifically valid alternative is Recommendation
A: publish the frozen result and report that the current O2 RL policy did not
surpass ERI.

No parameter choice in this recommendation was selected from formal-test raw
rows or by rerunning the formal test.
