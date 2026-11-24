# Retrieval benchmark: squad

2067 queries over 2067 documents (2925 chunks). Seed 42. Run 2026-11-24T04:19:03+00:00 on Windows AMD64, Python 3.11.4.

Means with 95% bootstrap confidence intervals.

| Config | recall@1 | recall@5 | recall@10 | mrr@10 | ndcg@10 | answer_in_top5 | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|---|
| bm25 | 0.780 [0.761, 0.797] | 0.926 [0.915, 0.937] | 0.948 [0.939, 0.957] | 0.843 [0.830, 0.856] | 0.869 [0.857, 0.880] | 0.941 [0.931, 0.951] | 9.94 | 20.31 |
| dense | 0.684 [0.664, 0.703] | 0.895 [0.881, 0.908] | 0.940 [0.929, 0.950] | 0.773 [0.757, 0.788] | 0.814 [0.800, 0.827] | 0.917 [0.905, 0.929] | 146.42 | 200.27 |
| hybrid | 0.766 [0.748, 0.784] | 0.945 [0.935, 0.955] | 0.978 [0.971, 0.984] | 0.846 [0.833, 0.859] | 0.879 [0.868, 0.889] | 0.961 [0.953, 0.969] | 166.4 | 218.73 |
| hybrid+light | 0.782 [0.764, 0.800] | 0.947 [0.937, 0.956] | 0.972 [0.965, 0.979] | 0.854 [0.842, 0.867] | 0.883 [0.873, 0.894] | 0.957 [0.948, 0.966] | 203.59 | 272.7 |

## Paired difference vs `bm25` (same queries)

| Config | Δ recall@5 [95% CI] | p | Δ MRR@10 [95% CI] | p |
|---|---|---|---|---|
| dense | -0.031 [-0.047, -0.015] | 0.000 | -0.070 [-0.087, -0.054] | 0.000 |
| hybrid | +0.019 [+0.009, +0.029] | 0.000 | +0.003 [-0.008, +0.014] | 0.575 |
| hybrid+light | +0.020 [+0.011, +0.030] | 0.000 | +0.011 [+0.001, +0.022] | 0.040 |

## Where each config fails

| Config | hit@1 | hit@2-5 | answer_in_other_passage | same_article_wrong_paragraph | miss |
|---|---|---|---|---|---|
| bm25 | 1612 | 303 | 32 | 70 | 50 |
| dense | 1413 | 437 | 52 | 89 | 76 |
| hybrid | 1584 | 370 | 34 | 50 | 29 |
| hybrid+light | 1617 | 340 | 23 | 56 | 31 |

Bucket meanings: `answer_in_other_passage` = the labelled paragraph was missed but a top-5 passage still contains the gold answer (often a labelling gap, not a retrieval error); `same_article_wrong_paragraph` = the top result is from the right article; `miss` = neither.

## Sample misses for `hybrid+light`

- **Who presided over the assembly?** (article: Martin Luther) -> top result from: Scottish Parliament
- **What was Temur Khan's Chinese-style name?** (article: Yuan dynasty) -> top result from: Genghis Khan
- **What directly correlates with the country's economic performance and wealth distribution?** (article: Kenya) -> top result from: Economic inequality
- **What has been the result of this publicity?** (article: Teacher) -> top result from: Nikola Tesla
- **How many casualties did British get?** (article: French and Indian War) -> top result from: Kenya
- **What chemical element was the cause of the Apollo 1 disastrous outcome?** (article: Oxygen) -> top result from: Apollo program
- **What did this agreement do?** (article: Huguenot) -> top result from: European Union law
- **At what university is Bellomy Field located?** (article: Super Bowl 50) -> top result from: University of Chicago
- **Where were populations centered in colonies?** (article: French and Indian War) -> top result from: Normans
- **Which type of livestock was the argricultural region known for?** (article: Southern California) -> top result from: Immune system
