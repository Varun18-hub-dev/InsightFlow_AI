# Retrieval benchmark: squad

2067 queries over 2067 documents (2925 chunks). Seed 42. Run 2026-11-22T16:24:09+00:00 on Linux x86_64, Python 3.13.18.

Means with 95% bootstrap confidence intervals.

| Config | recall@1 | recall@5 | recall@10 | mrr@10 | ndcg@10 | answer_in_top5 | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|---|
| bm25 | 0.780 [0.761, 0.797] | 0.926 [0.915, 0.937] | 0.948 [0.939, 0.957] | 0.843 [0.830, 0.856] | 0.869 [0.857, 0.880] | 0.941 [0.931, 0.951] | 3.34 | 7.3 |
| bm25+light | 0.794 [0.776, 0.811] | 0.930 [0.919, 0.940] | 0.951 [0.942, 0.960] | 0.852 [0.839, 0.865] | 0.876 [0.865, 0.888] | 0.941 [0.930, 0.951] | 4.0 | 8.08 |

## Paired difference vs `bm25` (same queries)

| Config | Δ recall@5 [95% CI] | p | Δ MRR@10 [95% CI] | p |
|---|---|---|---|---|
| bm25+light | +0.003 [-0.000, +0.007] | 0.110 | +0.009 [+0.005, +0.014] | 0.000 |

## Where each config fails

| Config | hit@1 | hit@2-5 | answer_in_other_passage | same_article_wrong_paragraph | miss |
|---|---|---|---|---|---|
| bm25 | 1612 | 303 | 32 | 70 | 50 |
| bm25+light | 1641 | 281 | 25 | 75 | 45 |

Bucket meanings: `answer_in_other_passage` = the labelled paragraph was missed but a top-5 passage still contains the gold answer (often a labelling gap, not a retrieval error); `same_article_wrong_paragraph` = the top result is from the right article; `miss` = neither.

## Sample misses for `bm25+light`

- **What were the origins of the Raouliii family?** (article: Normans) -> top result from: Newcastle upon Tyne
- **Who presided over the assembly?** (article: Martin Luther) -> top result from: Scottish Parliament
- **What was Temur Khan's Chinese-style name?** (article: Yuan dynasty) -> top result from: Genghis Khan
- **Who else did DATNET 1 refer to** (article: Packet switching) -> top result from: Intergovernmental Panel on Climate Change
- **What is a PPP also known as?** (article: Construction) -> top result from: Fresno, California
- **At what speed did the turbine operate?** (article: Nikola Tesla) -> top result from: Steam engine
- **Why did he fire the secretary?** (article: Nikola Tesla) -> top result from: Jacksonville, Florida
- **Where do platycenida live?** (article: Ctenophora) -> top result from: Super Bowl 50
- **What has been the result of this publicity?** (article: Teacher) -> top result from: Nikola Tesla
- **What will the Prime Minister have power over?** (article: Kenya) -> top result from: Harvard University
