# Research → design decisions

Goal of this note: every component of the agent cites the evidence it is built on, so a change can be
argued from the literature and then checked on `splits/val300.txt`. Two constraints drive everything:
PA punishes a wrong answer as much as it rewards a right one, and the private test brings new
industries plus its own rulebook, so nothing may be tuned to train's shops or rules.

Baseline v0 measured first (80 val questions, 8 shops): **PA −0.075**, 23 correct / 29 wrong /
28 abstain, precision 0.44. Answering without a confidence gate is worse than abstaining on everything.

## 1. Business semantics in the prompt (rulebook + README)

- BIRD: removing the external-knowledge "evidence" costs GPT-4 ~20 points of execution accuracy
  (34.9 vs 54.9). [BIRD reviews](https://liner.com/review/can-llm-already-serve-as-a-database-interface-a-big),
  [SEED](https://arxiv.org/html/2506.07423v1), [DIVER](https://arxiv.org/html/2602.12064)
- A 4 KB hand-written semantic layer (measures, conventions, disambiguation) lifts three frontier
  models by +17 to +23 points; model choice within a tier mattered less than the document.
  [Semantic Layers paper](https://arxiv.org/abs/2604.25149)

**Decision:** the split's own `rulebook.md` and the shop's README are always in the prompt, verbatim.
No rule is restated in code, so a new rulebook needs no change.

## 2. Table representation

- Hallucinated / misidentified columns are a documented failure: BLADE saw an invented
  "RefCountry" column; DataFrame QA lists column-reference errors and fixes them with column
  descriptions. [BLADE](https://arxiv.org/pdf/2408.09667), [DataFrame QA](https://arxiv.org/pdf/2401.15463),
  [DataSciBench](https://arxiv.org/pdf/2502.13897)
- SpreadsheetLLM: structure-aware compression (anchors, inverted index of repeated values) beats
  naive serialization by 25.6% on table detection. [SpreadsheetLLM](https://arxiv.org/html/2407.09025v2)
- TableMaster: table normalization and program-aided reasoning address numeric errors of text
  reasoning. [TableMaster](https://arxiv.org/abs/2501.19378)
- ReFoRCE: iterative column exploration and format restriction. [ReFoRCE](https://arxiv.org/abs/2502.00675)

**Decision (01_data §A5):** per column show the README meaning, cell kinds, blank share, the parsed
column to use and the date range over the whole column (a head sample hid 14 of 15 months in biz_101),
and ALL values of low-cardinality columns. Same-column monthly sheets are described once with the
union date range. Column exploration is done once, deterministically, instead of by the model.

## 3. Rule-agnostic helpers (normalization layer)

v0 logs: the model failed on `'1 thg 6 2025'` twice and repaired by rewriting the same broken code.
**Decision (src/datamate/helpers.py):** tested parsers for money, quantity and date spellings (Excel serials,
ISO without day-first, `1/4` with the sheet's year), text folding, fill-down-then-dedupe of order
lines, lunar→solar lookup. Mechanics only; which rows count as revenue stays with the rulebook.

## 3b. Normalize at load time instead of asking the model to parse

v2's logs (30 validation questions) showed the model calling `pd.to_datetime` on `'1 thg 6 2025'`
instead of the helpers: every date became NaT, the filter kept nothing and it answered 0 ("doanh thu
hôm qua" → 0 for a reference of 6,873,700). A 9B model does not reliably use helpers it is told about.
TableMaster's table normalization points the same way. [TableMaster](https://arxiv.org/abs/2501.19378)

**Decision (01_data §A3, used by prompt v3):** parse once, deterministically: `<col>__date`,
`<col>__dt`, `<col>__num` next to the raw columns; fill order-level cells down; stack same-column
tables into `ALL …*` views. Verified without a GPU: fixed-code monthly revenue over the normalized
tables equals all 71 total-revenue train answers it applies to (`03_evaluate` §4).

## 4. Value linking

- CHESS: retrieving database values that match the question adds ~5 points; schema pruning ~2.
  [CHESS](https://arxiv.org/html/2405.16755v1)

**Decision (02_agent §2, ValueIndex):** names in the data whose words all occur in the question (or
≥ 3 content words) are listed next to the question with their table and column, so the model
filters on a real spelling ('Trà vải (M)', 'Natri clorid', 'Size XL').

## 5. Where context goes, and what not to add

- Lost in the Middle: models use information at the start and end of a long context best.
  [Liu et al.](https://arxiv.org/abs/2307.03172)
- Irrelevant context sharply lowers accuracy (GSM-IC). [Shi et al. 2023](https://arxiv.org/pdf/2302.00093)

**Decision:** the question and its matched values sit last. Retrieving rulebook rows to repeat next to
the question is **off by default** (`RULES_K = 0`). Measured on 48 hand-labelled train questions
(`03_evaluate` §3, labels in `splits/rule_labels_train.json`):

| retriever | R@3 | P@3 | R@5 | P@5 |
|---|---|---|---|---|
| lexical (IDF overlap, time words kept) | 0.41 | 0.30 | 0.47 | 0.21 |
| Qwen3-Embedding-0.6B | 0.27 | 0.19 | 0.31 | 0.13 |
| bge-m3 | 0.29 | 0.21 | 0.36 | 0.16 |
| hybrid lexical + bge-m3 (RRF) | 0.36 | 0.26 | 0.41 | 0.18 |

Off-the-shelf bi-encoders did not beat lexical matching on short Vietnamese rule rows, and RRF did
not add over lexical. At P@3 ≈ 0.3, two of three injected rows would be irrelevant. Hybrid search is
expected to beat either part on larger corpora ([BGE-M3](https://www.emergentmind.com/topics/bge-m3),
[StratRAG](https://arxiv.org/pdf/2604.22757)), but not here. Next candidates if revisited: map time
expressions to §2 rows deterministically, or rerank with a cross-encoder.

## 6. No few-shot examples from train

- Calibrate Before Use: example choice, order and label balance swing accuracy from near chance
  to near SOTA (majority-label and recency bias). [Zhao et al. 2021](https://arxiv.org/pdf/2102.09690)

**Decision:** zero-shot. Train examples would anchor the model on train's industries, rulebook and
answer-type mix (63% numbers), none of which hold for the private test.

## 7. Interpretation and answerability before code

- TrustDABench: models rarely detect missing or conflicting evidence and continue down executable
  but unsupported analysis paths. [TrustDABench](https://arxiv.org/abs/2608.24145)
- Data Interpreter: step-wise verification raises InfiAgent-DABench from 75.9% to 94.9%.
  [Data Interpreter](https://arxiv.org/abs/2402.18679)

**Decision (prompt v1):** the reply starts with an interpretation: metric, exact period from AS_OF,
filters, rules applied, tables/columns, an explicit answerability check (information not recorded,
or period entirely outside the shown date ranges → unanswerable; never substitute a proxy), and the
answer type. The code must print intermediate counts and reconsider when a filter keeps 0 rows.

## 8. Sampling, voting and abstention

- Execution-based self-consistency: run candidates, drop errors, pick the most common result.
  [C3](https://arxiv.org/pdf/2307.07306), [MBR-exec](https://arxiv.org/abs/2204.11454),
  [Query and Conquer](https://arxiv.org/html/2503.24364v1); gains saturate with one prompt, mixing
  prompts adds diversity. ReFoRCE uses consensus to keep high-confidence answers and defer the rest.
- Semantic entropy: cluster sampled answers by meaning; dispersion flags confabulation, but it does
  not catch answers that are consistently wrong. [Farquhar et al. 2024](https://oatml.cs.ox.ac.uk/blog/2024/06/19/detecting_hallucinations_2024.html)
- Abstention survey: abstain when calibrated confidence is under a threshold; verbalized and
  probability confidence are both poorly calibrated. [Know Your Limits](https://arxiv.org/html/2407.18418v3)
- TrustSQL: under penalties, no method consistently beats "abstain on everything".
  [TrustSQL](https://arxiv.org/html/2403.15879v4)

**Decision (02_agent §5):** k samples (default 5, T=0.7, fixed seed), clustered by the value the scorer
compares; support = largest cluster / k; answer only if support ≥ τ. τ is tuned on validation by PA
(`03_evaluate` §2), never on the public test. Because consistent errors survive voting, rule-
based checks (date coverage, field availability) stay in the prompt.

## 9. Repair turns

- Self-repair gains are modest once its cost is counted; spending the budget on diverse first
  samples does better, and feedback quality is the bottleneck. [Olausson et al. 2024](https://arxiv.org/pdf/2306.09896)

**Decision (v7, 2026-11-01):** every failed sample gets one greedy repair turn and its fix replaces it in
the vote. The earlier rule (repair only when all K failed, the fix counting 1/K < τ) never produced an
answer, while about 10% of public samples crashed on a column or table name the error message names.
Superseded rule: repair only questions where every sample failed, once, greedily; a repaired answer
carries the support of a single sample.

## 10. Self-verification by the same model

- LLM judges prefer their own / low-perplexity outputs. [Wataoka et al.](https://arxiv.org/html/2410.21819v1)

**Decision:** no "is this answer right?" pass by the same model. Verification is by execution
agreement and deterministic checks.

## 11. Thinking mode

- Qwen3-8B on LiveCodeBench: 21.8 non-thinking vs 55.6 thinking in one paper's setup; thinking costs
  2–7× latency. [ThinKV](https://arxiv.org/pdf/2510.01290), [DuoMem](https://arxiv.org/pdf/2606.29961)

**Decision:** `--thinking` flag, to be measured on val300 against the 6 h budget (2 × RTX 5090).
Not on by default until measured.

## Related benchmarks worth reading for error taxonomies

[DABstep](https://arxiv.org/abs/2506.23719) (best agent 14.55% on hard multi-file tasks at launch),
[InfiAgent-DABench](https://arxiv.org/pdf/2401.05507), [Spider 2.0 / ReFoRCE](https://arxiv.org/abs/2502.00675).
