# Evaluation contract

Use `configs/eval/beir.yaml`, `miracl.yaml` or `lmeb.yaml` as a starting point.
Models and datasets are loaded from paths you provide; dataset download and
conversion are explicit external steps. Use a new output directory for each
model/configuration so candidate pools and results remain auditable.

## Local layouts

The exported BEIR layout is:

```text
/path/to/beir/<dataset>/corpus.jsonl
/path/to/beir/<dataset>/test_queries.jsonl
/path/to/beir/<dataset>/test_qrels.jsonl
```

The standard BEIR `queries.jsonl` plus `qrels/test.tsv` layout is also supported.
Archived exports with `test/test_queries.jsonl` and `test/test_qrels.jsonl`
are accepted as a fallback; flat and standard files take priority.
MSMARCO defaults to its development split through `splits: {msmarco: dev}`.
If an older local export named those same development examples `test`, override
that mapping explicitly; the loader cannot infer the identity of an aliased split.
CQADupStack has
12 child datasets under `cqadupstack/<subtask>/`; their metrics form one
macro-averaged task in the 13-task report.

MIRACL uses one directory per language, with `corpus.jsonl`,
`dev_queries.jsonl` and `dev_qrels.jsonl`. The public configurations request
all 18 languages. An incomplete requested task set must fail rather than
silently produce an average over the available subset.
German and Yoruba expose only `dev`; other supported languages expose `train`
and `dev`, matching the original loader's available-split contract.

Corpus rows contain `_id`, `text` and an optional `title`; query rows contain
`_id` and `text`. JSONL qrels use `query-id`, `corpus-id` and `score`. TSV qrels
use the corresponding three columns, with an optional header. IDs are kept as
strings. For corpora with titles, title and text are joined by a space, as in
the original benchmark loaders.

LMEB has shared corpus/candidate files and one or more query leaves:

```text
/path/to/lmeb/task_instructions.json
/path/to/lmeb/ConvoMem/corpus.jsonl
/path/to/lmeb/ConvoMem/candidates.jsonl
/path/to/lmeb/ConvoMem/<leaf>/queries.jsonl
/path/to/lmeb/ConvoMem/<leaf>/qrels.tsv
```

Each candidate row maps `scene_id` to `candidate_doc_ids`. Query rows may
specify `scene_id`; otherwise their query ID identifies the pool. Positive
qrels are restricted to that predefined pool, and queries without a remaining
positive are excluded. This filtering concerns the benchmark's permitted
pool, not the retriever's top-100 output. Retrieval misses within that pool
still count against the reranker evaluation.

For example, `task_instructions.json` can contain
`{"ConvoMem/demo": {"query": "Retrieve the relevant memory."}}` for a leaf
named `ConvoMem/demo`. The loader also resolves the benchmark's nested group
and task mappings. Candidate pools can be selected by an explicit `scene_id`
or the scene prefix in the query ID (for example, `scene_0_q_0` uses `scene_0`).

The release uses LMEB's supplied task instructions and aggregates leaf metrics
within each of the six Dialogue groups before taking the overall macro mean.
It does not implicitly include the other LMEB categories in the paper's score.

## Retrieval and scoring

The first-stage model is KaLM-Embedding-V2.5 with normalized mean pooling.
Pooling accumulates in FP32 even when model states are FP16/BF16. Inputs are
batched by token length and then restored to their original order, as in the
original FlagEmbedding encoder-only embedder.
Its Qwen2 checkpoint declares bidirectional attention; the native loader
preserves that behavior with an explicit bidirectional padding mask.
Queries receive the task instruction; documents use their original text.
BEIR uses the paper's task-specific instructions for both retrieval and reranking.
Reranking uses its task instruction, a default compression ratio of 4,
and `P(yes)` scoring. Query/passage limits are 512/512 for retrieval and
512/1024 for reranking, matching the reference configurations.
MSMARCO's retrieval and reranking prompts differ in the original evaluation;
the two prompt mappings retain that distinction.

BEIR excludes a document whose ID equals the query ID and fills the remaining
top-100 from the next retrieved candidates. MIRACL and LMEB retain such IDs.
The policy is recorded and checked when candidates are reused.

`--stage retrieve` writes candidates. `--stage all` writes candidates and
reranked results. `--stage rerank --candidate-dir /path/to/candidates` reuses
saved candidates without constructing the embedder. Candidate files contain
metadata and an input digest, which are checked against the loaded task.
Unknown/missing query or document IDs and non-finite scores are errors.

The primary metric is nDCG@10 with linear graded relevance gains, matching
`trec_eval`/BEIR. Recall and reciprocal rank regard any positive relevance as
relevant. Optional `map@k` divides summed precisions by all positive qrels.
For tied scores, nDCG/MAP/recall use descending document IDs, matching
`pytrec_eval`; MRR preserves insertion order, matching the original custom
MRR helper. Reranking and result serialization preserve candidate order within
score ties, including when saved candidates are reused at a top-k cutoff. LMEB
restricted retrieval uses the original NumPy `argsort` ordering.
Global retrieval preserves FAISS's returned order. The optional NumPy fallback
uses stable corpus order for ties; tied top-k boundaries can differ between
backends. Use the same backend and saved candidates when comparing models.
JSON metrics are fractions; only Markdown display multiplies by 100.
JSON retains full floating-point precision; the original FlagEmbedding summary
rounded metric averages to five decimal places.
Macro averaging gives each paper task equal weight. The small GPU smoke
fixtures test execution and candidate reuse, not published benchmark scores.
