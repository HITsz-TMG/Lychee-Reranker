BEIR_V1_TASKS = (
    "arguana",
    "climate-fever",
    "cqadupstack",
    "dbpedia-entity",
    "fever",
    "fiqa",
    "hotpotqa",
    "msmarco",
    "nfcorpus",
    "nq",
    "quora",
    "scidocs",
    "scifact",
)

MIRACL_LANGUAGES = (
    "ar",
    "bn",
    "de",
    "en",
    "es",
    "fa",
    "fi",
    "fr",
    "hi",
    "id",
    "ja",
    "ko",
    "ru",
    "sw",
    "te",
    "th",
    "yo",
    "zh",
)

LMEB_DIALOGUE_TASKS = ("ConvoMem", "LoCoMo", "LongMemEval", "MemBench", "REALTALK", "TMD")

DEFAULT_METRICS = {
    "beir": ("ndcg@10", "ndcg@100", "recall@10", "recall@100"),
    "miracl": ("ndcg@10", "recall@100"),
    "lmeb": ("ndcg@10", "ndcg@100", "recall@10", "recall@100", "mrr@10"),
}
