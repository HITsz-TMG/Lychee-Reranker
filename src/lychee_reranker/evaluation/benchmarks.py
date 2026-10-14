BEIR_RETRIEVER_INSTRUCTIONS = {
    "dbpedia-entity": "Given a query, retrieve relevant entity descriptions from DBPedia.",
    "arguana": "Given a claim, find documents that refute the claim.",
    "climate-fever": "Given a claim about climate change, retrieve documents that support or refute the claim.",
    "cqadupstack": "Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question.",
    "fever": "Given a claim, retrieve documents that support or refute the claim.",
    "fiqa": "Given a financial question, retrieve user replies that best answer the question.",
    "hotpotqa": "Given a multi-hop question, retrieve documents that can help answer the question.",
    "msmarco": "Given a web search query, retrieve relevant passages that answer the query.",
    "nfcorpus": "Given a question, retrieve relevant documents that best answer the question.",
    "nq": "Given a question, retrieve Wikipedia passages that answer the question.",
    "quora": "Given a question, retrieve questions that are semantically equivalent to the given question.",
    "scidocs": "Given a scientific paper title, retrieve paper abstracts that are cited by the given paper.",
    "scifact": "Given a scientific claim, retrieve documents that support or refute the claim.",
    "webis-touche2020": "Given a question, retrieve detailed and persuasive arguments that answer the question.",
    "trec-covid": "Given a query on COVID-19, retrieve documents that answer the query.",
}

# The reranker map follows FlagEmbedding's BEIRRerankerPrompts and the
# instructions listed in arXiv:2606.22807v4 Table 11.  Keep it separate from
# the retriever map because MSMARCO intentionally uses a different prompt.
BEIR_RERANKER_INSTRUCTIONS = {
    **BEIR_RETRIEVER_INSTRUCTIONS,
    "msmarco": "Given a query, retrieve documents that answer the query.",
    "webis-touche2020": "Given a query, retrieve documents that answer the query.",
    "trec-covid": "Given a query, retrieve documents that answer the query.",
}

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
