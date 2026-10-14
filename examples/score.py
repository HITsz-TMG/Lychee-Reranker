from lychee_reranker import LycheeReranker


reranker = LycheeReranker("models/Lychee-Reranker-V1-Nano")
scores = reranker.predict(
    [
        ("What is the capital of France?", "Paris is the capital of France."),
        ("What is the capital of France?", "Berlin is the capital of Germany."),
    ]
)
print(scores)
