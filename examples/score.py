import argparse

from lychee_reranker import LycheeReranker


def main():
    parser = argparse.ArgumentParser(description="Score two public example passages")
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--chunk-size", type=int, default=4)
    args = parser.parse_args()
    reranker = LycheeReranker(
        args.model, device=args.device, dtype=args.dtype, chunk_size=args.chunk_size
    )
    print(
        reranker.predict(
            [
                ("What is the capital of France?", "Paris is the capital of France."),
                ("What is the capital of France?", "Berlin is the capital of Germany."),
            ]
        )
    )


if __name__ == "__main__":
    main()
