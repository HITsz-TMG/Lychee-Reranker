"""Prompt templates shared by training and inference."""

DEFAULT_INSTRUCTION = "Given a query, retrieve documents that answer the query."
DEFAULT_SYSTEM_INSTRUCTION = (
    "Judge whether the Document meets the requirements based on the Query and "
    'the Instruct provided. Note that the answer can only be "yes" or "no".'
)


def build_decoder_prompt(query: str, instruction: str, system_instruction: str) -> str:
    """Build the exact decoder prefix used by Lychee-Reranker-V1."""
    return (
        "<bos><start_of_turn>user\n"
        f"{system_instruction}\n\n"
        f"<Instruct>: {instruction}\n"
        f"<Query>: {query}<end_of_turn>\n"
        "<start_of_turn>model\n\n\n\n"
    )


def build_document(document: str) -> str:
    return f"<Document>: {document}"
