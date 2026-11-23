"""Optional integrations for heavyweight research tooling.

These functions deliberately do not download models or contact services until
the caller invokes them.
"""

from __future__ import annotations


def labse_similarity(sources: list[str], targets: list[str], model_name: str = "sentence-transformers/LaBSE") -> list[float]:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise RuntimeError("Install sentence-transformers for LaBSE filtering.") from error
    model = SentenceTransformer(model_name)
    source_embeddings = model.encode(sources, normalize_embeddings=True)
    target_embeddings = model.encode(targets, normalize_embeddings=True)
    return [float(source @ target) for source, target in zip(source_embeddings, target_embeddings)]


def comet_scores(sources: list[str], hypotheses: list[str], references: list[str] | None = None, model_name: str = "Unbabel/wmt22-comet-da") -> list[float]:
    try:
        from comet import download_model, load_from_checkpoint
    except ImportError as error:
        raise RuntimeError("Install unbabel-comet for COMET/CometKiwi scores.") from error
    model = load_from_checkpoint(download_model(model_name))
    data = [{"src": source, "mt": hypothesis, **({"ref": reference} if references else {})} for source, hypothesis, reference in zip(sources, hypotheses, references or [None] * len(sources))]
    return list(model.predict(data, batch_size=8, gpus=1 if next(model.parameters()).is_cuda else 0).scores)


def stanza_pipeline():
    try:
        import stanza
    except ImportError as error:
        raise RuntimeError("Install stanza and run stanza.download('hi') for morphology analysis.") from error
    return stanza.Pipeline("hi", processors="tokenize,pos,lemma,depparse", use_gpu=False)
