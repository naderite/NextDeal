"""Benchmark full-corpus retrieval on the official InsuranceQA v2 test split."""
import argparse
import gzip
import hashlib
import json
from tempfile import NamedTemporaryFile
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

SOURCE = "https://github.com/shuzi/insuranceQA"


def embeddings(model_path, documents, questions, cache):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(str(model_path), local_files_only=True)
    model.max_seq_length = 128
    digest = hashlib.sha256(b"insuranceqa:e5:passage:128")
    digest.update("\0".join(documents).encode())
    for file in sorted(p for p in model_path.rglob("*") if p.is_file()):
        digest.update(str(file.relative_to(model_path)).encode())
        with file.open("rb") as stream:
            digest.update(hashlib.file_digest(stream, "sha256").digest())
    path = None if cache is None else cache / f"insuranceqa-{digest.hexdigest()}.npy"
    if path is not None and path.exists():
        passages = np.load(path, allow_pickle=False)
        if passages.shape != (len(documents), model.get_sentence_embedding_dimension()):
            raise ValueError("Invalid cached answer embedding shape")
    else:
        passages = model.encode(["passage: " + doc for doc in documents], batch_size=64,
                                normalize_embeddings=True, show_progress_bar=True)
        if path is not None:
            cache.mkdir(parents=True, exist_ok=True)
            with NamedTemporaryFile(dir=cache, suffix=".npy", delete=False) as output:
                tmp = Path(output.name)
                try:
                    np.save(output, passages, allow_pickle=False)
                    output.close()
                    tmp.replace(path)
                finally:
                    tmp.unlink(missing_ok=True)
    queries = model.encode(["query: " + q for q in questions], batch_size=64,
                           normalize_embeddings=True)
    return queries, passages


def benchmark(folder, model_path=None, cache=None, reranker_path=None):
    vocabulary = dict(line.rstrip("\n").split("\t", 1)
                      for line in (folder / "vocabulary").open(encoding="utf-8"))

    def decode(encoded):
        return " ".join(vocabulary[token] for token in encoded.split())

    with gzip.open(folder / "InsuranceQA.label2answer.raw.encoded.gz", "rt", encoding="utf-8") as stream:
        answers = [line.rstrip("\n").split("\t", 1) for line in stream]
    doc_ids = [k for k, _ in answers]
    documents = [decode(value) for _, value in answers]
    qs_path = folder / "InsuranceQA.question.anslabel.raw.100.pool.solr.test.encoded.gz"
    with gzip.open(qs_path, "rt", encoding="utf-8") as stream:
        rows = [line.rstrip("\n").split("\t") for line in stream]
    questions = [decode(row[1]) for row in rows]
    labels = [set(row[2].split()) for row in rows]
    assert len(questions) == 2000 and len(documents) == 27413
    valid_ids = set(doc_ids)
    if any(not relevant or not relevant <= valid_ids for relevant in labels):
        raise ValueError("Test question has missing relevance labels or unknown answer IDs")
    word = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", dtype=np.float32, sublinear_tf=True)
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, dtype=np.float32,
                           sublinear_tf=True)
    matrices = [v.fit_transform(documents).T.tocsc() for v in (word, char)]
    queries = [v.transform(questions) for v in (word, char)]
    dense = embeddings(model_path, documents, questions, cache) if model_path else None
    reranker = None
    if reranker_path:
        if dense is None:
            raise ValueError("Reranking requires --dense-model")
        from sentence_transformers import CrossEncoder

        reranker = CrossEncoder(str(reranker_path), max_length=128, local_files_only=True)
    stats = {"lexical": ([], [])}
    if dense is not None:
        stats.update({"dense": ([], []), "fusion": ([], [])})
    if reranker is not None:
        stats["reranked"] = ([], [])
    recall_at_100 = {method: [] for method in stats}
    union_recall = {100: [], 250: [], 500: []}
    rerank_candidate_recall = []
    for start in range(0, len(questions), 32):
        stop = start + 32
        lexical_scores = (.7 * (queries[0][start:stop] @ matrices[0])
                          + .3 * (queries[1][start:stop] @ matrices[1])).toarray()
        lexical_rank = np.argsort(-lexical_scores, axis=1, kind="stable")
        rankings = {"lexical": lexical_rank[:, :100]}
        if dense is not None:
            dense_scores = dense[0][start:stop] @ dense[1].T
            dense_rank = np.argsort(-dense_scores, axis=1, kind="stable")
            for row_index, (lex_order, dense_order) in enumerate(zip(lexical_rank, dense_rank, strict=True), start=start):
                relevant = labels[row_index]
                for cutoff, recalls in union_recall.items():
                    candidates = {doc_ids[j] for j in lex_order[:cutoff]} | {doc_ids[j] for j in dense_order[:cutoff]}
                    recalls.append(len(relevant & candidates) / len(relevant))
            rankings["dense"] = dense_rank[:, :100]
            # Give equal weight to lexical and semantic rank positions.
            rr = 1 / (20 + np.argsort(lexical_rank, axis=1))
            rr += 1 / (20 + np.argsort(dense_rank, axis=1))
            rankings["fusion"] = np.argsort(-rr, axis=1, kind="stable")[:, :100]
        if reranker is not None:
            candidates = [
                np.union1d(lex[:32], semantic[:32])
                for lex, semantic in zip(lexical_rank, dense_rank, strict=True)
            ]
            pairs = [(questions[start + i], documents[index])
                     for i, indexes in enumerate(candidates) for index in indexes]
            logits = reranker.predict(pairs, batch_size=128, show_progress_bar=False)
            rankings["reranked"] = []
            offset = 0
            for i, indexes in enumerate(candidates):
                end = offset + len(indexes)
                rankings["reranked"].append(indexes[np.argsort(-logits[offset:end], kind="stable")])
                relevant = labels[start + i]
                rerank_candidate_recall.append(len(relevant & {doc_ids[k] for k in indexes}) / len(relevant))
                offset = end
            if start % 256 == 0:
                print(f"Reranked {min(stop, len(questions))}/{len(questions)} questions", flush=True)
        for method, top in rankings.items():
            recalls, hits = stats[method]
            for row_index, indexes in enumerate(top, start=start):
                relevant = labels[row_index]
                found = relevant & {doc_ids[i] for i in indexes[:5]}
                recall_at_100[method].append(len(relevant & {doc_ids[i] for i in indexes}) / len(relevant))
                recalls.append(len(found) / len(relevant))
                hits.append(bool(found))
    summary = {
        method: {"mean_recall_at_5": float(np.mean(recalls)), "hit_rate_at_5": float(np.mean(hits)),
                 **({"mean_recall_at_100": float(np.mean(recall_at_100[method]))}
                    if method != "reranked" else {})}
        for method, (recalls, hits) in stats.items()
    }
    return {
        "source": SOURCE, "attribution": "Feng, Xiang, Glass, Wang and Zhou (ASRU 2015), "
                                        "Applying Deep Learning to Answer Selection: A Study and An Open Task",
        "use": "Research only per dataset publisher",
        "questions": len(questions), "candidate_answers_per_question": len(documents),
        "dense_model": model_path.name if model_path else None,
        "retrieval_results": summary,
        "reranker": {"model": reranker_path.name, "candidates_per_retriever": 32,
                     "candidate_recall": float(np.mean(rerank_candidate_recall))}
        if reranker is not None else None,
        "target_recall_at_5": .85,
        "candidate_union_recall": {
            str(k): float(np.mean(recall)) for k, recall in union_recall.items() if recall
        },
        "ideal_top5_recall_ceiling": float(np.mean([min(5, len(relevant)) / len(relevant) for relevant in labels])),
        "scope": "All publisher answer documents are searched. Candidate pools are ignored. "
                 "Publisher test labels measure answer retrieval, not correctness of generated advice."
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dense-model", type=Path, help="Local E5 sentence-transformer directory")
    parser.add_argument("--cache", type=Path, help="Directory for the passage embedding cache")
    parser.add_argument("--reranker", type=Path, help="Local cross-encoder directory")
    args = parser.parse_args()
    args.output.write_text(json.dumps(benchmark(args.directory, args.dense_model, args.cache, args.reranker), indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
