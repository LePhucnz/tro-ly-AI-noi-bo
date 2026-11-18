
"""
rag/retriever.py
Advanced Retriever hỗ trợ linh hoạt 3 chiến lược tìm kiếm tài liệu:
1. 'similarity': Tìm kiếm độ tương đồng Cosine cơ bản.
2. 'mmr': Maximal Marginal Relevance - giảm thiểu lặp ý, tăng tính đa dạng nguồn tin.
3. 'similarity_score_threshold': Lọc bỏ các kết quả có điểm tương đồng thấp hơn ngưỡng.
"""

from __future__ import annotations

import logging
import math
import re
import unicodedata
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None  # type: ignore

from core.config import get_settings
from core.exceptions import InvalidRetrieverStrategyError
from documents.loader_factory import Document
from rag.vector_store import ChromaVectorStoreManager

logger = logging.getLogger("advanced_retriever")


def _remove_accents(input_str: str) -> str:
    """Loại bỏ dấu tiếng Việt để đối chiếu không dấu."""
    nfkd = unicodedata.normalize("NFKD", input_str)
    return "".join(c for c in nfkd if not unicodedata.combining(c)).lower()


def _tokenize_text(text: str) -> List[str]:
    """Tokenize văn bản, bổ sung cả từ gốc và từ không dấu phục vụ BM25."""
    clean = re.sub(r"[^\w\s]", " ", text.lower())
    unaccent = _remove_accents(clean)
    return clean.split() + unaccent.split()


def _cosine_similarity(v1: Any, v2: Any) -> float:
    """Tính độ tương đồng Cosine giữa hai vector."""
    if hasattr(v1, "tolist"):
        v1 = v1.tolist()
    if hasattr(v2, "tolist"):
        v2 = v2.tolist()
    dot = sum(float(a) * float(b) for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(float(a) * float(a) for a in v1))
    norm2 = math.sqrt(sum(float(b) * float(b) for b in v2))
    if norm1 < 1e-9 or norm2 < 1e-9:
        return 0.0
    return max(0.0, min(1.0, dot / (norm1 * norm2)))


def _maximal_marginal_relevance(
    query_embedding: List[float],
    doc_embeddings: List[List[float]],
    documents: List[Document],
    k: int = 4,
    lambda_mult: float = 0.7,
) -> List[Document]:
    """
    Thuật toán Maximal Marginal Relevance (MMR) chuẩn quốc tế.
    Cân bằng giữa độ phù hợp với câu hỏi (Relevance) và tính đa dạng thông tin (Diversity):
    MMR = argmax [ lambda * Sim(d, query) - (1 - lambda) * max(Sim(d, d_selected)) ]
    """
    if not documents:
        return []

    if len(documents) <= k:
        return documents

    selected_indices: List[int] = []
    unselected_indices: List[int] = list(range(len(documents)))

    # Tính độ tương đồng giữa query và từng tài liệu ứng viên
    query_sims = [_cosine_similarity(query_embedding, emb) for emb in doc_embeddings]

    # Chọn tài liệu đầu tiên có độ tương đồng cao nhất
    best_first_idx = max(unselected_indices, key=lambda idx: query_sims[idx])
    selected_indices.append(best_first_idx)
    unselected_indices.remove(best_first_idx)

    # Chọn các tài liệu tiếp theo theo công thức MMR
    while len(selected_indices) < k and unselected_indices:
        best_score = -float("inf")
        best_idx = unselected_indices[0]

        for u_idx in unselected_indices:
            relevance = query_sims[u_idx]
            # Tính độ tương đồng lớn nhất với các tài liệu đã chọn
            max_redundancy = max(
                _cosine_similarity(doc_embeddings[u_idx], doc_embeddings[s_idx])
                for s_idx in selected_indices
            )
            mmr_score = (lambda_mult * relevance) - ((1.0 - lambda_mult) * max_redundancy)

            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = u_idx

        selected_indices.append(best_idx)
        unselected_indices.remove(best_idx)

    # Gán điểm số vào metadata
    result_docs: List[Document] = []
    for s_idx in selected_indices:
        doc = documents[s_idx]
        doc.metadata["score"] = round(query_sims[s_idx], 4)
        result_docs.append(doc)

    return result_docs


class AdvancedRetriever:
    """
    Advanced Retriever với tham số hóa linh hoạt 3 chiến lược tìm kiếm:
    - 'similarity': Top-k tương đồng cao nhất.
    - 'mmr': Tối ưu đa dạng, loại bỏ trùng lặp thông tin.
    - 'similarity_score_threshold': Lọc chặt chẽ theo ngưỡng tối thiểu.
    """

    SUPPORTED_STRATEGIES = ["similarity", "mmr", "similarity_score_threshold", "hybrid"]
    STRATEGY_ALIASES = {
        "threshold": "similarity_score_threshold",
        "score_threshold": "similarity_score_threshold",
        "similarity_threshold": "similarity_score_threshold",
        "bm25": "hybrid",
        "keyword": "hybrid",
    }

    def __init__(
        self,
        vector_store_manager: Optional[ChromaVectorStoreManager] = None,
    ) -> None:
        self.vector_store_manager = vector_store_manager or ChromaVectorStoreManager()
        self.settings = get_settings()

    def _embed_query(self, query: str) -> List[float]:
        """Tạo vector nhúng cho câu truy vấn."""
        emb_fn = self.vector_store_manager.embedding_function
        if hasattr(emb_fn, "embed_query"):
            return list(emb_fn.embed_query(query))
        elif callable(emb_fn):
            res = emb_fn(query)
            if hasattr(res, "tolist"):
                res = res.tolist()
            if isinstance(res, list) and len(res) > 0 and isinstance(res[0], (list, tuple)):
                return list(res[0])
            return list(res)
        return []

    def retrieve(
        self,
        query: str,
        strategy: str = "similarity",
        k: Optional[int] = None,
        threshold: Optional[float] = None,
        lambda_mult: Optional[float] = None,
        fetch_k: Optional[int] = None,
    ) -> List[Document]:
        """
        Thực hiện tìm kiếm tài liệu từ Vector Store theo chiến lược được chỉ định.

        :param query: Câu hỏi truy vấn của người dùng.
        :param strategy: 'similarity' | 'mmr' | 'similarity_score_threshold'.
        :param k: Số lượng tài liệu cần lấy (mặc định lấy từ Settings.DEFAULT_TOP_K).
        :param threshold: Ngưỡng lọc điểm tương đồng (mặc định lấy từ Settings.DEFAULT_SCORE_THRESHOLD).
        :param lambda_mult: Hệ số lambda cho thuật toán MMR.
        :param fetch_k: Số lượng ứng viên khởi tạo cho MMR.
        :return: Danh sách Document kèm metadata['score'].
        """
        clean_strategy = strategy.strip().lower()
        clean_strategy = self.STRATEGY_ALIASES.get(clean_strategy, clean_strategy)
        if clean_strategy not in self.SUPPORTED_STRATEGIES:
            raise InvalidRetrieverStrategyError(
                strategy=strategy,
                valid_strategies=self.SUPPORTED_STRATEGIES,
            )

        top_k = k or self.settings.DEFAULT_TOP_K
        score_thresh = threshold if threshold is not None else self.settings.DEFAULT_SCORE_THRESHOLD
        l_mult = lambda_mult if lambda_mult is not None else self.settings.MMR_LAMBDA_MULT
        f_k = fetch_k or self.settings.MMR_FETCH_K

        # 1. Tạo query embedding
        query_embedding = self._embed_query(query)

        # 2. Lấy dữ liệu ứng viên từ ChromaDB hoặc In-Memory (kết hợp Dense + BM25 Keyword)
        candidate_docs, candidate_embeddings = self._get_candidates(
            query=query,
            query_embedding=query_embedding,
            fetch_limit=max(f_k, top_k * 4),
            strategy=clean_strategy,
        )

        if not candidate_docs:
            logger.info("Vector Store chưa có dữ liệu nào cho truy vấn: '%s'", query)
            return []

        # 3. Áp dụng chiến lược tìm kiếm
        if clean_strategy in ("similarity", "hybrid"):
            # Sắp xếp theo score giảm dần và lấy top_k
            ranked = sorted(candidate_docs, key=lambda d: d.metadata.get("score", 0.0), reverse=True)
            return ranked[:top_k]

        elif clean_strategy == "similarity_score_threshold":
            # Lọc bỏ các kết quả có score < threshold
            filtered = [
                d for d in candidate_docs
                if d.metadata.get("score", 0.0) >= score_thresh
            ]
            ranked = sorted(filtered, key=lambda d: d.metadata.get("score", 0.0), reverse=True)
            logger.info(
                "Chiến lược threshold (>= %.2f): giữ lại %d/%d tài liệu.",
                score_thresh,
                len(ranked[:top_k]),
                len(candidate_docs),
            )
            return ranked[:top_k]

        elif clean_strategy == "mmr":
            # Thực thi giải thuật Maximal Marginal Relevance
            selected = _maximal_marginal_relevance(
                query_embedding=query_embedding,
                doc_embeddings=candidate_embeddings,
                documents=candidate_docs,
                k=top_k,
                lambda_mult=l_mult,
            )
            logger.info("Chiến lược MMR hoàn tất: chọn %d tài liệu đa dạng.", len(selected))
            return selected

        return []

    def _get_candidates(
        self,
        query: str,
        query_embedding: List[float],
        fetch_limit: int,
        strategy: str = "similarity",
    ) -> Tuple[List[Document], List[List[float]]]:
        """Trích xuất tài liệu ứng viên và embedding tương ứng từ Vector Store."""
        docs: List[Document] = []
        embeddings: List[List[float]] = []

        if self.vector_store_manager.is_native_chroma:
            try:
                collection = self.vector_store_manager.get_collection()
                if collection is None:
                    return [], []
                total_in_db = collection.count()
                if total_in_db == 0:
                    return [], []

                n_results = min(fetch_limit, total_in_db)
                results = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=n_results,
                    include=["documents", "metadatas", "distances", "embeddings"],
                )

                docs_list = results.get("documents")
                retrieved_texts = docs_list[0] if docs_list is not None and len(docs_list) > 0 else []
                metas_list = results.get("metadatas")
                retrieved_metas = metas_list[0] if metas_list is not None and len(metas_list) > 0 else []
                dists_list = results.get("distances")
                retrieved_dists = dists_list[0] if dists_list is not None and len(dists_list) > 0 else []
                embs_list = results.get("embeddings")
                retrieved_embs = embs_list[0] if embs_list is not None and len(embs_list) > 0 else None

                doc_id_set = set()
                for i, text in enumerate(retrieved_texts):
                    meta = dict(retrieved_metas[i] if i < len(retrieved_metas) else {})
                    dist = float(retrieved_dists[i]) if i < len(retrieved_dists) else 0.5
                    # Cosine Distance -> Cosine Similarity: 1.0 - distance
                    score = max(0.0, min(1.0, 1.0 - dist))
                    meta["score"] = round(score, 4)

                    chunk_id = meta.get("chunk_id", f"chk_{i}")
                    doc_id_set.add(chunk_id)

                    doc = Document(page_content=text, metadata=meta)
                    docs.append(doc)

                    if retrieved_embs is not None and i < len(retrieved_embs):
                        item_emb = retrieved_embs[i]
                        embeddings.append(item_emb.tolist() if hasattr(item_emb, "tolist") else list(item_emb))
                    else:
                        gen_emb = self.vector_store_manager.embedding_function(text)
                        if hasattr(gen_emb, "tolist"):
                            gen_emb = gen_emb.tolist()
                        if isinstance(gen_emb, list) and len(gen_emb) > 0 and isinstance(gen_emb[0], (list, tuple)):
                            gen_emb = gen_emb[0]
                        embeddings.append(list(gen_emb))

                # --- BM25 Keyword Search & Article Number Boosting ---
                if BM25Okapi is not None and query.strip():
                    try:
                        all_data = collection.get(include=["documents", "metadatas", "embeddings"])
                        all_docs_raw = all_data.get("documents") or []
                        all_metas_raw = all_data.get("metadatas") or []
                        all_embs_raw = all_data.get("embeddings")

                        if all_docs_raw:
                            tokenized_corpus = [_tokenize_text(d) for d in all_docs_raw]
                            bm25 = BM25Okapi(tokenized_corpus)
                            query_tokens = _tokenize_text(query)
                            bm25_scores = bm25.get_scores(query_tokens)
                            max_bm25 = float(max(bm25_scores)) if len(bm25_scores) > 0 and max(bm25_scores) > 0 else 1.0

                            # 1. Kiểm tra truy vấn có chứa số hiệu Điều/Khoản cụ thể (ví dụ: 'điều 26', 'dieu 26')
                            article_match = re.search(r"(?:điều|dieu|khoản|khoan|mục|muc)\s*(\d+)", query, re.IGNORECASE)
                            target_num = article_match.group(1) if article_match else None
                            article_pattern = re.compile(rf"(?:Điều|Khoản|Mục)\s+{target_num}\b", re.IGNORECASE) if target_num else None
                            next_article_pattern = re.compile(r"(?:Điều|Điêu)\s+\d+\b", re.IGNORECASE)

                            # Thu thập các chunk thuộc về cùng Điều khoản (Neighbor Chunk Expansion)
                            article_chunk_indices = set()
                            if target_num and article_pattern:
                                for j, raw_text in enumerate(all_docs_raw):
                                    if article_pattern.search(raw_text):
                                        article_chunk_indices.add(j)
                                        # Nối tiếp các chunk lân cận cùng thuộc Điều khoản đó (tối đa 4 chunk)
                                        for offset in range(1, 5):
                                            nxt = j + offset
                                            if nxt < len(all_docs_raw):
                                                nxt_text = all_docs_raw[nxt]
                                                # Dừng lại nếu gặp Điều luật tiếp theo (ví dụ: Điêu 27)
                                                nxt_match = next_article_pattern.search(nxt_text)
                                                if nxt_match and not article_pattern.search(nxt_text):
                                                    break
                                                article_chunk_indices.add(nxt)

                            # Tìm các chunk chứa đích danh điều khoản hoặc có điểm BM25 cao
                            for j, raw_text in enumerate(all_docs_raw):
                                j_meta = dict(all_metas_raw[j] if j < len(all_metas_raw) else {})
                                j_cid = j_meta.get("chunk_id", f"all_chk_{j}")
                                is_article_match = j in article_chunk_indices

                                norm_bm25 = float(bm25_scores[j]) / max_bm25 if max_bm25 > 0 else 0.0

                                # Nếu thuộc Điều luật được tìm kiếm -> gán điểm ưu tiên cực đại (0.95)
                                if is_article_match:
                                    boosted_score = 0.95
                                    j_meta["score"] = round(boosted_score, 4)

                                    if j_cid in doc_id_set:
                                        # Cập nhật score của chunk đã có trong docs
                                        for d in docs:
                                            if d.metadata.get("chunk_id") == j_cid:
                                                d.metadata["score"] = round(boosted_score, 4)
                                    else:
                                        # Bổ sung chunk này vào danh sách
                                        docs.append(Document(page_content=raw_text, metadata=j_meta))
                                        doc_id_set.add(j_cid)
                                        if all_embs_raw is not None and j < len(all_embs_raw):
                                            embeddings.append(list(all_embs_raw[j]))
                                        else:
                                            embeddings.append(list(self.vector_store_manager.embedding_function(raw_text)))

                                # Nếu là chiến lược hybrid, kết hợp điểm dense + điểm BM25
                                elif strategy == "hybrid":
                                    if j_cid in doc_id_set:
                                        for d in docs:
                                            if d.metadata.get("chunk_id") == j_cid:
                                                dense_sc = d.metadata.get("score", 0.5)
                                                d.metadata["score"] = round(0.55 * dense_sc + 0.45 * norm_bm25, 4)
                                    elif norm_bm25 >= 0.70:
                                        # Thêm top kết quả BM25 xuất sắc nếu vector search bỏ sót
                                        j_meta["score"] = round(0.45 * norm_bm25, 4)
                                        docs.append(Document(page_content=raw_text, metadata=j_meta))
                                        doc_id_set.add(j_cid)
                                        if all_embs_raw is not None and j < len(all_embs_raw):
                                            embeddings.append(list(all_embs_raw[j]))
                                        else:
                                            embeddings.append(list(self.vector_store_manager.embedding_function(raw_text)))

                    except Exception as bm_err:
                        logger.warning("BM25 / Keyword boosting error: %s", bm_err)

                return docs, embeddings

            except Exception as e:
                logger.warning("Truy vấn ChromaDB thất bại (%s), sử dụng fallback buffer.", e)

        # Fallback từ in-memory buffer
        in_mem_items = list(self.vector_store_manager._in_memory_docs.values())
        if not in_mem_items:
            return [], []

        for doc in in_mem_items:
            doc_emb = doc.metadata.get("_embedding")
            if not doc_emb:
                doc_emb = self.vector_store_manager.embedding_function(doc.page_content)
                if isinstance(doc_emb, list) and doc_emb and isinstance(doc_emb[0], list):
                    doc_emb = doc_emb[0]

            score = _cosine_similarity(query_embedding, doc_emb)
            new_meta = dict(doc.metadata)
            new_meta["score"] = round(score, 4)
            d = Document(page_content=doc.page_content, metadata=new_meta)
            docs.append(d)
            embeddings.append(doc_emb)

        # Sắp xếp trước theo điểm similarity
        combined = list(zip(docs, embeddings))
        combined.sort(key=lambda item: item[0].metadata.get("score", 0.0), reverse=True)
        combined = combined[:fetch_limit]

        return [c[0] for c in combined], [c[1] for c in combined]
