r"""Qdrant vector store wrapper (embedded local store, or a server via QDRANT_URL).

Stores one point per chunk: the embedding vector + a payload of chunk metadata
and text. The collection is created with an HNSW config and cosine distance.

Note: the embedded/local Qdrant performs exact KNN (perfect recall, fast at this
corpus size). Set ``QDRANT_URL`` to point at a running Qdrant server to use the
true HNSW ANN index — the rest of the code is identical.
"""
from __future__ import annotations

import atexit
from functools import lru_cache

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (Distance, FieldCondition, Filter,
                                  HnswConfigDiff, MatchValue, PointStruct,
                                  SearchParams, VectorParams)

from .config import (EMBEDDING_DIM, HNSW_EF_CONSTRUCT, HNSW_EF_SEARCH, HNSW_M,
                     QDRANT_COLLECTION, QDRANT_PATH, QDRANT_URL, TOP_K_RETRIEVE)


def _open_local_client() -> QdrantClient:
    """Open the embedded store, clearing a *stale* lock left by a hard-killed
    process. On Windows the lock file can only be unlinked if no live process
    still holds it, so this is safe: if a real instance is running, the unlink
    raises and we re-raise the original error."""
    try:
        return QdrantClient(path=str(QDRANT_PATH))
    except RuntimeError as e:
        if "already accessed" not in str(e):
            raise
        try:
            (QDRANT_PATH / ".lock").unlink(missing_ok=True)
        except OSError:
            raise e
        return QdrantClient(path=str(QDRANT_PATH))


class VectorStore:
    def __init__(self, collection: str = QDRANT_COLLECTION):
        if QDRANT_URL:
            self.client = QdrantClient(url=QDRANT_URL)
            self.mode = f"server ({QDRANT_URL})"
        else:
            QDRANT_PATH.mkdir(parents=True, exist_ok=True)
            self.client = _open_local_client()
            self.mode = "embedded (exact KNN)"
        self.collection = collection
        atexit.register(self.close)  # release the local lock cleanly on exit

    def close(self) -> None:
        try:
            self.client.close()
        except Exception:
            pass

    def ensure_collection(self, dim: int = EMBEDDING_DIM, recreate: bool = False) -> None:
        exists = self.client.collection_exists(self.collection)
        if recreate and exists:
            self.client.delete_collection(self.collection)
            exists = False
        if not exists:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
                hnsw_config=HnswConfigDiff(m=HNSW_M, ef_construct=HNSW_EF_CONSTRUCT),
            )

    def upsert(self, ids: list[str], vectors: np.ndarray, payloads: list[dict],
               batch: int = 256) -> None:
        for s in range(0, len(ids), batch):
            pts = [
                PointStruct(id=i, vector=v.tolist(), payload=p)
                for i, v, p in zip(ids[s:s + batch], vectors[s:s + batch], payloads[s:s + batch])
            ]
            self.client.upsert(collection_name=self.collection, points=pts)

    def search(self, vector: np.ndarray, limit: int = TOP_K_RETRIEVE,
               filename: str | None = None) -> list[tuple]:
        flt = None
        if filename:
            flt = Filter(must=[FieldCondition(key="filename", match=MatchValue(value=filename))])
        res = self.client.query_points(
            collection_name=self.collection, query=vector.tolist(), limit=limit,
            query_filter=flt, with_payload=True,
            search_params=SearchParams(hnsw_ef=HNSW_EF_SEARCH),
        )
        return [(p.id, float(p.score), p.payload) for p in res.points]

    def count(self) -> int:
        try:
            return self.client.count(self.collection).count
        except Exception:
            return 0

    def filename_exists(self, filename: str) -> bool:
        if not self.client.collection_exists(self.collection):
            return False
        res = self.client.count(
            self.collection,
            count_filter=Filter(must=[FieldCondition(key="filename", match=MatchValue(value=filename))]),
        )
        return res.count > 0

    def list_files(self) -> dict[str, int]:
        """Return {filename: chunk_count} across the collection (best-effort)."""
        counts: dict[str, int] = {}
        if not self.client.collection_exists(self.collection):
            return counts
        offset = None
        while True:
            points, offset = self.client.scroll(
                self.collection, with_payload=["filename"], limit=1000, offset=offset,
            )
            for p in points:
                fn = (p.payload or {}).get("filename", "?")
                counts[fn] = counts.get(fn, 0) + 1
            if offset is None:
                break
        return counts


@lru_cache(maxsize=1)
def get_store() -> VectorStore:
    return VectorStore()
