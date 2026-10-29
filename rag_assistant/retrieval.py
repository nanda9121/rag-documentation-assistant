"""Gemini embedding-backed in-memory cosine-similarity search."""

from dataclasses import dataclass
import math
from typing import Protocol, Sequence

from google import genai

from rag_assistant.documents import DocumentChunk


class Embedder(Protocol):
    def embed_documents(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        """Return one vector per document passage."""

    def embed_query(self, query: str) -> Sequence[float]:
        """Return a vector for a search query."""


class GeminiEmbedder:
    def __init__(
        self,
        api_key: str,
        model_name: str = "gemini-embedding-2",
    ):
        if not api_key.strip():
            raise ValueError("Set GEMINI_API_KEY in the .env file before indexing documents.")
        self._client = genai.Client(api_key=api_key)
        self._model_name = model_name

    def _embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            response = self._client.models.embed_content(
                model=self._model_name,
                contents=text,
            )
            embeddings = response.embeddings or []
            if len(embeddings) != 1 or not embeddings[0].values:
                raise RuntimeError(
                    "Gemini returned an invalid embedding for one document passage."
                )
            vectors.append(list(embeddings[0].values))
        return vectors

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        formatted = [f"title: none | text: {text}" for text in texts]
        return self._embed(formatted)

    def embed_query(self, query: str) -> list[float]:
        vectors = self._embed([f"task: question answering | query: {query}"])
        if not vectors or not vectors[0]:
            raise RuntimeError("Gemini returned an empty query embedding.")
        return vectors[0]


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: DocumentChunk
    score: float


class VectorIndex:
    def __init__(self, embedder: Embedder):
        self._embedder = embedder
        self._chunks: list[DocumentChunk] = []
        self._vectors: list[list[float]] = []

    @property
    def size(self) -> int:
        return len(self._chunks)

    def add_documents(self, chunks: Sequence[DocumentChunk]) -> None:
        if not chunks:
            return
        vectors = self._embedder.embed_documents([chunk.content for chunk in chunks])
        if len(vectors) != len(chunks):
            raise ValueError("The embedder must return one vector per document chunk")
        if any(not vector for vector in vectors):
            raise ValueError("The embedder must return non-empty vectors")
        if any(len(vector) != len(vectors[0]) for vector in vectors):
            raise ValueError("Document embedding dimensions do not match")
        if self._vectors and any(
            len(vector) != len(self._vectors[0]) for vector in vectors
        ):
            raise ValueError("Document embedding dimensions do not match")
        self._chunks.extend(chunks)
        self._vectors.extend([list(vector) for vector in vectors])

    def search(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        if not query.strip():
            raise ValueError("Search query cannot be empty")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if not self._chunks:
            return []

        query_vector = list(self._embedder.embed_query(query))
        if not query_vector:
            raise ValueError("The embedder must return a non-empty query vector")
        if any(len(vector) != len(query_vector) for vector in self._vectors):
            raise ValueError("Query and document embedding dimensions do not match")
        query_norm = math.sqrt(sum(value * value for value in query_vector))
        if query_norm == 0:
            raise ValueError("The embedder returned a zero-length query vector")

        ranked = []
        for chunk, vector in zip(self._chunks, self._vectors):
            vector_norm = math.sqrt(sum(value * value for value in vector))
            score = (
                sum(a * b for a, b in zip(vector, query_vector))
                / (vector_norm * query_norm)
                if vector_norm
                else 0.0
            )
            ranked.append(RetrievedChunk(chunk, score))
        return sorted(ranked, key=lambda result: result.score, reverse=True)[:top_k]
