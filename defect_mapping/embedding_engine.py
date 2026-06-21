"""
Sentence-Transformer embedding engine for semantic text matching.
"""

import numpy as np

import os
import warnings

# Suppress Hugging Face unauthenticated request warnings and telemetry
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
warnings.filterwarnings("ignore", message=".*unauthenticated requests.*")

try:
    from sentence_transformers import SentenceTransformer, util as st_util
    _SBERT_AVAILABLE = True
except ImportError:
    _SBERT_AVAILABLE = False


class EmbeddingEngine:

    DEFAULT_MODEL = "all-MiniLM-L6-v2"

    def __init__(self, model_name: str | None = None):
        """
        Load a pre-trained sentence-transformer model.

        Parameters
        ----------
        model_name : str, optional
            HuggingFace model name. Defaults to 'all-MiniLM-L6-v2'.
        """
        if not _SBERT_AVAILABLE:
            raise ImportError(
                "sentence-transformers is required.  "
                "Install with:  pip install sentence-transformers"
            )

        self.model_name = model_name or self.DEFAULT_MODEL
        print(f"  Loading sentence-transformer model: {self.model_name} ...")
        self.model = SentenceTransformer(self.model_name)
        self._dim = self.model.get_embedding_dimension()
        print(f"  [OK] Model loaded  ({self._dim}-dim embeddings)")

    @property
    def embedding_dim(self) -> int:
        """Dimensionality of the embedding vectors."""
        return self._dim

    # ── Core Encoding ───────────────────────────────────────────────

    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        """
        Encode a list of texts into embedding vectors.

        Parameters
        ----------
        texts : list[str]
            Texts to encode.
        batch_size : int
            Batch size for encoding (higher = faster, more memory).

        Returns
        -------
        np.ndarray
            Matrix of shape (len(texts), embedding_dim).
        """
        if not texts:
            return np.empty((0, self._dim), dtype=np.float32)

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,  # L2-normalised for cosine sim
        )
        return embeddings

    # ── Similarity ──────────────────────────────────────────────────

    def similarity(self, text_a: str, text_b: str) -> float:
        """
        Compute cosine similarity between two texts.

        Returns
        -------
        float
            Similarity score between 0.0 (unrelated) and 1.0 (identical meaning).
        """
        embs = self.encode([text_a, text_b])
        sim = float(np.dot(embs[0], embs[1]))
        # Clamp to [0, 1] — negative cosine sim means anti-correlated,
        # which is semantically "not related" for our purposes.
        return max(0.0, min(1.0, sim))

    def batch_similarity(
        self,
        queries: list[str],
        corpus: list[str],
    ) -> np.ndarray:
        """
        Compute pairwise cosine similarity between query texts and corpus texts.

        Parameters
        ----------
        queries : list[str]
            Query texts (e.g. defect descriptions).
        corpus : list[str]
            Corpus texts (e.g. failure messages).

        Returns
        -------
        np.ndarray
            Similarity matrix of shape (len(queries), len(corpus)).
            Entry [i][j] is the cosine similarity between queries[i] and corpus[j].
        """
        if not queries or not corpus:
            return np.empty((len(queries), len(corpus)), dtype=np.float32)

        q_embs = self.encode(queries)
        c_embs = self.encode(corpus)

        # Dot product of L2-normalised vectors = cosine similarity
        sim_matrix = np.dot(q_embs, c_embs.T)
        # Clamp to [0, 1]
        return np.clip(sim_matrix, 0.0, 1.0)
