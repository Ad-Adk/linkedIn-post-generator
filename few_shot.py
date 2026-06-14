import json
import numpy as np
import os
import pickle
import pandas as pd

EMBEDDING_CACHE = "data/embeddings_cache.pkl"


def _tfidf_embeddings(texts):
    """Lightweight fallback: TF-IDF bag-of-words vectors (no internet needed)."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(stop_words="english", max_features=512)
    matrix = vec.fit_transform(texts).toarray().astype(np.float32)
    return matrix, vec


class FewShotPosts:
    def __init__(self, file_path="data/expanded_posts.json"):
        self.df = None
        self.unique_tags = None
        self.model = None          # SentenceTransformer (if available)
        self.tfidf_vec = None      # fallback vectorizer
        self.embeddings = None
        self.use_semantic = False
        self.load_posts(file_path)

    def load_posts(self, file_path):
        with open(file_path, encoding="utf-8") as f:
            posts = json.load(f)
            self.df = pd.json_normalize(posts)
            self.df["length"] = self.df["line_count"].apply(self.categorize_length)
            all_tags = self.df["tags"].apply(lambda x: x).sum()
            self.unique_tags = sorted(list(set(all_tags)))
        self._load_or_build_embeddings(file_path)

    def _load_or_build_embeddings(self, data_path):
        cache_valid = (
            os.path.exists(EMBEDDING_CACHE)
            and os.path.getmtime(EMBEDDING_CACHE) >= os.path.getmtime(data_path)
        )
        if cache_valid:
            with open(EMBEDDING_CACHE, "rb") as f:
                cache = pickle.load(f)
            self.embeddings = cache["embeddings"]
            self.use_semantic = cache.get("use_semantic", False)
            self.tfidf_vec = cache.get("tfidf_vec", None)
            return

        texts = self.df["text"].tolist()

        # Try sentence-transformers first (best quality)
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
            self.embeddings = self.model.encode(texts, convert_to_numpy=True)
            self.use_semantic = True
            print("[FewShot] Using SentenceTransformer embeddings.")
        except Exception:
            # Fallback to TF-IDF (works offline, no extra download)
            self.embeddings, self.tfidf_vec = _tfidf_embeddings(texts)
            self.use_semantic = False
            print("[FewShot] SentenceTransformer unavailable. Using TF-IDF fallback.")

        os.makedirs("data", exist_ok=True)
        with open(EMBEDDING_CACHE, "wb") as f:
            pickle.dump({
                "embeddings": self.embeddings,
                "use_semantic": self.use_semantic,
                "tfidf_vec": self.tfidf_vec,
            }, f)

    def _embed_query(self, text: str) -> np.ndarray:
        if self.use_semantic:
            return self.model.encode([text], convert_to_numpy=True)[0]
        else:
            return self.tfidf_vec.transform([text]).toarray()[0].astype(np.float32)

    def categorize_length(self, line_count):
        if line_count < 5:
            return "Short"
        elif line_count <= 10:
            return "Medium"
        else:
            return "Long"

    def get_tags(self):
        return self.unique_tags

    def get_df(self):
        return self.df

    def get_filtered_posts(self, length, language, tag, top_k=2):
        """
        Semantic (or TF-IDF) retrieval with graceful fallback:
        1. Strict: matching language + length + tag, ranked by similarity
        2. Relax length: matching language + tag, any length
        3. Language-only fallback: best semantic match in same language
        4. Absolute fallback: top-k globally
        """
        query_embedding = self._embed_query(tag)

        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        normed = self.embeddings / (norms + 1e-9)
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-9)
        scores = normed @ query_norm

        ranked_df = self.df.copy()
        ranked_df["_score"] = scores

        def top(df_subset):
            return (
                df_subset.sort_values("_score", ascending=False)
                .head(top_k)
                .drop(columns=["_score"])
                .to_dict(orient="records")
            )

        # 1. Strict
        strict = ranked_df[
            (ranked_df["language"] == language)
            & (ranked_df["length"] == length)
            & (ranked_df["tags"].apply(lambda t: tag in t))
        ]
        if len(strict) >= top_k:
            return top(strict)

        # 2. Relax length
        medium = ranked_df[
            (ranked_df["language"] == language)
            & (ranked_df["tags"].apply(lambda t: tag in t))
        ]
        if len(medium) >= 1:
            return top(medium)

        # 3. Language-only semantic
        fallback = ranked_df[ranked_df["language"] == language]
        if len(fallback) >= 1:
            return top(fallback)

        # 4. Absolute
        return top(ranked_df)


if __name__ == "__main__":
    fs = FewShotPosts()
    print("Tags:", fs.get_tags())
    posts = fs.get_filtered_posts("Medium", "English", "Job Search")
    for p in posts:
        print("-", p["text"][:100])
        print("  tags:", p["tags"], "| length:", p["length"])
