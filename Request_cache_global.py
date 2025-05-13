import faiss
import numpy as np
import json
import os
from sentence_transformers import SentenceTransformer
from typing import Dict, List, Tuple

class GlobalRequestCache:
    def __init__(
        self,
        cached_requests: List[Dict],
        index_path: str = "request_index.faiss",
        metadata_path: str = "request_metadata.json",
        threshold: float = 0.75,
    ):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.threshold = threshold
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.index = None

        if os.path.exists(index_path) and os.path.exists(metadata_path):
            self._load_index()
        else:
            self.cached_requests = cached_requests
            self._build_index()
            self._save_index()

    def _build_index(self):
        print("Building FAISS index from cached requests...")
        for entry in self.cached_requests:
            entry["embedding"] = self.model.encode(entry["request_text"])

        emb_matrix = np.array([entry["embedding"] for entry in self.cached_requests])
        faiss.normalize_L2(emb_matrix)

        self.index = faiss.IndexFlatIP(emb_matrix.shape[1])
        self.index.add(emb_matrix)
        print("FAISS index built with", len(self.cached_requests), "requests.")

    def _save_index(self):
        print("Saving FAISS index and metadata...")
        faiss.write_index(self.index, self.index_path)

        # Don't save embeddings (too large) — just text and entities
        lightweight_metadata = [
            {
                "request_text": req["request_text"],
                "entities": req["entities"]
            }
            for req in self.cached_requests
        ]
        with open(self.metadata_path, "w") as f:
            json.dump(lightweight_metadata, f)

    def _load_index(self):
        print("Loading FAISS index and metadata...")
        self.index = faiss.read_index(self.index_path)

        with open(self.metadata_path, "r") as f:
            self.cached_requests = json.load(f)

    def search_similar(self, new_text: str) -> Tuple[Dict, float]:
        new_embedding = self.model.encode([new_text])
        faiss.normalize_L2(new_embedding)

        D, I = self.index.search(new_embedding, k=1)
        score = float(D[0][0])
        best_match = self.cached_requests[I[0][0]]

        if score >= self.threshold:
            print(f"Found match with similarity score: {score:.4f}")
            return best_match["entities"], score
        else:
            print(f"No good match (score: {score:.4f}). Use LLM.")
            return None, score

# -----------------------
# Usage Example
# -----------------------
if __name__ == "__main__":
    cached_requests = [
        {
            "request_text": "Please schedule an upgrade for the core router. This change is service affecting. Location is NY datacenter.",
            "entities": {
                "Activity Type": "Upgrade",
                "Network": "Core",
                "Service Impact": "Service Affecting",
                "Location": "NY datacenter"
            }
        },
        {
            "request_text": "Perform software installation in the RAN network with no expected service impact.",
            "entities": {
                "Activity Type": "Installation",
                "Network": "RAN",
                "Service Impact": "Non Service Affecting",
                "Location": "NO"
            }
        }
    ]

    # Initialize (will save index if not already present)
    global_cache = GlobalRequestCache(cached_requests)

    # New request
    new_request = "The change in this request may be service affecting. I want to perform a router upgrade in the NY datacenter on the core network."

    # Query
    entities, score = global_cache.search_similar(new_request)

    if entities:
        print(json.dumps(entities, indent=2))
    else:
        print("→ Fallback to LLM extraction.")
