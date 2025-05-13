import faiss
import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import Dict, List, Tuple

class UserRequestCache:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", threshold: float = 0.75, max_history: int = 5, storage_path: str = "./user_faiss/"):
        self.model = SentenceTransformer(model_name)
        self.threshold = threshold
        self.max_history = max_history
        self.storage_path = storage_path
        self.user_cache: Dict[str, List[Dict]] = {}
        self.user_indexes: Dict[str, faiss.IndexFlatIP] = {}

        os.makedirs(self.storage_path, exist_ok=True)
        self._load_all_users()

    def _faiss_path(self, user_id: str):
        return os.path.join(self.storage_path, f"faiss_index_user_{user_id}.index")

    def _json_path(self, user_id: str):
        return os.path.join(self.storage_path, f"user_cache_{user_id}.json")

    def _build_user_index(self, user_id: str):
        print(f"Rebuilding FAISS index for user: {user_id}")
        requests = self.user_cache[user_id]
        embeddings = np.array([r["embedding"] for r in requests])
        faiss.normalize_L2(embeddings)
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        self.user_indexes[user_id] = index
        faiss.write_index(index, self._faiss_path(user_id))
        print(f"Saved new FAISS index for user: {user_id} at {self._faiss_path(user_id)}")

    def _save_user_cache(self, user_id: str):
        data = [
            {
                "request_text": r["request_text"],
                "entities": r["entities"]
            } for r in self.user_cache[user_id]
        ]
        with open(self._json_path(user_id), "w") as f:
            json.dump(data, f)

    def _load_user_cache(self, user_id: str):
        path = self._json_path(user_id)
        if not os.path.exists(path):
            return

        with open(path, "r") as f:
            raw_data = json.load(f)

        self.user_cache[user_id] = []
        for entry in raw_data:
            embedding = self.model.encode(entry["request_text"])
            self.user_cache[user_id].append({
                "request_text": entry["request_text"],
                "entities": entry["entities"],
                "embedding": embedding
            })

        if os.path.exists(self._faiss_path(user_id)):
            self.user_indexes[user_id] = faiss.read_index(self._faiss_path(user_id))
            print(f"📂 Loaded FAISS index for user: {user_id} from {self._faiss_path(user_id)}")


    def _load_all_users(self):
        print("Loading all cached users...")
        for filename in os.listdir(self.storage_path):
            if filename.startswith("user_cache_") and filename.endswith(".json"):
                user_id = filename.replace("user_cache_", "").replace(".json", "")
                self._load_user_cache(user_id)

    def add_request(self, user_id: str, request_text: str, entities: Dict):
        print(f"Adding request to user {user_id}'s cache.")

        # Check for duplicate entry
        if any(r["request_text"] == request_text for r in self.user_cache.get(user_id, [])):
            print(f"Skipping duplicate request for user {user_id}.")
            return

        embedding = self.model.encode(request_text)
        entry = {
            "request_text": request_text,
            "entities": entities,
            "embedding": embedding
        }

        if user_id not in self.user_cache:
            self.user_cache[user_id] = []

        self.user_cache[user_id].append(entry)
        self.user_cache[user_id] = self.user_cache[user_id][-self.max_history:]

        self._build_user_index(user_id)
        self._save_user_cache(user_id)

    def search_user_history(self, user_id: str, new_request: str) -> Tuple[Dict, float]:
        if user_id not in self.user_cache or not self.user_cache[user_id]:
            print("No history for user. Use LLM.")
            return None, 0.0

        new_embedding = self.model.encode([new_request])
        faiss.normalize_L2(new_embedding)

        index = self.user_indexes[user_id]
        D, I = index.search(new_embedding, k=1)
        score = float(D[0][0])
        best_match = self.user_cache[user_id][I[0][0]]

        if score >= self.threshold:
            print(f"User cache hit for {user_id} (score: {score:.4f})")
            return best_match["entities"], score
        else:
            print(f"No good match in cache for {user_id} (score: {score:.4f})")
            return None, score


    def search_by_attribute(self, user_id: str, key: str, value: str) -> List[Dict]:
        if user_id not in self.user_cache or not self.user_cache[user_id]:
            print(f"⚠️ No cache found for user: {user_id}")
            return []

        results = []
        for entry in self.user_cache[user_id]:
            entity = entry.get("entities", {})
            if entity.get(key) and entity[key].lower() == value.lower():
                results.append(entity)

        if results:
            print(f"✅ Found {len(results)} match(es) for '{key}: {value}' in user {user_id}'s cache.")
        else:
            print(f"❌ No matches for '{key}: {value}' in user {user_id}'s cache.")
        
        return results


if __name__ == "__main__":
    user_cache = UserRequestCache()

    # User 1 requests
    user_cache.add_request(
        "user_1",
        "Upgrade required for core router at NY datacenter. This change is expected to impact service.",
        {
            "Activity Type": "Upgrade",
            "Network": "Core",
            "Service Impact": "Service Affecting",
            "Location": "NY datacenter"
        }
    )

    user_cache.add_request(
        "user_1",
        "Schedule reboot for aggregation switches. No service impact is expected.",
        {
            "Activity Type": "RESTART",
            "Network": "Aggregation",
            "Service Impact": "Non Service Affecting",
            "Location": "NO"
        }
    )

    # User 2 requests
    user_cache.add_request(
        "user_2",
        "Install new firmware on RAN nodes. Work is service affecting.",
        {
            "Activity Type": "Installation",
            "Network": "RAN",
            "Service Impact": "Service Affecting",
            "Location": "NO"
        }
    )

    user_cache.add_request(
        "user_2",
        "Non-service impacting maintenance scheduled on transport layer.",
        {
            "Activity Type": "MAINTENANCE",
            "Network": "TRANSPORT",
            "Service Impact": "Non Service Affecting",
            "Location": "NO"
        }
    )

    # Simulate new request for user_1
    new_request_user1 = "This upgrade will happen in the NY datacenter affecting the core network. It may cause brief outages."
    entities1, score1 = user_cache.search_user_history("user_1", new_request_user1)
    if entities1:
        print("User 1 matched entities:")
        print(json.dumps(entities1, indent=2))
    else:
        print("User 1 fallback to LLM")

    # Simulate new request for user_2
    new_request_user2 = "A new installation will be performed in the RAN segment. Expect service interruptions."
    entities2, score2 = user_cache.search_user_history("user_2", new_request_user2)
    if entities2:
        print("User 2 matched entities:")
        print(json.dumps(entities2, indent=2))
    else:
        print("User 2 fallback to LLM")

    print("\nAttribute search:")
    attribute_results = user_cache.search_by_attribute("user_2", "Network", "RAN")
    if attribute_results:
        print("Matching cached entities:")
        print(json.dumps(attribute_results, indent=2))
    else:
        print("No matching entities found by attribute")
