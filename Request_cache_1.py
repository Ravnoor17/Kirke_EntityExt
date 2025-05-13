import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import Dict, List
import re
import json

# Step 1: Setup model and cache structure
model = SentenceTransformer("all-MiniLM-L6-v2")

# Step 2: Build the sentence cache
raw_cache = [
    {
        "sentence": "This will cause no impact.",
        "entities": {"Service Impact": "Non Service Affecting"}
    },
    {
        "sentence": "The activity will start on 2025-05-10 at 6:00 PM.",
        "entities": {"Scheduled Start Date": "2025-05-10 18:00"}
    },
    {
        "sentence": "Change is categorized as break.",
        "entities": {"Change Type": "Break"}
    }
]

# Step 3: Encode and build FAISS index
for entry in raw_cache:
    entry["embedding"] = model.encode(entry["sentence"])

dimension = len(raw_cache[0]["embedding"])
index = faiss.IndexFlatIP(dimension)

# Normalize vectors for cosine similarity
embeddings = np.array([entry["embedding"] for entry in raw_cache])
faiss.normalize_L2(embeddings)
index.add(embeddings)

# Step 4: Incoming request to process
new_request = """
This will cause no impact.Upgrade is expected to complete by midnight.Change is categorized as break.The activity will start on 2025-05-10 at 6:00 PM.Please ensure everything is completed on time.
"""

# Step 5: Sentence-level tokenization
def sentence_tokenize(text: str) -> List[str]:
    return [s.strip() for s in re.split(r'(?<=[.!?])(?=\s*[A-Z])', text) if s.strip()]

sentences = sentence_tokenize(new_request)
found_entities = {}

for sentence in sentences:
    vec = model.encode([sentence])
    faiss.normalize_L2(vec)
    D, I = index.search(vec, k=1)

    if D[0][0] > 0.85:  # threshold
        matched = raw_cache[I[0][0]]
        print(f"Matched: \"{sentence}\" → \"{matched['sentence']}\" (score: {D[0][0]:.4f})")
        found_entities.update(matched["entities"])
    else:
        print(f"No confident match for sentence: \"{sentence}\"")

# Step 6: Mock LLM Output
llm_output = {
    "Scheduled Start Date": "2025-05-10 at 6:00 PM",
    "Scheduled End Date": "NO",
    "Activity Type": "UPGRADE",
    "Network": "NO",
    "Subnetwork Name": "NO",
    "Service Impact": "NO Impact",
    "Change Type": "Break",
    "Network Elements": "NO",
    "Location": "NO"
}

# Step 7: Merge cached values into LLM result
for key in llm_output:
    if key in found_entities:
        llm_output[key] = found_entities[key]

# Step 8: Output result
print("\nFinal Merged Entity Output:")
print(json.dumps(llm_output, indent=2))
