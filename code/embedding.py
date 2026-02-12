import json
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import faiss


def build_index(json_files, index_path, mapping_path, model_name="sentence-transformers/all-mpnet-base-v2", batch_size=128):
    """
    Build a FAISS index from element JSON files.
    
    Args:
        json_files: List of paths to element JSON files
        index_path: Output path for the FAISS index
        mapping_path: Output path for the ID mapping JSON
        model_name: Sentence transformer model name
        batch_size: Encoding batch size
    """
    model = SentenceTransformer(model_name)

    print("🔍 Loading JSON...")
    elements = []
    for json_file in json_files:
        with open(json_file, "r", encoding="utf-8") as f:
            elements += json.load(f)

    id_mapping = [e["element_id"] for e in elements]
    prompts = [e["prompt"] for e in elements]
    print(f"✅ Total {len(elements)} elements")

    print("⚡ Generating embeddings...")
    all_embeddings = []
    for i in tqdm(range(0, len(prompts), batch_size)):
        batch = prompts[i:i+batch_size]
        emb = model.encode(batch, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
        all_embeddings.append(emb)

    all_embeddings = np.vstack(all_embeddings).astype("float32")
    print("✅ Embedding generation complete:", all_embeddings.shape)

    dim = all_embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(all_embeddings)

    faiss.write_index(index, index_path)
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(id_mapping, f, ensure_ascii=False, indent=2)

    print(f"🎉 Complete! Index saved to {index_path}, mapping saved to {mapping_path}")


if __name__ == "__main__":
    # Example: build index from element JSON files
    JSON_FILE = "all_elements.json"
    JSON_FILE_2 = "crello_extra_elements.json"
    INDEX_PATH = "crello_public_and_extra_elements_local.index"
    MAPPING_PATH = "crello_public_and_extra_id_mapping_local.json"

    build_index(
        json_files=[JSON_FILE, JSON_FILE_2],
        index_path=INDEX_PATH,
        mapping_path=MAPPING_PATH,
    )


