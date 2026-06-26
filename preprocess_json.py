import requests
import os
import json
import pandas as pd
import joblib


def create_embedding(text_list):
    r = requests.post(
        "http://localhost:11434/api/embed",
        json={"model": "bge-m3", "input": text_list}
    )
    response = r.json()
    if "error" in response:
        raise Exception(response["error"])
    if "embeddings" not in response:
        raise Exception(f"Unexpected response: {response}")
    return response["embeddings"]


def batch_embed(texts, batch_size=50):
    all_embeddings = []
    total_batches = (len(texts) + batch_size - 1) // batch_size
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        print(f"  Embedding batch {i // batch_size + 1}/{total_batches}")
        all_embeddings.extend(create_embedding(batch))
    return all_embeddings


# FIX: Merge small Whisper segments into larger, meaningful chunks
# Each merged chunk will cover ~MERGE_WINDOW_SECONDS of audio
MERGE_WINDOW_SECONDS = 30  # tune this — 20-45s works well for lectures

def merge_segments(segments, window_seconds=MERGE_WINDOW_SECONDS):
    """Merge consecutive Whisper segments into chunks of ~window_seconds each."""
    if not segments:
        return []

    merged = []
    current_texts = []
    current_start = segments[0]["start"]
    current_end = segments[0]["end"]

    for seg in segments:
        if seg["end"] - current_start <= window_seconds:
            # Still within the window — accumulate
            current_texts.append(seg["text"].strip())
            current_end = seg["end"]
        else:
            # Window full — save current chunk and start a new one
            merged.append({
                "start": current_start,
                "end": current_end,
                "text": " ".join(current_texts)
            })
            current_texts = [seg["text"].strip()]
            current_start = seg["start"]
            current_end = seg["end"]

    # Save the last chunk
    if current_texts:
        merged.append({
            "start": current_start,
            "end": current_end,
            "text": " ".join(current_texts)
        })

    return merged


jsons = os.listdir("jsons")
my_dicts = []
chunk_id = 0

for json_file in jsons:
    print(f"\nProcessing: {json_file}")

    with open(f"jsons/{json_file}", "r", encoding="utf-8") as f:
        content = json.load(f)

    if "chunks" not in content:
        print(f"  Skipping: no 'chunks' key found")
        continue

    # Extract number and title from filename (e.g. "2_Lecture2.mp3.json")
    base = json_file.replace(".json", "")
    number = base.split("_")[0] if "_" in base else "?"
    title = base.split("_")[1] if "_" in base else base

    raw_segments = [c for c in content["chunks"] if "text" in c and isinstance(c["text"], str)]
    print(f"  Raw segments: {len(raw_segments)}")

    # Merge into larger chunks
    merged = merge_segments(raw_segments, window_seconds=MERGE_WINDOW_SECONDS)
    print(f"  Merged chunks ({MERGE_WINDOW_SECONDS}s window): {len(merged)}")

    if not merged:
        continue

    texts = [chunk["text"] for chunk in merged]
    print(f"  Avg chunk length: {sum(len(t) for t in texts) // len(texts)} chars")

    embeddings = batch_embed(texts)

    for chunk, embedding in zip(merged, embeddings):
        my_dicts.append({
            "chunk_id": chunk_id,
            "number": number,
            "title": title,
            "start": chunk["start"],
            "end": chunk["end"],
            "text": chunk["text"],
            "embedding": embedding
        })
        chunk_id += 1

print(f"\nTotal merged chunks: {len(my_dicts)}")
df = pd.DataFrame.from_records(my_dicts)
joblib.dump(df, "embeddings.joblib")
print("Saved embeddings.joblib")
print(df[["number", "title", "start", "end"]].head(10))