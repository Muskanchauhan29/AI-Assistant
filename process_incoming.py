import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import requests
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)


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


def inference(prompt):
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {
                # FIX 1: System prompt sets strict rules upfront
                "role": "system",
                "content": (
                    "You are a helpful assistant for a machine learning and AI course. "
                    "You ONLY answer based on the video subtitle chunks provided by the user. "
                    "You MUST NOT mention, reference, or guess about any lecture, topic, or timestamp "
                    "that is not explicitly present in the provided chunks. "
                    "If the chunks do not contain enough information to answer, say so clearly."
                )
            },
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content


df = joblib.load('embeddings.joblib')

incoming_query = input("Ask a Question: ")
question_embedding = create_embedding([incoming_query])[0]

embeddings_matrix = np.vstack(df['embedding'])
similarities = cosine_similarity(embeddings_matrix, [question_embedding]).flatten()

# FIX 2: Also filter by a minimum similarity threshold to avoid retrieving irrelevant chunks
top_result = 5
similarity_threshold = 0.3  # tune this value based on your data

top_indices = similarities.argsort()[::-1][:top_result]
top_indices = [i for i in top_indices if similarities[i] >= similarity_threshold]

if not top_indices:
    print("No sufficiently relevant chunks found for your query.")
    exit()

new_df = df.iloc[top_indices].copy()
new_df['similarity_score'] = [round(float(similarities[i]), 4) for i in top_indices]

# FIX 3: Show the similarity scores so you can debug retrieval quality
print("\n--- Retrieved chunks (with similarity scores) ---")
for _, row in new_df.iterrows():
    print(f"  [{row['similarity_score']:.3f}] Lecture {row['number']} | {row['start']:.0f}s-{row['end']:.0f}s | {row['text'][:80]}...")

# FIX 4: Rewritten prompt — forces the model to address EACH chunk explicitly
#         and forbids it from going beyond what's in the data
chunks_json = new_df[["title", "number", "text", "start", "end", "similarity_score"]].to_json(orient="records", indent=2)

prompt = f"""I am teaching Machine Learning and Artificial Intelligence. Below are {len(new_df)} subtitle chunks retrieved from course videos. Each chunk includes the video title, lecture number, start/end timestamps (in seconds), the spoken text, and a similarity score showing how relevant it is to the query.

RETRIEVED CHUNKS:
{chunks_json}

---
STUDENT QUESTION: {incoming_query}

INSTRUCTIONS:
1. Go through EACH of the {len(new_df)} chunks above one by one.
2. For each chunk, decide if it is relevant to the student's question. State your reasoning briefly.
3. For relevant chunks, tell the student which lecture and timestamp to go to (convert seconds to MM:SS format).
4. ONLY use information from the chunks above. Do NOT mention any lecture, topic, or timestamp not present in the chunks.
5. If none of the chunks are relevant, tell the student that this topic was not found in the retrieved content and suggest rephrasing.
6. If the student's question is completely unrelated to ML/AI course content, politely say you can only answer course-related questions.
"""

with open("prompt.txt", "w") as f:
    f.write(prompt)

response = inference(prompt)
print("\n--- Answer ---")
print(response)