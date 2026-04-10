from dotenv import load_dotenv
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq
import os

# -----------------------------
# 1. SET YOUR GROQ API KEY
# -----------------------------

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

# -----------------------------
# 2. YOUR CHUNKS (10 summaries)
# -----------------------------

summaries=[]
# -----------------------------
# 3. LOAD EMBEDDING MODEL
# -----------------------------

model = SentenceTransformer("all-MiniLM-L6-v2")

dimension = 384

index = faiss.IndexFlatL2(dimension)

documents = []

# -----------------------------
# 4. STORE CHUNKS
# -----------------------------

def store_chunks(chunk_list):

    for chunk in chunk_list:

        embedding = model.encode([chunk])

        index.add(np.array(embedding))

        documents.append(chunk)

    print(f"{len(chunk_list)} chunks stored.")


# -----------------------------
# 5. RETRIEVE RELEVANT CHUNKS
# -----------------------------

def retrieve(query, k=3):

    query_embedding = model.encode([query])

    distances, indices = index.search(
        np.array(query_embedding),
        k
    )

    results = []

    for i in indices[0]:

        if i < len(documents):
            results.append(documents[i])

    return results


# -----------------------------
# 6. ASK GROQ USING CONTEXT
# -----------------------------

def ask_groq(question, context):

    prompt = f"""
You are an assistant answering questions using the provided context.

Context:
{context}

Question:
{question}

Answer clearly using the context.
"""

    response = client.chat.completions.create(

        model="llama-3.3-70b-versatile",

        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],

        temperature=0.3
    )

    return response.choices[0].message.content


# -----------------------------
# 7. BUILD VECTOR DB
# -----------------------------

store_chunks(summaries)

# -----------------------------
# 8. CHAT LOOP
# -----------------------------

print("\nRAG system ready.")
print("Type 'exit' to stop.\n")

while True:

    question = input("Ask question: ")

    if question.lower() == "exit":
        print("Chat ended.")
        break

    context = retrieve(question)

    print("\nRetrieved context:")
    for c in context:
        print("-", c)

    answer = ask_groq(question, context)

    print("\nAnswer:")
    print(answer)
    print()