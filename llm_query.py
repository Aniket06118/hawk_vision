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

summaries = [
    """The video presents a visual collage that juxtaposes text and imagery to convey a playful competition theme. A graphic featuring the text "TUBER CHALLENGE" in a bold sans-serif typeface and surrounded by a smoky effect sets the stage for a challenge. The background is a vibrant blue, with the text and graphic standing out in stark contrast. This is followed by a close-up of a detailed, symmetrical snowflake illuminated by a bright light source, adding a touch of wintery wonder. The scene then shifts to a park setting where two young children are engaged in a playful game of catch, using a large orange ball. The interaction between the children, their contrasting outfits and the dynamic nature of the game contribute to the playful and energetic atmosphere of the video.""",

    """The video depicts a series of scenes featuring children and babies engaged in various actions. A park setting with a ball and cone is presented, showing two children throwing and catching a ball. A close-up shot reveals a young man holding the hand of a child, both standing on a black floor. Three subsequent frames offer snapshots of babies crawling on a carpeted surface. The babies are in their infancy, sporting white diapers, and are displayed next to a plain, dark wall. Finally, a baby, wearing a white diaper, is shown on a black surface, possibly a carpet or mat, with a light-colored wall in the background.""",

    """The video showcases a series of scenes depicting children engaged in various forms of physical activity. It opens with a young child, wearing a white diaper and slightly turning left while walking or running on a dark floor. The scene shifts to the child on a treadmill, positioned on exercise equipment. The next scene displays a baby crawling on a rubber mat in what appears to be a fitness center with exercise machines in the background. The video then transitions to a young boy playing on a playground slide. The image transitions to a child enjoying a pool float inside a large inflatable blue and white pool.""",

    """The video presents a montage of different children participating in various activities. A young boy is engaged in a playful slide activity. In another scene, a person glides down a blue inflatable water slide. As the scene shifts, a child runs on a paved road surrounded by green vegetation. Finally, a young person wearing a helmet is shown pedaling a bicycle through a residential street with a backpack.""",

    """The video showcases a diverse range of activities taking place in different outdoor settings. Two children interact with a tricycle and wagon. A person rides a bicycle along a dirt road with greenery. Another person is on a motorized scooter in a grassy area. The video transitions to a group engaged in a physical or sports activity. The final scene captures a young child standing on a wooden dock looking toward the water.""",

    """The video captures snapshots of children and birds interacting in a waterfront setting. A child splashes water from a dock while holding an object. A girl kneels beside a hawk perched on the dock. A group of people engage in fishing or boating. A solitary girl stands in a grassy field surrounded by trees. Natural elements like water, trees, and fences create a peaceful atmosphere.""",

    """The video showcases a webpage detailing the 'Tuber Challenge,' a fitness challenge encouraging physical activity. The webpage features a clean modern design with green and blue colors. The video transitions to a person running on a grassy field wearing dark clothing. Finally, a red bicycle is shown moving on a paved surface, suggesting a race or competitive event.""",

    """A red sedan with license plate 'BK 83 AJ' is parked on a paved road in a suburban setting. A person wearing a white shirt and dark pants walks along the road carrying a backpack. A young child stands in a grassy area wearing a striped shirt and dark pants, looking down at an object in their hands.""",

    """The video follows a young child's journey through different settings. The child stands on grass wearing a striped shirt and pants. Later the child appears older in different clothing. The child is barefoot and enjoys outdoor activities. The scene shows the child holding a red apple. Finally, the video transitions indoors where the child performs a dynamic airborne movement.""",

    """The video showcases moments of a young girl engaged in playful activities. She stands barefoot on a wooden floor wearing a white tank top and denim shorts. Another scene shows a person sitting on a couch using a smartphone. A bedroom scene shows someone sleeping under a blanket. The video transitions to a girl playing with a blue pool outdoors and smiling happily."""
]
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