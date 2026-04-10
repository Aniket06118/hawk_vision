import torch
from transformers import pipeline


llm_pipe = pipeline(
    "text-generation",
    model="google/gemma-2-2b-it",
    model_kwargs={"torch_dtype": torch.bfloat16},
    device="cuda",  # replace with "mps" to run on a Mac device
)


def summarize(responses):
    frames_text = "\n".join(f"Frame {i+1}: {r}" for i, r in enumerate(responses))
    messages = [
        {
            "role": "user",
            "content": f"""You are a video analysis assistant. Below are detailed frame-by-frame descriptions from a video. 
Write one detailed paragraph describing exactly what is happening in the video — who is present, what specific actions are taking place, how the scene evolves over time, and what objects or environment are involved. 
Be concrete and specific. Do not be vague or generic. Do not use bullet points, frame numbers, or headers.

Frame descriptions:
{frames_text}"""
        }
    ]
    out = llm_pipe(messages, max_new_tokens=512)
    return out[0]["generated_text"][-1]["content"].strip()