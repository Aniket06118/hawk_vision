# Hawk Vision: Real-Time Video Intelligence System

## Problem Statement

In modern surveillance systems, organizations and individuals often deal with **hours or even days of continuous CCTV footage**. When a specific incident needs to be investigated—such as identifying suspicious activity, confirming whether an event occurred, or locating a particular moment—manually reviewing the entire video is extremely time-consuming, inefficient, and often impractical.

Traditional systems require users to:

* Scrub through long video timelines
* Watch footage manually
* Spend significant time searching for specific events
* Risk missing important moments due to human fatigue

This creates a major challenge in surveillance, monitoring, and incident analysis workflows.

**Hawk Vision solves this problem by transforming passive video footage into an interactive, searchable system.** Instead of watching hours of video, users can simply **chat with the footage** and receive concise, intelligent responses about what happened, when it happened, and whether specific events occurred.

---

## Overview

**Hawk Vision** is a real-time video intelligence system that detects motion in video streams, analyzes relevant frames using a Vision-Language Model (VLM), stores semantic outputs temporarily, and generates concise summaries using a Large Language Model (LLM). The system is designed to be efficient, event-driven, and scalable for surveillance, monitoring, and situational awareness applications.

Instead of processing every frame continuously, Hawk Vision focuses on **motion-triggered analysis**, reducing computational overhead while maintaining meaningful situational understanding.

---

## Key Features

* Motion-based frame detection using OpenCV
* Vision-Language Model (VLM) inference for scene understanding
* Batch-based summarization using an LLM
* Temporary memory buffer for VLM outputs
* Automatic summarization after a fixed number of events
* Modular and extensible pipeline design

---

## System Architecture

```
Video Input
     ↓
Motion Detection (OpenCV)
     ↓
Frame Sampling
     ↓
Vision-Language Model (SmolVLM)
     ↓
Temporary Storage (Buffer)
     ↓
Batch Trigger (e.g., 10 events)
     ↓
LLM Summarization (Gemma)
     ↓
Output Summary
```

---

## Models Used

### Vision-Language Model (VLM)

**Model:** `HuggingFaceTB/SmolVLM-500M-Instruct`

Purpose:

* Understand visual scenes
* Generate textual descriptions of detected motion
* Lightweight and efficient for local inference

Why this model:

* Small and fast
* Suitable for real-time applications
* Runs on consumer GPUs

---

### Summarization Model (LLM)

**Model:** `google/gemma-2-2b-it`

Purpose:

* Summarize multiple VLM outputs
* Generate concise situational reports
* Reduce noise from raw event descriptions

---

### RAG Pipeline

**Provider:** Groq

Purpose:

* Fast inference for retrieval-augmented generation
* Low-latency response handling
* Scalable model serving

---

## Technologies Used

* Python
* OpenCV
* NumPy
* Hugging Face Transformers
* Groq API
* PIL (Python Imaging Library)
* dotenv

---

## Project Workflow

1. Load video from local path
2. Detect motion using frame differencing
3. Capture frames when motion is detected
4. Send frames to the Vision-Language Model
5. Store generated descriptions in memory
6. When buffer reaches threshold (e.g., 10 outputs):

   * Send outputs to LLM
   * Generate summary
   * Clear buffer

---

---

## Installation

### 1. Clone the Repository

```
git clone https://github.com/your-username/hawk-vision.git
cd hawk-vision
```

### 2. Create Virtual Environment

```
python -m venv venv
```

Windows:

```
venv\Scripts\activate
```

Linux / Mac:

```
source venv/bin/activate
```

### 3. Install Dependencies

```
pip install -r requirements.txt
```

---

## Environment Variables

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_api_key_here
HF_TOKEN=your_huggingface_token
```

Make sure `.env` is included in `.gitignore`.

---

## Running the Project

```
python main.py
```

Make sure to update the video path in the script:

```
video_path = r"C:\\path\\to\\your\\video.mp4"
```

---

## Project Structure

```
hawk-vision/
│
├── main.py
├── vlm.py
├── summarize.py
├── requirements.txt
├── .env
├── .gitignore
└── README.md
```

---

## Performance Considerations

* Motion-triggered processing reduces unnecessary inference
* Batch summarization reduces LLM calls
* Lightweight VLM improves real-time performance
* Designed to run on consumer GPUs (e.g., RTX 4060)

---

## Future Improvements

* Real-time camera stream support
* Database logging
* Alert generation system
* Web dashboard
* Multi-camera support
* Edge deployment optimization

---

## Use Cases

* Smart surveillance
* Campus monitoring
* Traffic monitoring
* Industrial safety
* Event detection

---

