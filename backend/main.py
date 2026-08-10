from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import shutil
import os
import requests
from groq import Groq

# -----------------------------
# 🔐 Load ENV
# -----------------------------
load_dotenv()

app = FastAPI()

# -----------------------------
# 🌐 CORS
# -----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# ⚡ GROQ
# -----------------------------
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# -----------------------------
# 🗄️ Hasura (optional)
# -----------------------------
HASURA_URL = "http://localhost:8080/v1/graphql"
MEETING_ID = "5777021f-15f4-4411-adfb-9d1463c317f2"

# -----------------------------
# 🧠 Memory
# -----------------------------
last_transcript = ""

# -----------------------------
# 🧠 PROMPT
# -----------------------------
PROMPT = """
You are an AI meeting assistant.

Return STRICTLY in this format:

Summary:

- one clear concise summary

Action Items:

- only factual next steps mentioned or implied
- do NOT assume decisions
- do NOT add opinions

Keep it neutral and professional.
"""

# -----------------------------
# 🧠 PARSE OUTPUT
# -----------------------------
def parse_output(output):
    try:
        parts = output.split("Action Items:")
        summary = parts[0].replace("Summary:", "").strip()

        actions = []
        if len(parts) > 1:
            actions = parts[1].strip().split("\n-")
            actions = [a.strip("- ").strip() for a in actions if a.strip()]

        return summary, actions
    except:
        return output, []

# -----------------------------
# 🎤 GROQ TRANSCRIPTION
# -----------------------------
def transcribe_audio(file_path):
    with open(file_path, "rb") as f:
        transcription = groq_client.audio.transcriptions.create(
            file=f,
            model="whisper-large-v3"
        )
    return transcription.text

# -----------------------------
# ⚡ GROQ SUMMARY
# -----------------------------
def summarize_with_groq(text):
    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": PROMPT},
            {"role": "user", "content": text}
        ],
        temperature=0.3
    )
    return response.choices[0].message.content

# -----------------------------
# 🎤 UPLOAD API
# -----------------------------
@app.post("/upload-audio/")
async def upload_audio(file: UploadFile = File(...)):
    global last_transcript

    try:
        file_location = f"temp_{file.filename}"

        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        print("✅ File saved")

        # 🔥 GROQ transcription
        text = transcribe_audio(file_location)
        last_transcript = text

        if os.path.exists(file_location):
            os.remove(file_location)

        print("📝 Transcript:", text)

        # 🔥 Summary
        raw_summary = summarize_with_groq(text)

        summary_text, action_items = parse_output(raw_summary)

        return {
            "transcript": text,
            "summary": summary_text,
            "action_items": action_items
        }

    except Exception as e:
        print("❌ ERROR:", str(e))
        return {"error": str(e)}

# -----------------------------
# 💬 Q&A API
# -----------------------------
@app.post("/ask/")
async def ask_question(question: str):
    global last_transcript

    if not last_transcript:
        return {"answer": "No transcript available. Upload audio first."}

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": "Answer ONLY using the transcript. If not found, say 'Not mentioned in transcript'."
                },
                {
                    "role": "user",
                    "content": f"Transcript:\n{last_transcript}\n\nQuestion: {question}"
                }
            ],
            temperature=0.2
        )

        answer = response.choices[0].message.content
        return {"answer": answer}

    except Exception as e:
        return {"error": str(e)}