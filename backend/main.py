from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import whisper
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
# 🎤 Whisper
# -----------------------------
model = whisper.load_model("base")

# -----------------------------
# ⚡ Groq
# -----------------------------
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# -----------------------------
# 🗄️ Hasura
# -----------------------------
HASURA_URL = "http://localhost:8080/v1/graphql"
MEETING_ID = "5777021f-15f4-4411-adfb-9d1463c317f2"

# -----------------------------
# 🧠 Memory
# -----------------------------
last_transcript = ""

# -----------------------------
# 🧠 PROMPT (FINAL)
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
# 🧠 OLLAMA FALLBACK
# -----------------------------
def summarize_with_ollama(text):
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3",
            "prompt": f"{PROMPT}\n\n{text}",
            "stream": False
        },
        timeout=10
    )
    return response.json()["response"]

# -----------------------------
# 🎤 UPLOAD API
# -----------------------------
@app.post("/upload-audio/")
async def upload_audio(file: UploadFile = File(...)):
    global last_transcript

    try:
        file_location = f"temp_{file.filename}"

        # Save file
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        print("✅ File saved")

        # Whisper transcription
        result = model.transcribe(file_location)
        text = result["text"]
        last_transcript = text

        # Remove file safely
        if os.path.exists(file_location):
            os.remove(file_location)

        print("📝 Transcript:", text)

        # 🔥 Groq → Ollama fallback
        try:
            raw_summary = summarize_with_groq(text)
            print("⚡ Used Groq")

        except Exception as e:
            print("❌ Groq failed:", e)

            try:
                raw_summary = summarize_with_ollama(text)
                print("🧠 Used Ollama")

            except Exception as ollama_error:
                print("❌ Ollama also failed:", ollama_error)
                raw_summary = "Summary:\n- AI unavailable\n\nAction Items:\n- Try again later"

        # Parse summary
        summary_text, action_items = parse_output(raw_summary)

        # -----------------------------
        # 💾 Save to Hasura
        # -----------------------------
        query = """
        mutation ($text: String!, $meeting_id: uuid!) {
          insert_transcripts_one(object: {
            meeting_id: $meeting_id,
            content: $text
          }) {
            id
          }
        }
        """

        db_response = requests.post(
            HASURA_URL,
            json={
                "query": query,
                "variables": {
                    "text": text,
                    "meeting_id": MEETING_ID
                }
            },
            timeout=5
        )

        return {
            "transcript": text,
            "summary": summary_text,
            "action_items": action_items,
            "db": db_response.json()
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
        # ⚡ Groq
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
            print("⚡ Groq Q&A")

        except Exception as e:
            print("❌ Groq failed:", e)

            try:
                response = requests.post(
                    "http://localhost:11434/api/generate",
                    json={
                        "model": "llama3",
                        "prompt": f"Transcript:\n{last_transcript}\n\nQuestion: {question}",
                        "stream": False
                    },
                    timeout=10
                )

                answer = response.json()["response"]
                print("🧠 Ollama Q&A")

            except Exception:
                answer = "⚠️ AI services unavailable"

        return {"answer": answer}

    except Exception as e:
        return {"error": str(e)}