import json
from matplotlib import text
import numpy as np
import re
from flask import render_template_string
from flask import Flask, request, jsonify

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json

from src.preprocess import clean_text
from src.responses import pick_response

MODEL_PATH = "models/emotion_model.keras"
TOKENIZER_PATH = "models/tokenizer.json"
LABELS_PATH = "models/labels.json"
MAX_LEN = 40

app = Flask(__name__)

model = tf.keras.models.load_model(MODEL_PATH)

def rule_override(text: str, predicted: str) -> str:
    t = text.lower()

    joy = [
    "happy",
    "happiness",
    "excited",
    "great",
    "amazing",
    "wonderful",
    "love",
    "positive",
    "confident"
]

    sadness = [
    "sad",
    "lonely",
    "miss",
    "cry",
    "depressed",
    "hopeless",
    "heartbroken",
    "disappointed"
]

    anxiety = [
    "stress",
    "stressed",
    "anxious",
    "anxiety",
    "worried",
    "worry",
    "panic",
    "overthinking",
    "nervous"
]

    anger = [
    "angry",
    "mad",
    "furious",
    "annoyed",
    "irritated",
    "hate",
    "frustrated"
]

    calm = [
    "calm",
    "relaxed",
    "peaceful",
    "fine",
    "okay",
    "ok",
    "chill",
    "at ease",
    "content"
]

    # word-boundary match for short words like "ok"
    def has(words):
        return any(re.search(rf"\b{re.escape(w)}\b", t) for w in words)


    if has(joy): return "joy"
    if has(anxiety): return "anxiety"
    if has(anger): return "anger"
    if has(sadness): return "sadness"
    if has(calm): return "calm"

    return predicted

with open(TOKENIZER_PATH, "r", encoding="utf-8") as f:
    tokenizer = tokenizer_from_json(f.read())

with open(LABELS_PATH, "r", encoding="utf-8") as f:
    labels = json.load(f)

def predict_emotion(text: str) -> str:
    cleaned = clean_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")
    probs = model.predict(padded, verbose=0)[0]
    return labels[int(np.argmax(probs))]

@app.get("/health")
def health():
    return jsonify({"status": "ok"})

@app.post("/chat")
def chat():
    body = request.get_json(force=True)
    text = body.get("text", "").strip()
    if not text:
        return jsonify({"error": "text is required"}), 400

    emotion = predict_emotion(text)
    emotion = rule_override(text, emotion)
    reply = pick_response(emotion)


    return jsonify({
        "input": text,
        "emotion": emotion,
        "reply": reply
    })

@app.get("/")
def ui():
    return render_template_string("""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Emotional Chatbot</title>
  <style>
    body { font-family: Arial, sans-serif; background:#f6f7fb; margin:0; }
    .wrap { max-width: 900px; margin: 24px auto; padding: 0 16px; }
    .card { background: white; border-radius: 14px; padding: 16px; box-shadow: 0 6px 24px rgba(0,0,0,0.08); }
    .title { display:flex; align-items:center; justify-content:space-between; gap:10px; }
    .badge { padding: 6px 10px; border-radius: 999px; background:#eef2ff; font-size: 12px; }
    #chat { height: 60vh; overflow:auto; padding: 12px; border: 1px solid #eee; border-radius: 12px; background:#fcfcff; }
    .msg { margin: 10px 0; display:flex; gap:10px; }
    .me { justify-content:flex-end; }
    .bubble { max-width: 75%; padding: 10px 12px; border-radius: 14px; line-height: 1.35; }
    .me .bubble { background:#2563eb; color:white; border-bottom-right-radius: 6px; }
    .bot .bubble { background:#e5e7eb; color:#111827; border-bottom-left-radius: 6px; }
    .meta { font-size: 12px; color:#6b7280; margin-top: 4px; }
    .row { display:flex; gap:10px; margin-top: 12px; }
    input { flex:1; padding: 12px; border-radius: 12px; border: 1px solid #ddd; outline:none; }
    button { padding: 12px 16px; border-radius: 12px; border: 0; background:#111827; color:white; cursor:pointer; }
    button:disabled { opacity:0.6; cursor:not-allowed; }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <div class="title">
        <h2 style="margin:0;">AI Emotional Assistance Chatbot</h2>
        <span class="badge">Flask + LSTM/GRU</span>
      </div>
      <p style="margin:8px 0 14px; color:#4b5563;">
        Type a message — the API returns detected emotion + supportive reply.
      </p>

      <div id="chat"></div>

      <div class="row">
        <input id="text" placeholder="Type here... (e.g., I am stressed about exams)" />
        <button id="send">Send</button>
      </div>

      <div class="meta" id="status"></div>
    </div>
  </div>

<script>
  const chat = document.getElementById("chat");
  const text = document.getElementById("text");
  const send = document.getElementById("send");
  const statusEl = document.getElementById("status");

function addMessage(role, message, emotion=null) {
  const row = document.createElement("div");
  row.className = "msg " + (role === "me" ? "me" : "bot");

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = message;

  row.appendChild(bubble);

  if (role === "bot" && emotion) {
    const meta = document.createElement("div");
    meta.className = "meta";
    meta.textContent = "Detected emotion: " + emotion;
    row.appendChild(meta);
  }

  chat.appendChild(row);
  chat.scrollTop = chat.scrollHeight;
}

  async function sendMessage() {
    const msg = text.value.trim();
    if (!msg) return;

    addMessage("me", msg);
    text.value = "";
    send.disabled = true;
    statusEl.textContent = "Thinking...";

    try {
      const res = await fetch("/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: msg })
      });

      if (!res.ok) {
        const t = await res.text();
        throw new Error("HTTP " + res.status + " - " + t);
      }

      const data = await res.json();
      addMessage("bot", data.reply || "(no reply)", data.emotion || "unknown");
      statusEl.textContent = "";
    } catch (err) {
      addMessage("bot", "Error: " + err.message);
      statusEl.textContent = "";
    } finally {
      send.disabled = false;
      text.focus();
    }
  }

  send.addEventListener("click", sendMessage);
  text.addEventListener("keydown", (e) => {
    if (e.key === "Enter") sendMessage();
  });

  // Welcome message
  addMessage("bot", "Hi! Tell me how you're feeling today 🙂");
  text.focus();
</script>
<footer style="
  text-align:center;
  margin-top:20px;
  padding-top:10px;
  border-top:1px solid #e5e7eb;
  font-size:13px;
  color:#6b7280;
">
  © 2026 Ishwar Birari · Built with Flask & LSTM/GRU
</footer>

</body>
</html>
""")

if __name__ == "__main__":
    app.run(port=5000, debug=True)
