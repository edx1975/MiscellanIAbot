import os
import json
import numpy as np
import faiss
from openai import OpenAI
from flask import Flask, request
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters

# --- Configuració ---
CORPUS_FILE = "data/corpus_original.jsonl"
INDEX_FILE = "data/faiss_index_g_pro_large.bin"
DOCS_FILE = "data/embeddings_g_pro_large.npy"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
MODEL_EMB = "text-embedding-3-small"
MODEL_CHAT = "gpt-4o-mini"

client = OpenAI(api_key=OPENAI_API_KEY)
app = Flask(__name__)

# --- Carregar corpus i index Faiss ---
def load_docs():
    docs = []
    with open(CORPUS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            text = " ".join([
                doc.get("title", ""),
                doc.get("summary", ""),
                " ".join(doc.get("topics", []))
            ])
            docs.append(text)
    return docs

docs = load_docs()

if os.path.exists(DOCS_FILE) and os.path.exists(INDEX_FILE):
    embeddings = np.load(DOCS_FILE)
    index = faiss.read_index(INDEX_FILE)
else:
    raise RuntimeError("No hi ha embeddings ni index; genera'ls localment primer.")

# --- Funcions ---
def get_top_docs(query, top_k=5, max_tokens=3000):
    # Obtenir embedding de la query
    q_emb = client.embeddings.create(model=MODEL_EMB, input=query).data[0].embedding
    q_emb = np.array([q_emb], dtype=np.float32)

    # Cercar els top_k documents
    distances, indices = index.search(q_emb, top_k)
    top_docs = [docs[idx] for idx in indices[0]]

    # Truncar context si és massa llarg
    context = "\n\n".join(top_docs)
    tokens_est = len(context.split())
    if tokens_est > max_tokens:
        context = " ".join(context.split()[:max_tokens])
    return context

async def answer_question(query):
    context = get_top_docs(query)
    prompt = (
        f"Usant aquest context del corpus, respon la pregunta de manera clara:\n\n"
        f"Context:\n{context}\n\nPregunta: {query}\nResposta:"
    )
    response = client.chat.completions.create(
        model=MODEL_CHAT,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# --- Handlers Telegram ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Hola! Pregunta'm qualsevol cosa sobre el corpus.")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.message.text
    answer = await answer_question(query)
    await update.message.reply_text(answer)

# --- Configuració bot Telegram ---
telegram_app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
telegram_app.add_handler(CommandHandler("start", start))
telegram_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

# --- Flask webhook per Railway ---
@app.route(f"/{TELEGRAM_TOKEN}", methods=["POST"])
def webhook():
    update = Update.de_json(request.get_json(force=True), telegram_app)
    telegram_app.update_queue.put(update)
    return "ok", 200

@app.route("/")
def index():
    return "Bot actiu", 200

# --- Main ---
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
