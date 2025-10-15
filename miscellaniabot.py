import os
import json
import logging
import numpy as np
import faiss
from pathlib import Path
from openai import OpenAI
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters

# ------------------------
# Config i paths
# ------------------------
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
METADATA_PATH = DATA_DIR / "corpus_original.jsonl"
EMB_PATH = DATA_DIR / "embeddings_G_pro_large.npy"
FAISS_INDEX_PATH = DATA_DIR / "faiss_index_G_pro_large.index"

# ------------------------
# Validacions
# ------------------------
if not TELEGRAM_TOKEN:
    raise ValueError("❌ Falta TELEGRAM_TOKEN (exporta'l amb `export TELEGRAM_TOKEN=...`)")

if not OPENAI_API_KEY:
    raise ValueError("❌ Falta OPENAI_API_KEY (exporta'l amb `export OPENAI_API_KEY=...`)")

# ------------------------
# Logging
# ------------------------
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# ------------------------
# Clients
# ------------------------
client = OpenAI(api_key=OPENAI_API_KEY)

# ------------------------
# Carregar dades
# ------------------------
if not EMB_PATH.exists() or not FAISS_INDEX_PATH.exists():
    raise RuntimeError("❌ No hi ha embeddings ni índex FAISS. Genera'ls primer.")

embeddings = np.load(EMB_PATH, allow_pickle=True)
index = faiss.read_index(str(FAISS_INDEX_PATH))

with open(METADATA_PATH, "r", encoding="utf-8") as f:
    metadata = [json.loads(line) for line in f]

logger.info(f"✅ Carregats {len(metadata)} documents i índex amb {index.ntotal} vectors.")

# ------------------------
# Funció d’embeddings (OpenAI)
# ------------------------
def query_to_embedding(query: str) -> np.ndarray:
    """Crea embedding normalitzat a partir d’un text amb OpenAI."""
    emb = client.embeddings.create(
        model="text-embedding-3-large",
        input=query.strip()
    )
    v = np.array(emb.data[0].embedding, dtype=np.float32)
    v /= np.maximum(np.linalg.norm(v), 1e-9)
    return v

# ------------------------
# Cerca semàntica
# ------------------------
def search(query: str, top_k: int = 3):
    """Cerca els documents més similars a la consulta."""
    vec = np.array([query_to_embedding(query)], dtype=np.float32)
    D, I = index.search(vec, top_k)
    results = [metadata[i] for i in I[0] if i < len(metadata)]
    return results

# ------------------------
# Handlers de Telegram
# ------------------------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("👋 Hola! Escriu'm una pregunta i cercaré al corpus per tu.")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.message.text.strip()
    if not query:
        await update.message.reply_text("❗ Escriu alguna cosa per cercar.")
        return

    try:
        results = search(query)
        if not results:
            await update.message.reply_text("No he trobat res rellevant.")
            return

        reply_parts = []
        for r in results:
            title = r.get("title", "Sense títol")
            summary = r.get("summary", r.get("text", ""))[:800]
            reply_parts.append(f"📄 *{title}*\n{summary}")

        reply = "\n\n".join(reply_parts)
        await update.message.reply_text(reply[:4000], parse_mode="Markdown")

    except Exception as e:
        logger.error(f"Error durant la cerca: {e}")
        await update.message.reply_text("⚠️ Hi ha hagut un error processant la consulta.")

# ------------------------
# Inicialitzar bot
# ------------------------
def main():
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    logger.info("🤖 Bot engegat i esperant missatges...")
    app.run_polling()

if __name__ == "__main__":
    main()
