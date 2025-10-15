import os
import json
import logging
import numpy as np
from pathlib import Path
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters

# ------------------------
# Config i paths
# ------------------------
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
METADATA_PATH = DATA_DIR / "corpus_original.jsonl"
EMB_PATH = DATA_DIR / "embeddings_G_pro_large.npy"
FAISS_INDEX_PATH = DATA_DIR / "faiss_index_G_pro_large.index"

# ------------------------
# Logging
# ------------------------
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# ------------------------
# Carregar embeddings i index
# ------------------------
if not EMB_PATH.exists() or not FAISS_INDEX_PATH.exists():
    raise RuntimeError("No hi ha embeddings ni index; genera'ls localment primer.")

# Carregar embeddings amb allow_pickle per evitar errors
embeddings = np.load(EMB_PATH, allow_pickle=True)

# Carregar index FAISS
import faiss
index = faiss.read_index(str(FAISS_INDEX_PATH))

# Carregar metadades
with open(METADATA_PATH, "r", encoding="utf-8") as f:
    metadata = [json.loads(line) for line in f]

# ------------------------
# Funcions helpers
# ------------------------
def search(query: str, top_k: int = 3):
    vec = np.array([query_to_embedding(query)])  # define query_to_embedding segons el teu model
    D, I = index.search(vec, top_k)
    results = [metadata[i] for i in I[0]]
    return results

def query_to_embedding(query: str):
    # Aquí hi posaràs la crida a OpenAI o al model que generi embedding
    # Exemple dummy:
    return np.random.rand(embeddings.shape[1]).astype("float32")

# ------------------------
# Handlers de Telegram
# ------------------------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Hola! Soc el teu bot IA. Escriu-me qualsevol cosa.")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.message.text
    results = search(query)
    reply = "\n\n".join([r.get("text", "") for r in results])
    if not reply:
        reply = "No he trobat res."
    await update.message.reply_text(reply[:4000])  # tallar si és massa llarg

# ------------------------
# Inicialitzar bot
# ------------------------
def main():
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    logger.info("Bot engegat...")
    app.run_polling()

if __name__ == "__main__":
    main()
