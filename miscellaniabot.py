import os
import time
import json
import logging
import faiss
import numpy as np
from pathlib import Path
from telegram import Update
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext
from dotenv import load_dotenv
from typing import List, Tuple

# ---- Config ----
load_dotenv()
logging.basicConfig(level=logging.INFO)

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
METADATA_PATH = Path(os.getenv("METADATA_PATH", DATA_DIR / "corpus_original.jsonl"))
EMB_PATH = Path(os.getenv("EMB_PATH", DATA_DIR / "embeddings_G_pro_large.npy"))
FAISS_INDEX_PATH = Path(os.getenv("FAISS_INDEX_PATH", DATA_DIR / "faiss_index_G_pro_large.index"))

TOP_K = 3  # Nombre de resultats més rellevants a retornar
TELEGRAM_CHUNK = 3800

# ---- Carregar embeddings i index ----
if not EMB_PATH.exists() or not FAISS_INDEX_PATH.exists():
    raise RuntimeError("No hi ha embeddings ni index; genera'ls localment primer.")

logging.info("Carregant embeddings...")
embeddings = np.load(EMB_PATH, allow_pickle=True)

logging.info("Carregant index FAISS...")
index = faiss.read_index(str(FAISS_INDEX_PATH))

# Carregar metadata (corpus original)
with open(METADATA_PATH, "r", encoding="utf-8") as f:
    metadata = [json.loads(line) for line in f]

# ---- Funcions helpers ----
def search(query: str, top_k: int = TOP_K) -> List[Tuple[str, float]]:
    """Fes una cerca a FAISS i retorna (text, score)"""
    query_vec = np.random.rand(embeddings.shape[1]).astype(np.float32)  # placeholder, substituir per embedding real
    D, I = index.search(np.array([query_vec]), top_k)
    results = [(metadata[i]["text"], float(D[0][idx])) for idx, i in enumerate(I[0])]
    return results

def chunk_text(text: str, chunk_size: int = TELEGRAM_CHUNK) -> List[str]:
    """Divideix text llarg en chunks per Telegram"""
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

# ---- Handlers de Telegram ----
def start(update: Update, context: CallbackContext):
    update.message.reply_text("Hola! Sóc el bot. Escriu qualsevol cosa i et respondre amb informació rellevant.")

def handle_message(update: Update, context: CallbackContext):
    user_text = update.message.text
    logging.info(f"Missatge rebut: {user_text}")

    # Cerca amb FAISS
    try:
        results = search(user_text)
        response_text = "\n\n".join([f"{r[0]} (score: {r[1]:.2f})" for r in results])
    except Exception as e:
        logging.error(f"Error cercant embeddings: {e}")
        response_text = "No s'ha pogut cercar informació en aquest moment."

    # Enviar resposta en chunks si és molt llarg
    for chunk in chunk_text(response_text):
        update.message.reply_text(chunk)

# ---- Inicialitzar bot ----
def main():
    if not TELEGRAM_TOKEN:
        raise RuntimeError("No s'ha trobat TELEGRAM_TOKEN a les variables d'entorn.")

    updater = Updater(token=TELEGRAM_TOKEN, use_context=True)
    dispatcher = updater.dispatcher

    dispatcher.add_handler(CommandHandler("start", start))
    dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, handle_message))

    logging.info("Bot iniciat. Esperant missatges...")
    updater.start_polling()
    updater.idle()

if __name__ == "__main__":
    main()
