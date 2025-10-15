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
    await update.message.reply_text("👋 Benvingut a *miscellanIAbot*!\n\n"
        "Pregunta sobre temes, pobles o articles de la Ribera d’Ebre.")

#import tiktoken  # pip install tiktoken

MAX_TELEGRAM_CHARS = 4000
MAX_MODEL_TOKENS = 3000  # màxim tokens per al prompt + resposta
ESTIM_TOKENS_PER_CHAR = 0.25  # aproximació: 1 token ~ 4 caràcters

def truncate_text_for_tokens(text, max_tokens):
    """Trunca text segons el nombre aproximat de tokens."""
    max_chars = int(max_tokens / ESTIM_TOKENS_PER_CHAR)
    if len(text) > max_chars:
        return text[:max_chars] + "..."
    return text

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.message.text.strip()
    if not query:
        await update.message.reply_text("❗ Escriu alguna cosa per cercar.")
        return

    try:
        results = search(query, top_k=5)  # més fragments, però després truncarem

        # Truncar fragments segons el límit total de tokens
        available_tokens_for_context = MAX_MODEL_TOKENS - 600  # reservem 600 per la resposta
        tokens_per_fragment = available_tokens_for_context // len(results) if results else 0

        context_texts = []
        for r in results:
            text = r.get("text", "")
            truncated = truncate_text_for_tokens(text, tokens_per_fragment)
            context_texts.append(truncated)
        context_text = "\n\n".join(context_texts)

        prompt = f"""
Ets un assistent conversacional sobre la historia, cultura, patrimoni de la Ribera d'ebre (Tarragona) molt amable i clar.
La persona et fa aquesta pregunta:
{query}

Centra't en aquests fragments d'informació que he trobat:
{context_text}
Evita contestar amb informacio de fora del corpus que et passo.
No inventis.
Cita fets importants relacionats amb la pregunta.
No donis respostes generalistes.

Respon de manera natural i conversacional, amb les teves paraules.
No afegeixis cap format especial ni markdown.
Si la resposta és llarga, resumeix-la per no passar el límit de Telegram.
"""

        # Crida al model
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=600
        )

        answer = response.choices[0].message.content.strip()
        await update.message.reply_text(answer[:MAX_TELEGRAM_CHARS])

    except Exception as e:
        logger.error(f"Error durant la cerca/conversa: {e}")
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
