import os
import numpy as np
import faiss
from flask import Flask, request, jsonify
from openai import OpenAI

# Rutes dels fitxers
EMBEDDINGS_FILE = "data/embeddings_G_pro_large.npy"
INDEX_FILE = "data/faiss_index_g_pro_large.bin"
CORPUS_FILE = "data/corpus_original.jsonl"

# Inicialitzar client OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Comprovació de fitxers
if not os.path.exists(EMBEDDINGS_FILE) or not os.path.exists(INDEX_FILE):
    raise RuntimeError("No hi ha embeddings ni index; genera'ls localment primer.")

# Carregar embeddings i index
embeddings = np.load(EMBEDDINGS_FILE, allow_pickle=True)

index = faiss.read_index(INDEX_FILE)

# Inicialitzar Flask
app = Flask(__name__)

def get_top_docs(query, k=5):
    """Retorna els k documents més similars a la query."""
    q_emb = client.embeddings.create(
        model="text-embedding-3-large",
        input=query
    ).data[0].embedding
    q_emb = np.array(q_emb, dtype=np.float32).reshape(1, -1)
    distances, indices = index.search(q_emb, k)
    # Aquí pots carregar els documents del corpus segons els indices
    top_docs = []
    with open(CORPUS_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for idx in indices[0]:
            top_docs.append(lines[idx].strip())
    return top_docs

@app.route("/webhook", methods=["POST"])
def webhook():
    """Endpoint per Telegram."""
    data = request.get_json()
    message = data.get("message", {}).get("text")
    if not message:
        return jsonify({"status": "no message"}), 400

    # Obtenir els documents rellevants
    try:
        top_docs = get_top_docs(message)
        answer = " ".join(top_docs[:3])  # per exemple, resumim els 3 primers
    except Exception as e:
        answer = f"Error: {e}"

    # Aquí pots afegir la crida a Telegram per enviar la resposta
    return jsonify({"answer": answer})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
