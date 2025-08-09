import os
import numpy as np
import pandas as pd
import faiss
import gradio as gr
from sklearn.preprocessing import normalize
from sentence_transformers import SentenceTransformer

# Load exported artifacts
BOOKS = pd.read_csv("../data/books_meta.csv")
D_MAT  = np.load("../data/desc_emb.npy").astype("float32")
R_MAT  = np.load("../data/rev_emb.npy").astype("float32")

# Normalize (inner product == cosine)
D_MAT = normalize(D_MAT).astype("float32")
R_MAT = normalize(R_MAT).astype("float32")

# Build FAISS
dim = D_MAT.shape[1]
IDX_D = faiss.IndexFlatIP(dim)
IDX_D.add(D_MAT)
IDX_R = faiss.IndexFlatIP(dim)
IDX_R.add(R_MAT)

# Query encoder (same as used)
EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")

def _encode_query(q: str):
    v = EMBEDDER.encode([q], convert_to_numpy=True, show_progress_bar=False)
    return normalize(v).astype("float32")

def fused_search(query, top_k=10, alpha=0.75, overfetch=50):
    q = _encode_query(query)
    K = min(int(overfetch), D_MAT.shape[0])
    Sd, Id = IDX_D.search(q, K)
    Sr, Ir = IDX_R.search(q, K)
    scores = {}
    for s,i in zip(Sd[0], Id[0]): scores[i] = scores.get(i, 0) + alpha*s
    for s,i in zip(Sr[0], Ir[0]): scores[i] = scores.get(i, 0) + (1-alpha)*s
    ranked = sorted(scores.items(), key=lambda x: -x[1])[:int(top_k)]
    rows, scs = [i for i,_ in ranked], [s for _,s in ranked]
    return rows, scs

def format_card(i):
    row = BOOKS.iloc[i]
    title = f"**{row['title']}**"
    author = f"{row['author_name']}"
    meta  = f"**Genre:** {row.get('genre','?')} | **Tone:** {row.get('tone','?')}"
    desc  = (str(row['best_desc'])[:400] + '...') if isinstance(row['best_desc'], str) else ''
    body  = f"{title}\n\n{author}\n\n{meta}\n\n{desc}"
    return body, row.get('image_url', None)

def recommend(query, alpha, top_k, genre_filter, tone_filter):
    if not query or not str(query).strip():
        return "_Enter a query_", []
    rows, _ = fused_search(query, top_k=int(top_k), alpha=float(alpha), overfetch=50)

    # Filters
    filtered = []
    for r in rows:
        g = BOOKS.iloc[r].get('genre', None)
        t = BOOKS.iloc[r].get('tone', None)
        if (genre_filter == "Any" or g == genre_filter) and (tone_filter == "Any" or t == tone_filter):
            filtered.append(r)

    cards, imgs = [], []
    for r in filtered[:int(top_k)]:
        c, img = format_card(r)
        cards.append(c); imgs.append((img, ""))
    return ("\n\n---\n\n".join(cards) if cards else "_No results_"), imgs

with gr.Blocks() as demo:
    gr.Markdown("# Book Recommender — Team 11")
    q = gr.Textbox(label="Describe a book, vibe, themes, or paste a title",
                   placeholder="e.g., tender coming-of-age memoir with humor")
    with gr.Row():
        a = gr.Slider(0.0, 1.0, value=0.75, step=0.05, label="Emphasis (Topic↔Vibe)")
        k = gr.Slider(5, 30, value=10, step=1, label="Results")
    with gr.Row():
        gsel = gr.Dropdown(["Any","Fiction","Nonfiction"], value="Any", label="Genre")
        tsel = gr.Dropdown(["Any","uplifting","dark","mixed"], value="Any", label="Tone")
    res = gr.Markdown()
    gal = gr.Gallery(label="Covers", columns=5, height=220)
    btn = gr.Button("Recommend")
    btn.click(recommend, inputs=[q, a, k, gsel, tsel], outputs=[res, gal])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", 7860)))
