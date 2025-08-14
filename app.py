import os
import csv
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

# Query encoder (same model used for embeddings)
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
    for s,i in zip(Sd[0], Id[0]): 
        scores[i] = scores.get(i, 0) + alpha*s
    for s,i in zip(Sr[0], Ir[0]): 
        scores[i] = scores.get(i, 0) + (1-alpha)*s
    ranked = sorted(scores.items(), key=lambda x: -x[1])[:int(top_k)]
    rows, scs = [i for i,_ in ranked], [s for _,s in ranked]
    return rows, scs

def format_card(i):
    row = BOOKS.iloc[i]
    img_url = row.get('image_url', '')
    title = f"<strong>{row['title']}</strong>"
    author = f"{row['author_name']}"
    meta = f"<strong>Genre:</strong> {row.get('genre','?')} | <strong>Tone:</strong> {row.get('tone','?')}"
    desc = (str(row['description'])[:200] + '...') if isinstance(row['description'], str) else ''

    return f"""
    <div style="display:flex; gap:15px; align-items:flex-start; margin-bottom:20px;">
        <img src="{img_url}" alt="Cover" style="width:120px; height:auto; border-radius:6px; flex-shrink:0;">
        <div>
            {title}<br>
            {author}<br>
            {meta}<br><br>
            {desc}
        </div>
    </div>
    """

# ---------- Feedback logging ----------
FEEDBACK_PATH = "../data/feedback.csv"

def _ensure_feedback_header():
    if not os.path.exists(FEEDBACK_PATH):
        with open(FEEDBACK_PATH, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=[
                "ts_iso","useful","query","book_id","title","alpha","genre_filter","tone_filter","comment"
            ])
            w.writeheader()
_ensure_feedback_header()

def _log_feedback(row_dict):
    with open(FEEDBACK_PATH, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "ts_iso","useful","query","book_id","title","alpha","genre_filter","tone_filter","comment"
        ])
        w.writerow(row_dict)

def _parse_selection(sel_text, rows):
    # "1. Title (book_id)" -> index 0
    if not sel_text or not rows:
        return None
    try:
        idx = int(sel_text.split(".", 1)[0].strip()) - 1
        if 0 <= idx < len(rows):
            return rows[idx]
    except Exception:
        pass
    return None

# ---------- Core handlers ----------
def recommend(query, alpha, top_k, genre_filter, tone_filter):
    if not query or not str(query).strip():
        return "_Enter a query_"
    
    rows, _ = fused_search(query, top_k=int(top_k), alpha=float(alpha), overfetch=50)

    out = []
    for r in rows:
        g = BOOKS.iloc[r].get('genre', None)
        t = BOOKS.iloc[r].get('tone', None)
        if (genre_filter == "Any" or g == genre_filter) and (tone_filter == "Any" or t == tone_filter):
            out.append(r)

    cards_html = [format_card(r) for r in out[:int(top_k)]]

    if not cards_html:
        return "_No results_"

    # Dropdown options like "1. Title (book_id)"
    opts = [f"{i+1}. {BOOKS.iloc[r]['title']} ({BOOKS.iloc[r]['book_id']})" for i, r in enumerate(out[:int(top_k)])]
    dd = gr.update(choices=opts, value=(opts[0] if opts else None))

    # Return results, gallery, dropdown, and states (rows + query)
    return "".join(cards_html), dd, out[:int(top_k)], query

def send_feedback(useful, selection, rows_state, query_state, alpha, genre_filter, tone_filter, comment):
    ridx = _parse_selection(selection, rows_state)
    if ridx is None:
        return "⚠️ Select a result in the dropdown first, then click 👍/👎."
    row = BOOKS.iloc[ridx]
    _log_feedback({
        "ts_iso": datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "useful": "yes" if useful else "no",
        "query": str(query_state),
        "book_id": str(row.get("book_id","")),
        "title": str(row.get("title","")),
        "alpha": float(alpha),
        "genre_filter": str(genre_filter),
        "tone_filter": str(tone_filter),
        "comment": (comment or "").strip(),
    })
    return "✅ Feedback saved. Thanks!"

# --- Admin-gated download (set ADMIN_KEY in Settings > Secrets) ---
ADMIN_KEY = os.getenv("ADMIN_KEY", "")

def gate_download(key: str):
    ok = bool(ADMIN_KEY) and (key == ADMIN_KEY)
    # Only show the download button when the key matches
    return gr.update(visible=ok)

# ---------- UI ----------
custom_css = """
    /* Background Image */
    .gradio-container {
        background-image: url('background.png');  
        background-color: #bedbed;  /* baby blue */
        background-size: 200px 200px;  
        background-position: right bottom;  
        background-repeat: no-repeat;
       /* height: 100vh;   Ensure it covers the entire viewport height */
        color: white;
        padding: 20px;
        margin: 10px;
        border-radius: 10px;  
    }

    /* Typography */
    body {
        font-family: 'Arial', sans-serif;  
    }

    h1, h2, h3, .gradio-title {
        font-family: 'Arial', serif;  
        font-weight: 700;  
        font-size: 32px;  
        color: #333;  
        margin-bottom: 20px;
    }
    .bottom-left-image {
        position: absolute;
        bottom: 20px;  
        left: 20px;    
        width: 150px;  
        height: auto;  
        border-radius: 10px;  
    }

    /* Button Styling */
    .gradio-button {
        background-color: #001F3F;  
        color: white;
        padding: 12px 30px;
        border-radius: 8px;
        font-size: 16px;
        font-weight: 600;
        border: none;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);  
        transition: background-color 0.3s ease, transform 0.2s ease;  
    }

    .gradio-button:hover {
        background-color: #FF6347;  
        transform: scale(1.05);  
    }

    /* Slider Styling */
    .gradio-slider {
        width: 80%;  
        margin: auto;
        border-radius: 8px;
        background-color: #F0F0F0;
    }

    .gradio-slider input {
        border-radius: 8px;  
    }

    /* Input Field Styling */
    .gradio-input {
        border-radius: 8px;
        border: 1px solid #ddd;
        padding: 10px;
        background-color: rgba(255, 255, 255, 0.8);
        font-size: 16px;
    }

    .gradio-input:focus {
        border: 2px solid #FF6347;  
    }

    /* Dropdown Styling */
    .gradio-dropdown {
        border-radius: 8px;
        padding: 10px;
        background-color: rgba(255, 255, 255, 0.8);
        border: 1px solid #ddd;
    }

    /* Markdown Results (Cards) */
    .gradio-markdown {
        padding: 20px;
        background-color: white;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);  
        margin-bottom: 20px;
        font-size: 16px;
    }

    /* Gallery (Book Covers) */
    .gradio-gallery {
        display: grid;
        grid-template-columns: repeat(3, 1fr);  /* 3 columns for book covers */
        gap: 20px;
        padding: 10px;
    }

    .gradio-gallery img {
        border-radius: 10px;
        transition: transform 0.3s ease-in-out;
    }

    .gradio-gallery img:hover {
        transform: scale(1.05);  
    }
"""
with gr.Blocks() as demo:
    gr.HTML(f"<style>{custom_css}</style>")
    with gr.Row():
        gr.Markdown("# 📚 Book Recommender 📚")
    with gr.Row():
        gr.Markdown("### Team 11, SIADS Capstone 2025")
    # Keep state of last results/query for feedback
    rows_state = gr.State([])
    query_state = gr.State("")

    # Inputs
    q = gr.Textbox(label="Describe a book, vibe, themes, or paste a title",
                   placeholder="e.g., tender coming-of-age memoir with humor")
    with gr.Row():
        gsel = gr.Dropdown(["Any","Fiction","Nonfiction"], value="Any", label="Genre")
        tsel = gr.Dropdown(["Any","uplifting","dark","mixed"], value="Any", label="Tone")
    with gr.Row():
        a = gr.Slider(0.0, 1.0, value=0.75, step=0.05, label="Is the book topic or tone more important? (Topic↔Tone)")
        k = gr.Slider(5, 30, value=10, step=1, label="Results")

    # 👉 Recommend button placed BEFORE results/feedback
    btn = gr.Button("Recommend")

    # Results
    res_html = gr.HTML()

    # Feedback area (after results)
    pick = gr.Dropdown(choices=[], label="Pick a result to rate", value=None, interactive=True)
    fb_comment = gr.Textbox(label="Optional comment", placeholder="What did you like / not like?", lines=2)
    with gr.Row():
        btn_up = gr.Button("👍 Useful")
        btn_dn = gr.Button("👎 Not useful")
    fb_msg = gr.Markdown()

    # Admin-only download (hidden unless you set ADMIN_KEY in Secrets)
    admin_pw = gr.Textbox(
        label="Admin key (optional)",
        type="password",
        placeholder="Enter admin key to unlock feedback download",
        visible=bool(ADMIN_KEY)
    )
    dl = gr.DownloadButton("Download feedback.csv", value=FEEDBACK_PATH, visible=False)

    # Wire actions
    btn.click(
        recommend,
        inputs=[q, a, k, gsel, tsel],
        outputs=[res_html, pick, rows_state, query_state]
    )
    btn_up.click(
        lambda sel, rs, qs, alpha, gf, tf, c: send_feedback(True, sel, rs, qs, alpha, gf, tf, c),
        inputs=[pick, rows_state, query_state, a, gsel, tsel, fb_comment],
        outputs=[fb_msg]
    )
    btn_dn.click(
        lambda sel, rs, qs, alpha, gf, tf, c: send_feedback(False, sel, rs, qs, alpha, gf, tf, c),
        inputs=[pick, rows_state, query_state, a, gsel, tsel, fb_comment],
        outputs=[fb_msg]
    )
    admin_pw.change(gate_download, inputs=[admin_pw], outputs=[dl])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", 7860)))
