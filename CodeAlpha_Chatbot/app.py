"""
ShopBot Flask Backend — Bilingual FAQ Chatbot
Dark AI Theme | Cosine Similarity NLP | English + Bangla
"""

from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import time
from flask_cors import CORS
import json, os, re, math

app = Flask(__name__)
CORS(app)

FAQ_FILE = os.path.join(os.path.dirname(__file__), "data", "faqs.json")
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ── NLP Utilities ──────────────────────────────────────────────────────────────

def load_faqs():
    with open(FAQ_FILE, "r", encoding="utf-8") as f:
        return json.load(f)["faqs"]

def save_faqs(faqs):
    with open(FAQ_FILE, "w", encoding="utf-8") as f:
        json.dump({"faqs": faqs}, f, ensure_ascii=False, indent=2)

def tokenize(text):
    text = text.lower()
    text = re.sub(r'[।,.!?;:\'"()\[\]{}]', ' ', text)
    return [t for t in text.split() if t]

def term_frequency(tokens):
    tf = {}
    for t in tokens:
        tf[t] = tf.get(t, 0) + 1
    return tf

def cosine_similarity(vec1, vec2):
    all_terms = set(vec1) | set(vec2)
    dot = sum(vec1.get(t, 0) * vec2.get(t, 0) for t in all_terms)
    mag1 = math.sqrt(sum(v**2 for v in vec1.values()))
    mag2 = math.sqrt(sum(v**2 for v in vec2.values()))
    if mag1 == 0 or mag2 == 0:
        return 0
    return dot / (mag1 * mag2)

def detect_language(text):
    # Bangla Unicode range \u0980-\u09FF
    return "bn" if re.search(r'[\u0980-\u09FF]', text) else "en"

def build_doc_string(faq, lang):
    if lang == "bn":
        return f"{faq.get('question_bn','')} {faq.get('answer_bn','')} {' '.join(faq.get('keywords',[]))}"
    return f"{faq.get('question_en','')} {faq.get('answer_en','')} {faq.get('category','')} {' '.join(faq.get('keywords',[]))}"

def find_best_match(user_query, faqs):
    lang = detect_language(user_query)
    q_tokens = tokenize(user_query)
    q_tf = term_frequency(q_tokens)
    best_score, best_faq = 0, None

    for faq in faqs:
        doc_str = build_doc_string(faq, lang)
        doc_tf = term_frequency(tokenize(doc_str))
        score = cosine_similarity(q_tf, doc_tf)

        # Keyword boost
        kw_boost = 0.3 if any(
            kw.lower() in user_query.lower()
            for kw in faq.get("keywords", [])
        ) else 0

        final_score = score + kw_boost
        if final_score > best_score:
            best_score, best_faq = final_score, faq

    if best_score < 0.05:
        return None, 0, lang
    return best_faq, best_score, lang

def get_next_id(faqs):
    return max((f["id"] for f in faqs), default=0) + 1


# Image upload endpoint
@app.route('/api/upload_image', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    f = request.files['image']
    if f.filename == '':
        return jsonify({'error': 'Empty filename'}), 400
    filename = secure_filename(f.filename)
    # prefix with timestamp to avoid collisions
    filename = f"{int(time.time())}_{filename}"
    dest = os.path.join(UPLOAD_FOLDER, filename)
    f.save(dest)
    url = f"/static/uploads/{filename}"

    # Add as a new FAQ with a fixed price answer (10 $) unless it already exists
    try:
        faqs = load_faqs()
        exists = any(f.get('image') == url for f in faqs)
        if not exists:
            new_faq = {
                "id": get_next_id(faqs),
                "category": "Products",
                "question_en": f"Image: {filename}",
                "answer_en": "700 taka",
                "question_bn": "",
                "answer_bn": "৭০০ টাকা",
                "keywords": ["price", "cost", "how much", "price?"],
                "image": url
            }
            faqs.append(new_faq)
            save_faqs(faqs)
    except Exception:
        # don't fail the upload if saving FAQ fails
        pass

    return jsonify({'url': url}), 201

# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/admin")
def admin():
    return render_template("admin.html")

# Chat
@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json()
    message = (data.get("message") or "").strip()
    image_url = data.get("image_url") if data else None
    if not message:
        # If an image is provided but no textual message, acknowledge image
        if image_url:
            lang = detect_language(image_url)
            fallback = (
                "আমি আপনার চিত্রটি পেয়েছি। দয়া করে বলুন এই ছবির বিষয়ে আপনি কী জানতে চান।"
                if lang == "bn" else
                "I received your image. Please tell me what you'd like me to check in the image."
            )
            return jsonify({"answer": fallback, "matched": False, "lang": lang, "confidence": 0})
        return jsonify({"error": "Message required"}), 400

    faqs = load_faqs()
    lang = detect_language(message)
    # If there's an image, give a simple acknowledgement (image analysis not implemented)
    if image_url:
        answer = (
            "আমি আপনার চিত্রটি পেয়েছি। দয়া করে বলুন এই ছবির বিষয়ে আপনি কী জানতে চান।"
            if lang == "bn" else
            f"I received your image at {image_url}. Please tell me what you'd like me to check."
        )
        return jsonify({"answer": answer, "matched": False, "lang": lang, "confidence": 0})
    faq, score, lang = find_best_match(message, faqs)

    if not faq:
        fallback = (
            "দুঃখিত, আমি আপনার প্রশ্নের উত্তর খুঁজে পাইনি। support@ourstore.com এ যোগাযোগ করুন।"
            if lang == "bn" else
            "Sorry, I couldn't find a matching answer. Please contact support@ourstore.com or WhatsApp: +880 1700-000000"
        )
        return jsonify({"answer": fallback, "matched": False, "lang": lang, "confidence": 0})

    answer = faq["answer_bn"] if lang == "bn" else faq["answer_en"]
    question = faq["question_bn"] if lang == "bn" else faq["question_en"]
    return jsonify({
        "answer": answer,
        "matched": True,
        "matchedQuestion": question,
        "category": faq["category"],
        "lang": lang,
        "confidence": min(int(score * 100), 99),
        "faqId": faq["id"]
    })

# Suggestions
@app.route("/api/suggestions")
def suggestions():
    q = (request.args.get("q") or "").strip()
    if len(q) < 2:
        return jsonify({"suggestions": []})
    faqs = load_faqs()
    lang = detect_language(q)
    ql = q.lower()
    results = []
    for faq in faqs:
        question = faq["question_bn"] if lang == "bn" else faq["question_en"]
        kw_match = any(k.lower() in ql for k in faq.get("keywords", []))
        if ql in question.lower() or kw_match:
            results.append({"id": faq["id"], "question": question, "category": faq["category"]})
        if len(results) >= 5:
            break
    return jsonify({"suggestions": results})

# Public FAQ list
@app.route("/api/faqs")
def faqs_list():
    faqs = load_faqs()
    category = request.args.get("category")
    if category:
        faqs = [f for f in faqs if f["category"].lower() == category.lower()]
    return jsonify({"faqs": faqs, "total": len(faqs)})

@app.route("/api/faqs/categories")
def categories():
    faqs = load_faqs()
    cats = list(dict.fromkeys(f["category"] for f in faqs))
    return jsonify({"categories": cats})

# Admin CRUD
@app.route("/api/admin/faqs")
def admin_faqs():
    return jsonify({"faqs": load_faqs()})

@app.route("/api/admin/faqs", methods=["POST"])
def admin_add_faq():
    d = request.get_json()
    if not d.get("category") or not d.get("question_en") or not d.get("answer_en"):
        return jsonify({"error": "category, question_en, answer_en required"}), 400
    faqs = load_faqs()
    keywords = d.get("keywords", [])
    if isinstance(keywords, str):
        keywords = [k.strip() for k in keywords.split(",") if k.strip()]
    new_faq = {
        "id": get_next_id(faqs),
        "category": d["category"],
        "question_en": d["question_en"],
        "answer_en": d["answer_en"],
        "question_bn": d.get("question_bn", ""),
        "answer_bn": d.get("answer_bn", ""),
        "keywords": keywords
    }
    faqs.append(new_faq)
    save_faqs(faqs)
    return jsonify({"success": True, "faq": new_faq}), 201

@app.route("/api/admin/faqs/<int:faq_id>", methods=["PUT"])
def admin_update_faq(faq_id):
    faqs = load_faqs()
    idx = next((i for i, f in enumerate(faqs) if f["id"] == faq_id), None)
    if idx is None:
        return jsonify({"error": "Not found"}), 404
    d = request.get_json()
    keywords = d.get("keywords", faqs[idx].get("keywords", []))
    if isinstance(keywords, str):
        keywords = [k.strip() for k in keywords.split(",") if k.strip()]
    faqs[idx] = {**faqs[idx], **d, "id": faq_id, "keywords": keywords}
    save_faqs(faqs)
    return jsonify({"success": True, "faq": faqs[idx]})

@app.route("/api/admin/faqs/<int:faq_id>", methods=["DELETE"])
def admin_delete_faq(faq_id):
    faqs = load_faqs()
    idx = next((i for i, f in enumerate(faqs) if f["id"] == faq_id), None)
    if idx is None:
        return jsonify({"error": "Not found"}), 404
    deleted = faqs.pop(idx)
    save_faqs(faqs)
    return jsonify({"success": True, "deleted": deleted})

@app.route("/api/admin/stats")
def admin_stats():
    faqs = load_faqs()
    cats = {}
    for f in faqs:
        cats[f["category"]] = cats.get(f["category"], 0) + 1
    bilingual = sum(1 for f in faqs if f.get("question_bn") and f.get("answer_bn"))
    return jsonify({"totalFAQs": len(faqs), "categories": cats, "bilingualCount": bilingual})

if __name__ == "__main__":
    app.run(debug=True, port=5000)
