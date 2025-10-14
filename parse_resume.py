import re
import os
import tempfile
from pdfminer.high_level import extract_text
from docx import Document
import pytesseract
from PIL import Image
import fitz  # PyMuPDF
import threading
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import fuzz

# -------------------- Lazy-loaded models --------------------
nlp = None
sbert = None

def get_nlp():
    global nlp
    if nlp is None:
        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            os.system("python -m spacy download en_core_web_sm")
            nlp = spacy.load("en_core_web_sm")
    return nlp

def get_sbert():
    global sbert
    if sbert is None:
        sbert = SentenceTransformer("all-MiniLM-L6-v2")
    return sbert

# -------------------- Regex --------------------
EMAIL_RE = re.compile(r"[\w\.-]+@[\w\.-]+\.\w+")
PHONE_RE = re.compile(r"(\+?\d[\d\-\s\(\)]{7,}\d)")

# -------------------- Utilities --------------------
def clean_text(text: str) -> str:
    text = re.sub(r"\r\n|\r", "\n", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^A-Za-z0-9@+.,:;!?()\-\n\s#\/]", " ", text)
    return text.strip()

def read_file_text(path: str) -> str:
    text = ""
    if path.lower().endswith(".pdf"):
        try:
            text = extract_text(path).strip()
        except:
            text = ""
        try:
            doc = fitz.open(path)
            for page in doc:
                page_text = page.get_text("text")
                if len(page_text.strip()) < 20:
                    pix = page.get_pixmap(dpi=300)
                    img_path = tempfile.mktemp(suffix=".png")
                    pix.save(img_path)
                    img = Image.open(img_path).convert("L")
                    page_text = pytesseract.image_to_string(img, config=r'--oem 3 --psm 6')
                    os.remove(img_path)
                text += "\n" + page_text
        except:
            pass
    elif path.lower().endswith(".docx"):
        try:
            doc = Document(path)
            text = "\n".join([p.text for p in doc.paragraphs])
        except:
            pass
    else:
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        except:
            pass
    return clean_text(text)

def split_into_sections(text: str) -> dict:
    headers = {
        "summary": ["summary", "profile", "objective"],
        "skills": ["skills", "technical skills", "skillset", "technologies"],
        "experience": ["work experience", "experience", "employment", "work history"],
        "projects": ["projects", "personal projects"],
        "education": ["education", "academic", "qualifications"],
    }
    lines = text.split('\n')
    positions = []
    for idx, line in enumerate(lines):
        l = line.strip().lower()
        for key, variants in headers.items():
            if any(v in l for v in variants):
                positions.append((idx, key))
                break
    if not positions:
        return {"other": text}
    positions.sort()
    sections = {}
    for i, (pos, key) in enumerate(positions):
        start = pos
        end = positions[i+1][0] if i+1 < len(positions) else len(lines)
        sec_text = "\n".join(lines[start:end]).strip()
        sections[key] = sections.get(key, "") + "\n" + sec_text
    return sections

def extract_skills_from_text(text: str, job_keywords: list) -> list:
    found = set()
    text_lower = text.lower()
    for kw in job_keywords:
        kw = kw.strip().lower()
        if kw and kw in text_lower:
            found.add(kw)
    return list(found)

# -------------------- Scoring --------------------
def fuzzy_keyword_score(resume_text: str, job_keywords: list, threshold=70) -> float:
    if not job_keywords:
        return 0.0
    matches = 0
    resume_words = set(resume_text.split())
    for kw in job_keywords:
        kw = kw.lower()
        max_score = 0
        for word in resume_words:
            s = fuzz.partial_ratio(kw, word)
            if s > max_score:
                max_score = s
        if max_score >= threshold:
            matches += 1
    return matches / len(job_keywords)

def semantic_score(resume_text: str, job_keywords: list) -> float:
    if not job_keywords:
        return 0.0
    nlp_model = get_nlp()
    sbert_model = get_sbert()
    sentences = [sent.text for sent in nlp_model(resume_text).sents if len(sent.text.strip()) > 5]
    emb_sentences = sbert_model.encode(sentences, convert_to_tensor=True)
    emb_keywords = sbert_model.encode(job_keywords, convert_to_tensor=True)
    sim_matrix = util.cos_sim(emb_sentences, emb_keywords)
    max_per_kw = sim_matrix.max(dim=0)[0]
    return float(max_per_kw.mean())

def tfidf_similarity(resume_text: str, job_text: str) -> float:
    try:
        vect = TfidfVectorizer(stop_words="english", ngram_range=(1,2))
        X = vect.fit_transform([resume_text, job_text])
        return cosine_similarity(X[0:1], X[1:2])[0][0]
    except:
        return 0.0

# -------------------- Resume Summary --------------------
def summarize_resume(text: str, job_keywords: list) -> str:
    bullets = []
    text_lower = text.lower()

    # Education
    edu_lines = [line.strip() for line in text.split("\n") if any(word in line.lower() for word in
                  ["college", "university", "school", "degree", "undergraduate", "postgraduate"])]
    if edu_lines:
        bullets.append(f"- Education: {edu_lines[0][:100]}{'...' if len(edu_lines[0])>100 else ''}")

    # Technical Skills
    skills_found = extract_skills_from_text(text, job_keywords)
    if skills_found:
        bullets.append(f"- Technical Skills: {', '.join(skills_found[:10])}")

    # Projects
    project_keywords = ["tensorflow","pytorch","resnet","ocr","yolo","opencv","ml","ai","cnn","rnn","neural network"]
    project_lines = [line.strip() for line in text.split("\n") if any(tool in line.lower() for tool in project_keywords)]
    for pl in project_lines[:3]:
        bullets.append(f"- Project: {pl[:120]}{'...' if len(pl)>120 else ''}")

    # Achievements
    achievement_keywords = ["award","winner","hackathon","competition","challenge","semi-finalist","finalist"]
    achievement_lines = [line.strip() for line in text.split("\n") if any(word in line.lower() for word in achievement_keywords)]
    for al in achievement_lines[:2]:
        bullets.append(f"- Achievement: {al[:120]}{'...' if len(al)>120 else ''}")

    # Profiles
    profile_links = []
    if "github.com" in text_lower: profile_links.append("GitHub")
    if "linkedin.com" in text_lower: profile_links.append("LinkedIn")
    if "portfolio" in text_lower: profile_links.append("Portfolio")
    if profile_links:
        bullets.append("- Profiles: " + ", ".join(profile_links))

    return "\n".join(bullets)

# -------------------- Main --------------------
def parse_resume_and_score(path: str, job_description: str = "", job_keywords: list = None, weights: dict = None) -> dict:
    if weights is None:
        weights = {"keyword": 0.7, "semantic": 0.2, "tfidf": 0.1}

    text = read_file_text(path)
    if not text or len(text) < 20:
        return {"error": "Failed to extract text from resume", "score": 0.0}

    sections = split_into_sections(text)
    full_text = " ".join(sections.values())

    email_match = EMAIL_RE.search(text)
    phone_match = PHONE_RE.search(text)
    contacts = {
        "email": email_match.group(0) if email_match else None,
        "phone": phone_match.group(0) if phone_match else None
    }

    job_keywords = job_keywords or []
    job_description = job_description or ""

    skills_list = extract_skills_from_text(full_text, job_keywords)

    kscore = fuzzy_keyword_score(full_text, job_keywords)
    sscore = semantic_score(full_text, job_keywords)
    tfidf_score = tfidf_similarity(full_text, job_description)

    final_score = (weights["keyword"] * kscore +
                   weights["semantic"] * sscore +
                   weights["tfidf"] * tfidf_score)
    if kscore > 0.8:
        final_score = min(final_score * 1.15, 1.0)

    summary_text = summarize_resume(full_text, job_keywords)

    return {
        "email": contacts["email"],
        "phone": contacts["phone"],
        "skills": skills_list,
        "score": round(final_score * 100, 2),
        "breakdown": {
            "keyword": round(kscore*100, 2),
            "semantic": round(sscore*100, 2),
            "tfidf": round(tfidf_score*100, 2)
        },
        "sections": list(sections.keys()),
        "full_text": full_text,
        "resume_summary": summary_text
    }

# Optional: preload models in background (non-blocking)
def preload_models():
    threading.Thread(target=get_nlp).start()
    threading.Thread(target=get_sbert).start()
