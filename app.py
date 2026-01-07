# ======================================
# app.py — Robust SpaCy + TextBlob Hybrid System
# ======================================

import streamlit as st
import pickle
from nltk.metrics import edit_distance
from textblob import Word
import spacy
import lemminflect
from lemminflect import Lemmatizer
from spacy.tokens import Token

# -----------------------------
# Fix spaCy / lemminflect conflicts
# -----------------------------
if not Token.has_extension("inflect"):
    Token.set_extension("inflect", method=lemminflect.getInflection)

Token.set_extension("lemma", method=Lemmatizer().spacyGetLemma, force=True)

# -----------------------------
# Streamlit config
# -----------------------------
st.set_page_config(
    page_title="Spelling & Grammar Correction System",
    page_icon="✍️",
    layout="centered"
)

# -----------------------------
# Load spaCy
# -----------------------------
nlp = spacy.load("en_core_web_sm")

# -----------------------------
# Constants
# -----------------------------
BE_VERBS = {"am", "is", "are", "was", "were"}
HAS_VERBS = {"has", "have", "had"}
DO_VERBS = {"do", "does", "did"}
AUX_VERBS = BE_VERBS | HAS_VERBS | DO_VERBS

FUNCTION_WORDS = {
    "no","this","that","which","who","whom","whose",
    "it","they","we","he","she","a","an","the",
    "and","or","but","of","in","on","for","to",
    "with","by","at","not","any","already","very","i","me","my","mine","you","yours","your","he"
    ,"him","his","she","her","hers","we","us","ours","they","their","theirs","more","once","these","those","there"
}

IRREGULAR_PAST = {
    "be": "been", "have": "had", "do": "done", "go": "gone", 
    "come": "come", "rise": "risen", "see": "seen", "use": "used", "start": "started"
}
PAST_PARTICIPLES = set(IRREGULAR_PAST.values())
QUANTIFIERS = {"many", "few", "several", "all", "some", "most", "each"}

# -----------------------------
# Load corpus (Updated for Everygram/Trigram)
# -----------------------------
@st.cache_resource
def load_models():
    with open("vocabulary.txt", "r", encoding="utf-8") as f:
        vocab = set(w.lower() for w in f.read().splitlines())
    with open("word_freq.pkl", "rb") as f:
        word_freq = pickle.load(f)
    with open("bigram_counts.pkl", "rb") as f:
        bigram_counts = pickle.load(f)
    with open("trigram_counts.pkl", "rb") as f:  # Added Trigram file
        trigram_counts = pickle.load(f)
    
    # Using word_freq for unigram counts
    unigram_counts = word_freq 

    vocab |= FUNCTION_WORDS | AUX_VERBS | PAST_PARTICIPLES
    return vocab, word_freq, bigram_counts, trigram_counts, unigram_counts, sum(unigram_counts.values())

VOCAB, WORD_FREQ, BIGRAM_COUNTS, TRIGRAM_COUNTS, UNIGRAM_COUNTS, TOTAL_UNIGRAMS = load_models()
VOCAB_SIZE = len(VOCAB)

# -----------------------------
# Helper functions (Updated for Everygram/Trigram)
# -----------------------------
def get_contextual_prob(word, prev_word, prev_2_word):
    # 1. Try Trigram Score
    if prev_2_word and prev_word:
        tri_count = TRIGRAM_COUNTS.get((prev_2_word, prev_word, word), 0)
        if tri_count > 0:
            bi_context = BIGRAM_COUNTS.get((prev_2_word, prev_word), 0)
            return (tri_count / (bi_context + VOCAB_SIZE)) * 10.0
    
    # 2. Try Bigram Score
    if prev_word:
        bi_count = BIGRAM_COUNTS.get((prev_word, word), 0)
        if bi_count > 0:
            uni_context = UNIGRAM_COUNTS.get(prev_word, 0)
            return (bi_count / (uni_context + VOCAB_SIZE)) * 5.0
            
    # 3. Fallback to Unigram Probability
    return UNIGRAM_COUNTS.get(word, 0) / TOTAL_UNIGRAMS

def generate_candidates(word):
    return [v for v in VOCAB if edit_distance(word.lower(), v) <= 2]

def rank_candidates(candidates, prev_word, prev_2_word, original):
    ranked = []
    for c in candidates:
        # Everygram score (Trigram -> Bigram -> Unigram)
        score = get_contextual_prob(c, prev_word, prev_2_word)
        
        # Add edit distance weighting
        dist = edit_distance(original.lower(), c)
        score += max(0, 1 - dist / 5)
        
        ranked.append({"word": c, "edit_distance": dist, "score": score})
    return sorted(ranked, key=lambda x: x["score"], reverse=True)[:5]

# -----------------------------
# Grammar correction (Original Logic Intact)
# -----------------------------
def apply_grammar(doc):
    corrected_tokens = [t.text for t in doc]
    grammar_indices = []
    
    # --- STEP 1: QUANTIFIER LOGIC ---
    plural_fix_made = False
    for i, token in enumerate(doc):
        if token.text.lower() in {"many", "several", "few", "all", "some", "most", "various"}:
            for j in range(i + 1, min(len(doc), i + 4)):
                if doc[j].is_punct or doc[j].pos_ == "ADP": break
                if doc[j].pos_ == "NOUN" or doc[j].tag_ == "NN":
                    if doc[j].tag_ != "NNS":
                        plural_word = doc[j]._.inflect("NNS")
                        if plural_word:
                            corrected_tokens[j] = plural_word
                            grammar_indices.append(j)
                            plural_fix_made = True
                            break

    # --- FIX: RE-PROCESS AFTER STEP 1 ---
    # We move this OUTSIDE the loop so it only runs once if any change was made
    if plural_fix_made:
        new_text = " ".join(corrected_tokens)
        doc = nlp(new_text)

    # --- STEP 2: HYBRID SUBJECT-VERB AGREEMENT ---
    for token in doc:
        # A: Auxiliary Guard
        if token.pos_ == "VERB" and any(w.lemma_ == "do" for w in token.ancestors):
            if token.tag_ == "VBZ": 
                base = token._.inflect("VB")
                if base:
                    corrected_tokens[token.i] = base
                    grammar_indices.append(token.i)
            continue 

        # B: Find the Subject
        subj = None
        nsubj = [w for w in token.lefts if w.dep_ == "nsubj"]
        if nsubj:
            subj = nsubj[0]
        elif token.i > 0:
            prev_t = doc[token.i - 1]
            if prev_t.pos_ in {"PRON", "NOUN", "PROPN"}:
                subj = prev_t

        if subj and (token.pos_ in {"VERB", "AUX"}):
            subj_word = corrected_tokens[subj.i].lower()
            token_text_low = token.text.lower()
            
            # AI Check: Ensure AI is treated as 3rd Person Singular
            is_sing_3rd = (subj_word in {"he", "she", "it", "ai"} or subj.tag_ in {"NN", "NNP"})
            if subj.text.isupper() and 2 <= len(subj.text) <= 7:
                is_sing_3rd = True

            is_plural_type = (subj_word in {"i", "you", "we", "they"} or 
                              subj.tag_ in {"NNS", "NNPS"} or 
                              subj_word in {"children", "people", "men", "women"})

            if is_sing_3rd:
                # Fix "AI have" -> "AI has" / "AI do" -> "AI does"
                if token.lemma_ == "do" and token_text_low == "do":
                    corrected_tokens[token.i] = "does"
                    grammar_indices.append(token.i)
                elif token.lemma_ == "be" and token_text_low == "were":
                    corrected_tokens[token.i] = "was"
                    grammar_indices.append(token.i)
                elif token.lemma_ == "have" and token_text_low == "have": # Explicit Have fix
                    corrected_tokens[token.i] = "has"
                    grammar_indices.append(token.i)
                elif token.tag_ == "VBP":
                    sing_v = token._.inflect("VBZ")
                    if sing_v:
                        corrected_tokens[token.i] = sing_v
                        grammar_indices.append(token.i)

            elif is_plural_type:
                if subj_word == "i" and token_text_low == "was":
                    continue
                if token.lemma_ == "do" and token_text_low == "does":
                    corrected_tokens[token.i] = "do"
                    grammar_indices.append(token.i)
                elif token.tag_ == "VBZ":
                    plural_v = token._.inflect("VBP") or token._.inflect("VB")
                    if plural_v:
                        corrected_tokens[token.i] = plural_v
                        grammar_indices.append(token.i)

            # --- STEP 3: FINAL CLEANUP ---
            if corrected_tokens:
                if corrected_tokens[0][0].islower():
                    corrected_tokens[0] = corrected_tokens[0].capitalize()
                    if 0 not in grammar_indices: grammar_indices.append(0)
                for i in range(len(corrected_tokens)):
                    if corrected_tokens[i].lower() == "i" and len(corrected_tokens[i]) == 1:
                        corrected_tokens[i] = "I"
                        if i not in grammar_indices: grammar_indices.append(i)
                    if i > 0 and corrected_tokens[i-1] in {".", "!", "?"}:
                        if corrected_tokens[i][0].islower():
                            corrected_tokens[i] = corrected_tokens[i].capitalize()
                            if i not in grammar_indices: grammar_indices.append(i)

    return corrected_tokens, list(set(grammar_indices))

def detect_errors(text, nlp):
    doc = nlp(text)
    corrected_tokens, grammar_indices = apply_grammar(doc)
    spelling_errors = []
    
    for i, token in enumerate(doc):
        word_to_check = corrected_tokens[i]
        if not word_to_check.isalpha() or i in grammar_indices: continue
        low = word_to_check.lower()
        if low == "i" or low in FUNCTION_WORDS or low in AUX_VERBS: continue
        lemma = token.lemma_.lower()
        if low not in VOCAB and lemma not in VOCAB:
            try: suggestion = Word(low).spellcheck()[0][0]
            except: suggestion = low
            spelling_errors.append({"word": word_to_check, "index": i, "suggestion": suggestion})

    return corrected_tokens, grammar_indices, {}, spelling_errors

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("✍️ Spelling & Grammar Correction System")
st.markdown("""
**How to use:**  
1. Enter your text  
2. Click **Check Text**  
3. 🔴 Red → spelling error  
4. 🟢 Green → grammar correction  
5. Click words to see suggestions
""")
user_input = st.text_area("Enter text:", height=150)

if st.button("🔍 Check Text", use_container_width=True):
    corrected_tokens, grammar_indices, _, errors = detect_errors(user_input, nlp)
    spelling_idx = {e["index"] for e in errors}

    highlighted = []
    for i, tok in enumerate(corrected_tokens):
        if tok in {".", ",", "!", "?", ";", ":"}: highlighted.append(tok)
        elif i in spelling_idx: highlighted.append(f"<span style='color:red;font-weight:bold'>{tok}</span>")
        elif i in grammar_indices: highlighted.append(f"<span style='color:green;font-weight:bold'>{tok}</span>")
        else: highlighted.append(tok)

    html_text = ""
    for w in highlighted:
        if w in {".", ",", "!", "?", ";", ":"}: html_text = html_text.rstrip() + w + " "
        else: html_text += w + " "
    st.subheader("🖍 Highlighted Text")
    st.markdown(html_text.strip(), unsafe_allow_html=True)

    if errors:
        st.subheader("📌 Suggestions")
        for e in errors:
            with st.expander(f"`{e['word']}`"):
                idx = e["index"]
                prev = corrected_tokens[idx-1].lower() if idx > 0 else None
                prev_2 = corrected_tokens[idx-2].lower() if idx > 1 else None
                for s in rank_candidates(generate_candidates(e["word"].lower()), prev, prev_2, e["word"].lower()):
                    st.markdown(f"- **{s['word']}** | score `{round(s['score'],4)}`")
# -----------------------------
# Compact Dictionary Explorer
# -----------------------------
st.divider()
st.subheader("🔎 Dictionary Explorer")

# 1. Initialize session state for the search input
if "search_key" not in st.session_state:
    st.session_state.search_key = ""

# 2. Define clear logic
def clear_text():
    st.session_state.search_key = ""

# 3. GUI Layout: Search bar and Clear button side-by-side
col1, col2 = st.columns([0.8, 0.2])

with col1:
    # We use 'key' to link this directly to session_state
    search_query = st.text_input(
        "Search corpus:", 
        placeholder="Type a word...", 
        key="search_key",
        label_visibility="collapsed"
    )

with col2:
    st.button("Clear", on_click=clear_text, use_container_width=True)

# 4. Search Result Logic
if search_query:
    low_query = search_query.lower().strip()
    if low_query in VOCAB:
        freq = WORD_FREQ.get(low_query, 0)
        st.success(f"**{low_query}** found! Frequency: `{freq}`")
    else:
        st.error(f"'{low_query}' not in corpus.")

# 5. Scrollable Dictionary List
with st.expander("View Full Dictionary List", expanded=False):
    # Create list of [Word, Frequency] sorted by most frequent
    dict_list = [{"Word": k, "Freq": v} for k, v in sorted(WORD_FREQ.items(), key=lambda x: x[1], reverse=True)]
    
    st.dataframe(
        dict_list,
        use_container_width=True,
        height=250, # Controls scrollable height
        column_config={
            "Word": st.column_config.TextColumn("Word"),
            "Freq": st.column_config.NumberColumn("Frequency", format="%d 🔢")
        },
        hide_index=True
    )

st.caption("📘 MSc Artificial Intelligence | NLP System")