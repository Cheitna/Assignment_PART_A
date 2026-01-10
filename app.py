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
    "and","or","but","of","in","on","for",
    "with","by","at","not","any","already","very","i","me","my","mine","you","yours","your","he"
    ,"him","his","she","her","hers","we","us","ours","they","theirs","more","once","these","those"
}

IRREGULAR_PAST = {
    "be": "been", "have": "had", "do": "done", "go": "gone", 
    "come": "come", "rise": "risen", "see": "seen", "use": "used", "start": "started"
}
PAST_PARTICIPLES = set(IRREGULAR_PAST.values())
QUANTIFIERS = {"many", "few", "several", "all", "some", "most", "each"}

# -----------------------------
# Load corpus 
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
def ngram_probability(word, prev_word=None, prev_2_word=None):
    # Trigram
    if prev_2_word and prev_word:
        tri = TRIGRAM_COUNTS.get((prev_2_word, prev_word, word), 0)
        bi = BIGRAM_COUNTS.get((prev_2_word, prev_word), 0)
        if bi > 0:
            return (tri + 1) / (bi + VOCAB_SIZE)

    # Bigram
    if prev_word:
        bi = BIGRAM_COUNTS.get((prev_word, word), 0)
        uni = UNIGRAM_COUNTS.get(prev_word, 0)
        if uni > 0:
            return (bi + 1) / (uni + VOCAB_SIZE)

    # Unigram
    return (UNIGRAM_COUNTS.get(word, 0) + 1) / (TOTAL_UNIGRAMS + VOCAB_SIZE)

# -----------------------------
# Candidate generation
# -----------------------------
MODALS = {
    "will", "would", "can", "could", "may", "might",
    "shall", "should", "must"
}

# -----------------------------
# Confusion sets
# -----------------------------
CONFUSION_SETS = {
    "to": ["to", "too", "two"],
    "too": ["to", "too", "two"],
    "two": ["to", "too", "two"],
    "their": ["their", "there", "they're"],
    "there": ["their", "there", "they're"],
    "they're": ["their", "there", "they're"]
}

# -----------------------------
# Generate candidates (edit-distance + confusion set)
# -----------------------------
def generate_candidates(word, real_word=False):
    low = word.lower()

    # -----------------------------
    # Real-word candidates
    # -----------------------------
    if real_word:
        candidates = []

        if low in CONFUSION_SETS:
            candidates.extend(CONFUSION_SETS[low])

        candidates.extend(
            v for v in VOCAB if edit_distance(low, v) <= 1
        )

        return list(set(c for c in candidates if c != low))

    # -----------------------------
    # Non-word candidates (FIXED)
    # -----------------------------
    else:
        return [
            v for v in VOCAB
            if edit_distance(low, v) <= 2
            and v != low          # ✅ CRITICAL
        ]

####-------------------------
def rank_candidates(candidates, prev_word, prev_2_word, original):
    ranked = []
    for c in candidates:
        # Everygram score (Trigram -> Bigram -> Unigram)
        score = ngram_probability(c, prev_word, prev_2_word)
        
        # Add edit distance weighting
        dist = edit_distance(original.lower(), c)
        score += max(0, 1 - dist / 5)
        
        ranked.append({"word": c, "edit_distance": dist, "score": score})
    return sorted(ranked, key=lambda x: x["score"], reverse=True)[:5]

# -----------------------------
# Grammar correction
# -----------------------------
def apply_grammar(doc):
    corrected_tokens = [t.text for t in doc]
    grammar_indices = set()

    # --- STEP 1: Pluralize nouns after quantifiers ---
    quantifiers = {"many", "few", "several", "all", "some", "most", "various"}
    pluralized_subjects = set()  # Track which nouns were pluralized

    for i, token in enumerate(doc):
        if token.text.lower() in quantifiers:
            for j in range(i+1, min(i+4, len(doc))):
                if doc[j].is_punct or doc[j].pos_ == "ADP":
                    break
                if doc[j].pos_ == "NOUN" and doc[j].tag_ != "NNS":
                    plural_form = doc[j]._.inflect("NNS")
                    if plural_form:
                        if corrected_tokens[j] != plural_form:
                            corrected_tokens[j] = plural_form
                            grammar_indices.add(j)
                        pluralized_subjects.add(j)
                        break

    # --- STEP 2: Subject-verb agreement ---
    for i, token in enumerate(doc):
        if token.pos_ not in {"VERB", "AUX"}:
            continue
        if token.tag_ == "MD" or any(a.tag_ == "MD" for a in token.ancestors):
            continue

        # Find subject: prefer left children, else previous token
        subj = None
        nsubj = [w for w in token.lefts if w.dep_ == "nsubj"]
        if nsubj:
            subj = nsubj[0]
        elif i > 0 and doc[i-1].pos_ in {"PRON", "NOUN", "PROPN"}:
            subj = doc[i-1]

        if not subj:
            continue

        subj_idx = subj.i
        subj_word = corrected_tokens[subj_idx].lower()
        subj_tag = subj.tag_

        is_singular = subj_tag in {"NN", "NNP"} or subj_word in {"he", "she", "it"}
        is_plural = subj_tag in {"NNS", "NNPS"} or subj_word in {"i", "you", "we", "they"} \
                    or subj_word in {"children", "people", "men", "women"} or subj_tag == "NNS"

        if subj_idx in pluralized_subjects:
            is_singular = False
            is_plural = True

        # --- BE verbs ---
        if token.lemma_ == "be":
            new_val = "is"
            if subj_word == "i":
                new_val = "am"
            elif is_plural:
                new_val = "are"
            if corrected_tokens[i] != new_val:
                corrected_tokens[i] = new_val
                grammar_indices.add(i)

        # --- HAVE verbs ---
        elif token.lemma_ == "have":
            new_val = "has"
            if subj_word == "i" or is_plural:
                new_val = "have"
            if corrected_tokens[i] != new_val:
                corrected_tokens[i] = new_val
                grammar_indices.add(i)

        # --- DO verbs ---
        elif token.lemma_ == "do":
            new_val = "does"
            if subj_word == "i" or is_plural:
                new_val = "do"
            if corrected_tokens[i] != new_val:
                corrected_tokens[i] = new_val
                grammar_indices.add(i)

        # --- Main verbs (present tense) ---
        elif token.pos_ == "VERB" and token.tag_ in {"VB", "VBP", "VBZ"}:
            if token.tag_ in {"VBN", "VBG"}:
                continue
            base = token.lemma_
            inflected = token.text
            if is_singular:
                inflected = token._.inflect("VBZ") or token.text
            elif is_plural:
                inflected = token._.inflect("VBP") or token.text
            if corrected_tokens[i] != inflected:
                corrected_tokens[i] = inflected
                grammar_indices.add(i)

    # --- STEP 3: Capitalization ---
    for i, word in enumerate(corrected_tokens):
        new_word = word
        if i == 0:
            new_word = word.capitalize()
        if word.lower() == "i" and len(word) == 1:
            new_word = "I"
        if i > 0 and corrected_tokens[i-1] in {".", "!", "?"}:
            new_word = word.capitalize()
        if corrected_tokens[i] != new_word:
            corrected_tokens[i] = new_word
            grammar_indices.add(i)

    return corrected_tokens, list(grammar_indices)
# -----------------------------
# Detect errors with confusion matrix
# -----------------------------
def detect_errors(text, nlp, threshold_ratio=1.5):
    """
    Hybrid spelling & real-word error detection using trigram probabilities.
    - Non-word: TextBlob spellcheck
    - Real-word: candidate edit-distance + trigram scoring
    - Confusion sets are always considered and can bypass threshold_ratio
    """
    doc = nlp(text)
    corrected_tokens, grammar_indices = apply_grammar(doc)
    errors = []
    confusion_matrix = {}

    for i, token in enumerate(corrected_tokens):
        low = token.lower()

        # Skip punctuation or grammar-fixed tokens
        if not token.isalpha() or i in grammar_indices:
            continue

        # Skip 'I'
        if low == "i":
            continue

        prev = corrected_tokens[i-1].lower() if i > 0 else None
        prev2 = corrected_tokens[i-2].lower() if i > 1 else None

        # -----------------------------
        # Non-word errors
        if low not in VOCAB:
            try:
                suggestion = Word(low).spellcheck()[0][0]
            except:
                suggestion = low
            errors.append({
                "word": token,
                "index": i,
                "suggestion": suggestion,
                "type": "non-word"
            })
            confusion_matrix[low] = suggestion
            continue

        # -----------------------------
        # Real-word errors
        candidates = []
        # Skip pronouns that should never be flagged
        if low in {"i", "he", "she", "we", "they", "it", "you"}:
            continue

        # 1. Confusion set first (always considered)
        in_confusion_set = False
        if low in CONFUSION_SETS:
            candidates.extend([c for c in CONFUSION_SETS[low] if c != low])
            in_confusion_set = True

        # 2️. Edit-distance 1 from vocab
        candidates.extend([v for v in VOCAB if edit_distance(low, v) == 1 and v != low])

        if not candidates:
            continue

        # Step 2: compute trigram/bigram probability for each candidate
        candidate_probs = []
        for c in candidates:
            prob = 0
            if prev2 and prev:
                tri = TRIGRAM_COUNTS.get((prev2, prev, c), 0)
                bi = BIGRAM_COUNTS.get((prev2, prev), 0)
                prob = (tri + 1) / (bi + VOCAB_SIZE + 1) if bi > 0 else 0
            elif prev:
                bi_count = BIGRAM_COUNTS.get((prev, c), 0)
                uni_count = UNIGRAM_COUNTS.get(prev, 0)
                prob = (bi_count + 1) / (uni_count + VOCAB_SIZE + 1) if uni_count > 0 else 0
            else:
                prob = (UNIGRAM_COUNTS.get(c, 0) + 1) / (TOTAL_UNIGRAMS + VOCAB_SIZE + 1)
            candidate_probs.append({"word": c, "prob": prob})

        # Step 3: compute current word probability
        if prev2 and prev:
            tri = TRIGRAM_COUNTS.get((prev2, prev, low), 0)
            bi = BIGRAM_COUNTS.get((prev2, prev), 0)
            current_prob = (tri + 1) / (bi + VOCAB_SIZE + 1) if bi > 0 else 0
        elif prev:
            bi_count = BIGRAM_COUNTS.get((prev, low), 0)
            uni_count = UNIGRAM_COUNTS.get(prev, 0)
            current_prob = (bi_count + 1) / (uni_count + VOCAB_SIZE + 1) if uni_count > 0 else 0
        else:
            current_prob = (UNIGRAM_COUNTS.get(low, 0) + 1) / (TOTAL_UNIGRAMS + VOCAB_SIZE + 1)

        # Step 4: pick best candidate
        best_candidate = max(candidate_probs, key=lambda x: x["prob"])

        # Step 5: determine if should flag
        flag_error = False
        if in_confusion_set:
            # Always flag if confusion set suggests a different word
            if best_candidate["word"] != low:
                flag_error = True
        else:
            # Otherwise, only flag if candidate significantly more likely
            if best_candidate["word"] != low and best_candidate["prob"] > current_prob * threshold_ratio:
                flag_error = True

        if flag_error:
            errors.append({
                "word": token,
                "index": i,
                "suggestion": best_candidate["word"],
                "type": "real-word"
            })
            confusion_matrix[low] = best_candidate["word"]

    return corrected_tokens, grammar_indices, confusion_matrix, errors

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("✍️ Spelling & Grammar Correction System")
st.markdown("""
**How to use:**  
1. Enter your text  
2. Click **Check Text**  
3. 🔴 Red → spelling or real-word error  
4. 🟢 Green → grammar correction  
5. Click words to see suggestions
""")
# Initialize variables
corrected_tokens = []
grammar_indices = []
errors = []

user_input = st.text_area("**Enter text (max 500 words):**", height=150)

if st.button("🔍 Check Text", use_container_width=True):
    corrected_tokens, grammar_indices, _, errors = detect_errors(user_input, nlp)
    spelling_idx = {e["index"] for e in errors}

    # Highlighted text
    highlighted = []
    for i, tok in enumerate(corrected_tokens):
        if tok in {".", ",", "!", "?", ";", ":"}:
            highlighted.append(tok)
        elif i in spelling_idx:
            highlighted.append(f"<span style='color:red;font-weight:bold'>{tok}</span>")
        elif i in grammar_indices:
            highlighted.append(f"<span style='color:green;font-weight:bold'>{tok}</span>")
        else:
            highlighted.append(tok)

    html_text = ""
    for w in highlighted:
        if w in {".", ",", "!", "?", ";", ":"}:
            html_text = html_text.rstrip() + w + " "
        else:
            html_text += w + " "

    st.subheader("🖍 Highlighted Text")
    st.markdown(html_text.strip(), unsafe_allow_html=True)

    # ---------------- Suggestions panel ----------------
    if errors:  # This now works
        st.subheader("📌 Suggestions")
for e in errors:
    idx = e["index"]
    prev = corrected_tokens[idx-1].lower() if idx > 0 else None
    prev_2 = corrected_tokens[idx-2].lower() if idx > 1 else None
    with st.expander(f"`{e['word']}` → `{e['suggestion']}` ({e['type']})"):
        is_real_word = e["type"] == "real-word"
        candidates = generate_candidates(e['word'], real_word=is_real_word)
        ranked = []

        if is_real_word:
            # For real-word, use trigram score as before
            for c in candidates:
                score = 0
                if prev_2 and prev:
                    score = TRIGRAM_COUNTS.get((prev_2, prev, c), 0)
                elif prev:
                    score = BIGRAM_COUNTS.get((prev, c), 0)
                if score == 0:
                    score = UNIGRAM_COUNTS.get(c, 0)
                ranked.append({"word": c, "score": score})
            ranked = sorted(ranked, key=lambda x: x["score"], reverse=True)[:5]
            for s in ranked:
                st.markdown(f"- **{s['word']}** | score `{round(s['score'], 4)}`")
        else:
            # For non-word, use edit distance
            for c in candidates:
                dist = edit_distance(e['word'].lower(), c)
                ranked.append({"word": c, "edit_distance": dist})
            ranked = sorted(ranked, key=lambda x: x["edit_distance"])[:5]
            for s in ranked:
                st.markdown(f"- **{s['word']}** | edit distance `{s['edit_distance']}`")


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

