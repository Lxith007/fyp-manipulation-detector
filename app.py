import streamlit as st
import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertForSequenceClassification
from PIL import Image, ImageEnhance, ImageFilter
import pytesseract
import pickle
import re
import os

st.set_page_config(page_title="Manipulation Detector", layout="centered")

@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "model")
    model = BertForSequenceClassification.from_pretrained(model_path, output_attentions=True)
    model.eval()
    tokenizer = BertTokenizer.from_pretrained(model_path)
    with open(os.path.join(model_path, "label_encoder.pkl"), "rb") as f:
        le = pickle.load(f)
    return model, tokenizer, le

model, tokenizer, le = load_model()

def clean_text(text):
    emoji_pattern = re.compile("[" 
        u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F9FF"
        u"\U00002702-\U000027B0"
        "]+", flags=re.UNICODE)
    text = emoji_pattern.sub("", text)
    text = re.sub(r"\b\d{1,2}:\d{2}\b", "", text)
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        line = line.strip()
        if len(line) < 3:
            continue
        alpha_ratio = sum(c.isalpha() for c in line) / max(len(line), 1)
        if alpha_ratio < 0.3:
            continue
        cleaned.append(line)
    return "\n".join(cleaned)

def preprocess_image(image):
    image = image.convert("L")
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2.0)
    image = image.filter(ImageFilter.SHARPEN)
    w, h = image.size
    image = image.resize((w * 2, h * 2), Image.LANCZOS)
    return image

def predict(text):
    encoding = tokenizer(text, max_length=128, padding="max_length", truncation=True, return_tensors="pt")
    with torch.no_grad():
        out = model(input_ids=encoding["input_ids"], attention_mask=encoding["attention_mask"])
        probs = F.softmax(out.logits, dim=1).squeeze().numpy()
        attentions = out.attentions
    predicted = le.classes_[probs.argmax()]
    neutral_idx = list(le.classes_).index("neutral")
    manip_score = float(1 - probs[neutral_idx])
    tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"][0])
    avg_attention = torch.stack(attentions).squeeze(1).mean(dim=0).mean(dim=0)
    cls_attention = avg_attention[0].numpy()
    return predicted, manip_score, probs, tokens, cls_attention, encoding["attention_mask"][0].numpy()

def highlight_text(tokens, attention_weights, mask):
    words, weights = [], []
    current_word, current_weight = "", 0
    for token, weight, m in zip(tokens, attention_weights, mask):
        if m == 0:
            break
        if token in ["[CLS]", "[SEP]", "[PAD]"]:
            continue
        if token.startswith("##"):
            current_word += token[2:]
            current_weight = max(current_weight, weight)
        else:
            if current_word:
                words.append(current_word)
                weights.append(current_weight)
            current_word = token
            current_weight = weight
    if current_word:
        words.append(current_word)
        weights.append(current_weight)
    if not words:
        return ""
    max_w = max(weights) if max(weights) > 0 else 1
    min_w = min(weights)
    html = "<div style='line-height:2.4;font-size:16px;padding:14px;background:#1e1e1e;border-radius:8px;'>"
    for word, weight in zip(words, weights):
        norm = (weight - min_w) / (max_w - min_w + 1e-9)
        if len(word) <= 1 or not any(c.isalpha() for c in word):
            html += f"<span style='color:#aaa;margin:2px'>{word}</span> "
        elif norm > 0.55:
            html += f"<span style='background:rgba(231,76,60,{norm:.2f});padding:2px 5px;border-radius:4px;margin:2px;color:white;font-weight:700'>{word}</span> "
        elif norm > 0.25:
            html += f"<span style='background:rgba(230,126,34,{norm:.2f});padding:2px 5px;border-radius:4px;margin:2px;color:white'>{word}</span> "
        else:
            html += f"<span style='color:#aaa;margin:2px'>{word}</span> "
    html += "</div>"
    return html

class_colours = {
    "guilt_tripping": "#e74c3c",
    "gaslighting": "#8e44ad",
    "love_bombing": "#e91e8c",
    "direct_coercion": "#e67e22",
    "charm_flattery": "#f1c40f",
    "passive_aggressive": "#1abc9c",
    "neutral": "#2ecc71",
}

def show_results(predicted, manip_score, probs, tokens, attention, mask):
    colour = class_colours.get(predicted, "#999")
    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Predicted type**")
        st.markdown(f"<span style='font-size:22px;font-weight:700;color:{colour}'>{predicted.replace('_', ' ').title()}</span>", unsafe_allow_html=True)
    with col2:
        st.markdown("**Manipulation score**")
        st.markdown(f"<span style='font-size:22px;font-weight:700;color:{colour}'>{manip_score:.1%}</span>", unsafe_allow_html=True)
    st.progress(manip_score)
    st.divider()
    st.markdown("**Why did the model predict this?**")
    st.caption("Words in red had the highest influence on the prediction. Orange words had moderate influence.")
    highlighted = highlight_text(tokens, attention, mask)
    if highlighted:
        st.markdown(highlighted, unsafe_allow_html=True)
    st.divider()
    st.markdown("**Confidence breakdown**")
    for cls, prob in sorted(zip(le.classes_, probs), key=lambda x: -x[1]):
        col_a, col_b = st.columns([3, 7])
        with col_a:
            st.markdown(f"<small>{cls.replace('_', ' ').title()}</small>", unsafe_allow_html=True)
        with col_b:
            st.progress(float(prob), text=f"{prob:.1%}")

st.title("Emotional Manipulation Detector")
st.markdown("Detect emotional manipulation in online conversations using AI.")
st.divider()

tab1, tab2 = st.tabs(["Type or paste text", "Upload screenshot"])

with tab1:
    st.markdown("#### Enter a conversation")
    text_input = st.text_area(label="Conversation", placeholder="A: After everything I have done for you...\nB: I am just busy.\nA: I guess I never mattered to you.", height=200, label_visibility="collapsed")
    if st.button("Analyse", key="btn_text", type="primary"):
        if text_input.strip():
            cleaned = clean_text(text_input)
            with st.spinner("Analysing..."):
                predicted, manip_score, probs, tokens, attention, mask = predict(cleaned)
            show_results(predicted, manip_score, probs, tokens, attention, mask)
        else:
            st.warning("Please enter a conversation first.")

with tab2:
    st.markdown("#### Upload a screenshot")
    st.caption("Works with WhatsApp, iMessage, Instagram, and other chat apps.")
    uploaded_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"], label_visibility="collapsed")
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded screenshot", use_container_width=True)
        with st.spinner("Processing image and extracting text..."):
            processed = preprocess_image(image)
            raw_text = pytesseract.image_to_string(processed)
            extracted_text = clean_text(raw_text)
        st.divider()
        st.markdown("#### Extracted text")
        st.caption("Timestamps, symbols and emojis have been automatically removed. Edit if needed.")
        edited_text = st.text_area(label="Extracted text", value=extracted_text.strip(), height=180, label_visibility="collapsed")
        if st.button("Analyse", key="btn_image", type="primary"):
            if edited_text.strip():
                with st.spinner("Analysing..."):
                    predicted, manip_score, probs, tokens, attention, mask = predict(edited_text)
                show_results(predicted, manip_score, probs, tokens, attention, mask)
            else:
                st.warning("No text found in image. Try a clearer screenshot.")

st.divider()
st.caption("CB011219 - Lasith Siriwardena | Final Year Project | APIIT / University of Staffordshire")
