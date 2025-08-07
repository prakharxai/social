import streamlit as st
import easyocr
from PIL import Image
import re
import io
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from langdetect import detect
import PIL
import os

# Set page configuration
st.set_page_config(page_title="Multilingual Social/Anti-Social Text Classifier", layout="wide")

# Title and description
st.title("Multilingual Social/Anti-Social Text Classifier")
st.write("Upload an image containing text, and the app will perform OCR using EasyOCR to extract the text, detect its language, and classify it as social or antisocial using BERT-based sentiment analysis. You can also customize keyword lists for additional rule-based classification.")

# Initialize EasyOCR reader
@st.cache_resource
def init_easyocr():
    try:
        reader = easyocr.Reader(['en'], gpu=False)  # Default to English, GPU disabled for simplicity
        st.sidebar.success("EasyOCR initialized successfully.")
        return reader
    except Exception as e:
        st.sidebar.error(f"Failed to initialize EasyOCR: {str(e)}")
        return None

reader = init_easyocr()

# Initialize BERT model for sentiment analysis
@st.cache_resource
def load_bert_model():
    try:
        model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        st.sidebar.success("BERT model loaded successfully.")
        return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    except Exception as e:
        st.sidebar.error(f"Failed to load BERT model: {str(e)}")
        return None

sentiment_analyzer = load_bert_model()

# File uploader for image
uploaded_image = st.file_uploader("Upload an image (JPG, PNG, JPEG, BMP)", type=["jpg", "png", "jpeg", "bmp"])

# Custom keyword input
st.subheader("Customize Keyword Lists")
col1, col2 = st.columns(2)
with col1:
    social_keywords_input = st.text_area("Social Keywords (one per line)", "friend\nhappy\nlove\nsupport\nkind\npositive\ncare\nhelp\ncommunity\ntogether\nsharing\nencourage\nrespect\ntrust\njoy\npeace\nharmony\ngratitude")
    social_keywords = [kw.strip().lower() for kw in social_keywords_input.split("\n") if kw.strip()]
with col2:
    antisocial_keywords_input = st.text_area("Antisocial Keywords (one per line)", "hate\nangry\nviolence\ninsult\nbully\nthreat\nabuse\nnegative\ndiscriminate\nharass\noffensive\ncruel\naggressive\nhostile")
    antisocial_keywords = [kw.strip().lower() for kw in antisocial_keywords_input.split("\n") if kw.strip()]

def detect_language(text):
    """
    Detect the language of the extracted text.
    Returns language code (e.g., 'en' for English).
    """
    try:
        return detect(text)
    except:
        return "unknown"

def classify_text_bert(text):
    """
    Classify text using BERT-based sentiment analysis.
    Returns 'Social', 'Antisocial', or 'Neutral' based on sentiment score.
    """
    if sentiment_analyzer is None:
        return "Neutral (BERT model unavailable)"
    try:
        result = sentiment_analyzer(text, truncation=True, max_length=512)[0]
        label = result['label']
        # Convert star-based labels to sentiment
        if '5' in label or '4' in label:
            return "Social"
        elif '1' in label or '2' in label:
            return "Antisocial"
        else:
            return "Neutral"
    except Exception as e:
        st.error(f"Error in BERT classification: {str(e)}")
        return "Neutral"

def classify_text_keywords(text, social_keywords, antisocial_keywords):
    """
    Classify text based on custom keyword matching.
    Returns 'Social', 'Antisocial', or 'Neutral'.
    """
    text = text.lower()
    social_score = sum(len(re.findall(r'\b' + re.escape(kw) + r'\b', text)) for kw in social_keywords)
    antisocial_score = sum(len(re.findall(r'\b' + re.escape(kw) + r'\b', text)) for kw in antisocial_keywords)

    if social_score > antisocial_score and social_score > 0:
        return "Social"
    elif antisocial_score > social_score and antisocial_score > 0:
        return "Antisocial"
    else:
        return "Neutral"

if uploaded_image is not None:
    # Validate and process the image
    try:
        # Check if the uploaded file is a valid image
        image = Image.open(uploaded_image)
        if image.format not in ["JPEG", "PNG", "BMP"]:
            raise ValueError(f"Unsupported image format: {image.format}. Please upload a JPG, PNG, or BMP image.")
        
        # Convert image to RGB if necessary
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Display the uploaded image
        st.image(image, caption="Uploaded Image", use_container_width=True)

        # Perform OCR with EasyOCR
        if reader is None:
            st.error("EasyOCR is not initialized. Please check the debug information in the sidebar.")
        else:
            try:
                # Convert PIL image to a format EasyOCR can process
                image_path = "temp_image.png"
                image.save(image_path)
                results = reader.readtext(image_path, detail=0)  # detail=0 returns only text
                extracted_text = " ".join(results)  # Combine all detected text
                os.remove(image_path)  # Clean up temporary file

                if extracted_text.strip():
                    st.subheader("Extracted Text")
                    st.write(extracted_text)

                    # Detect language
                    lang = detect_language(extracted_text)
                    st.subheader("Detected Language")
                    st.write(f"Language: {lang}")

                    # Perform BERT-based classification
                    bert_classification = classify_text_bert(extracted_text)
                    st.subheader("BERT Sentiment Classification")
                    st.write(f"The text is classified as: **{bert_classification}**")
                    if bert_classification == "Social":
                        st.success("The content promotes positive, social behavior (BERT).")
                    elif bert_classification == "Antisocial":
                        st.error("The content contains antisocial or negative elements (BERT).")
                    else:
                        st.info("The content is neutral (BERT).")

                    # Perform keyword-based classification
                    keyword_classification = classify_text_keywords(extracted_text, social_keywords, antisocial_keywords)
                    st.subheader("Keyword-Based Classification")
                    st.write(f"The text is classified as: **{keyword_classification}**")
                    if keyword_classification == "Social":
                        st.success("The content promotes positive, social behavior (Keywords).")
                    elif keyword_classification == "Antisocial":
                        st.error("The content contains antisocial or negative elements (Keywords).")
                    else:
                        st.info("The content is neutral (Keywords).")
                else:
                    st.warning("No text could be extracted from the image.")
            except Exception as e:
                st.error(f"Error processing OCR with EasyOCR: {str(e)}. Ensure the image contains readable text.")
            
    except ValueError as ve:
        st.error(str(ve))
    except PIL.UnidentifiedImageError:
        st.error("Invalid image file. Please upload a valid JPG, PNG, or BMP image.")
    except Exception as e:
        st.error(f"Error processing the image: {str(e)}")
else:
    st.info("Please upload an image to begin.")