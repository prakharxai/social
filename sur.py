import streamlit as st
from PIL import Image
import re
import io
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from langdetect import detect
import PIL
import os
try:
    from surya.ocr import OCR
    from surya.model.detection.model import load_model as load_det_model, load_processor as load_det_processor
    from surya.model.recognition.model import load_model as load_rec_model
    from surya.model.recognition.processor import load_processor as load_rec_processor
except ImportError as e:
    st.error(f"Failed to import Surya OCR: {str(e)}. Please ensure 'surya-ocr' is installed with 'pip install surya-ocr'. If the issue persists, check for naming conflicts or try reinstalling.")
    st.stop()

# Set page configuration
st.set_page_config(page_title="Multilingual Social/Anti-Social Text Classifier", layout="wide")

# Title and description
st.title("Multilingual Social/Anti-Social Text Classifier")
st.write("Upload an image containing text, and the app will perform OCR using Surya OCR to extract the text, detect its language, and classify it as social or antisocial using BERT-based sentiment analysis. You can also customize keyword lists for additional rule-based classification.")

# Debug section for Surya OCR
st.sidebar.header("Debug Information")
try:
    import surya
    st.sidebar.success(f"Surya OCR version: {surya.__version__}")
except ImportError:
    st.sidebar.error("Surya OCR is not installed. Run 'pip install surya-ocr' in your environment.")

# Initialize Surya OCR
@st.cache_resource
def init_surya_ocr():
    try:
        det_processor = load_det_processor()
        det_model = load_det_model()
        rec_model = load_rec_model()
        rec_processor = load_rec_processor()
        ocr = OCR(langs=["en"], det_model=det_model, det_processor=det_processor, rec_model=rec_model, rec_processor=rec_processor)
        st.sidebar.success("Surya OCR initialized successfully.")
        return ocr
    except Exception as e:
        st.sidebar.error(f"Failed to initialize Surya OCR: {str(e)}")
        return None

surya_ocr = init_surya_ocr()

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

        # Perform OCR with Surya
        if surya_ocr is None:
            st.error("Surya OCR is not initialized. Please check the debug information in the sidebar.")
        else:
            try:
                # Run Surya OCR
                predictions = surya_ocr.run([image])
                extracted_text = " ".join([line.text for prediction in predictions for line in prediction.text_lines])

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
                st.error(f"Error processing OCR with Surya: {str(e)}. Ensure the image contains readable text.")
            
    except ValueError as ve:
        st.error(str(ve))
    except PIL.UnidentifiedImageError:
        st.error("Invalid image file. Please upload a valid JPG, PNG, or BMP image.")
    except Exception as e:
        st.error(f"Error processing the image: {str(e)}")
else:
    st.info("Please upload an image to begin.")