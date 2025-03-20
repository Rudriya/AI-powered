import torch
import whisper
import nltk
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sentence_transformers import SentenceTransformer, util
from nltk.sentiment import SentimentIntensityAnalyzer

# Download nltk lexicon
nltk.download('vader_lexicon')

# Load Sentiment Analyzer
sia = SentimentIntensityAnalyzer()

# Load RoBERTa Sentiment Model
sentiment_model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
sentiment_tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")

# Load BERT for Semantic Similarity
similarity_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Load Open-Source Whisper Model
whisper_model = whisper.load_model("base")  # Options: tiny, base, small, medium, large

def transcribe_audio(audio_file_path):
    """Convert speech to text using Whisper."""
    result = whisper_model.transcribe(audio_file_path)
    return result["text"]

def analyze_sentiment(text):
    """Analyze sentiment using both VADER and RoBERTa."""
    vader_score = sia.polarity_scores(text)['compound']

    # RoBERTa Sentiment Analysis
    encoded_input = sentiment_tokenizer(text, return_tensors="pt")
    output = sentiment_model(**encoded_input)
    sentiment_score = torch.softmax(output.logits, dim=1).tolist()[0]
    
    sentiment_result = {
        "negative": sentiment_score[0],
        "neutral": sentiment_score[1],
        "positive": sentiment_score[2],
        "vader_compound": vader_score
    }
    
    return sentiment_result

def evaluate_correctness(candidate_answer, model_answer):
    """Evaluate answer correctness using BERT Semantic Similarity."""
    embeddings = similarity_model.encode([candidate_answer, model_answer], convert_to_tensor=True)
    similarity_score = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()
    return similarity_score  # Score between 0 to 1 (higher means more similar)

def detect_hesitation(speech_text):
    """Detect hesitation based on filler words and pauses."""
    filler_words = ["uh", "um", "like", "you know", "I mean", "sort of", "kind of"]
    words = speech_text.lower().split()
    hesitation_count = sum(words.count(word) for word in filler_words)
    
    hesitation_score = hesitation_count / len(words) if len(words) > 0 else 0
    return hesitation_score  # Higher score means more hesitation

def assess_behavioral_metrics(facial_confidence, speech_rate, tone_analysis):
    """Combine behavioral metrics into a confidence score."""
    # Weighted score based on facial confidence, speech rate, and tone
    weight_facial = 0.5
    weight_speech = 0.3
    weight_tone = 0.2

    final_confidence = (facial_confidence * weight_facial) + (speech_rate * weight_speech) + (tone_analysis * weight_tone)
    return final_confidence  # Value between 0 to 1

def assess_candidate(audio_file_path, model_answer, facial_confidence, speech_rate, tone_analysis):
    """Perform full assessment based on all parameters."""
    transcribed_text = transcribe_audio(audio_file_path)
    sentiment_result = analyze_sentiment(transcribed_text)
    correctness_score = evaluate_correctness(transcribed_text, model_answer)
    hesitation_score = detect_hesitation(transcribed_text)
    behavioral_score = assess_behavioral_metrics(facial_confidence, speech_rate, tone_analysis)

    # Compute Final Score (Weighted)
    final_score = (correctness_score * 0.4) + (behavioral_score * 0.3) + ((1 - hesitation_score) * 0.3)

    # Generate Actionable Feedback
    feedback = f"""
    *Technical Analysis:*
    - Correctness: {correctness_score:.2f} (Higher is better)
    
    *Behavioral Insights:*
    - Confidence Level: {behavioral_score:.2f}
    - Hesitation Score: {hesitation_score:.2f} (Lower is better)
    - Sentiment: {sentiment_result}
    
    *Final Score: {final_score:.2f}/1.0*
    Recommendations: Focus on fluency and clarity. Reduce hesitation to improve confidence.
    """

    final_result = {
        "transcribed_text": transcribed_text,
        "sentiment_analysis": sentiment_result,
        "correctness_score": correctness_score,
        "hesitation_score": hesitation_score,
        "behavioral_score": behavioral_score,
        "final_score": final_score,
        "feedback": feedback
    }
    
    return final_result

# ✅ Corrected Main Function
if __name__ == "__main__":
    audio_path = "candidate_response.wav"  # Replace with actual audio file
    model_ans = "A linked list is a data structure where elements are connected using pointers."

    # Example values from video/audio processing models
    facial_conf = 0.85  # Confidence from facial emotion tracking
    speech_rate = 0.78  # Fluency metric from speech processing
    tone_analysis = 0.7  # Tone positivity metric

    result = assess_candidate(audio_path, model_ans, facial_conf, speech_rate, tone_analysis)

    # ✅ Print feedback so you get an output
    print(result["feedback"])
