from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from transformers import pipeline
from ultralytics import YOLO
import torch.nn.functional as F
import time
import cv2
import numpy as np
import os
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification
import torch
import librosa
import traceback
from moviepy.editor import VideoFileClip

# Initialize models at module level to load them once
model = OllamaLLM(model="deepseek-r1:1.5b")  # change to your model name
facemod = YOLO('best.pt')
whisper = pipeline("automatic-speech-recognition", model="whisper-small/")
sent_class = pipeline(
    task="text-classification",
    model="roberta-base-go_emotions/",
    tokenizer="roberta-base-go_emotions/",
    top_k=None
)
aud_processor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base')
aud_class = Wav2Vec2ForSequenceClassification.from_pretrained('facebook/wav2vec2-base')
aud_class.eval()

def extract_audio(input_video_path: str, output_audio_path: str = 'temp.wav'):
    """
    Extracts audio from a video file and saves it to an output path.
    """
    try:
        if os.path.exists(output_audio_path):
            os.remove(output_audio_path)
        video_clip = VideoFileClip(input_video_path)
        audio_clip = video_clip.audio
        audio_clip.write_audiofile(output_audio_path, logger=None)
        video_clip.close()
        audio_clip.close()
        return output_audio_path
    except Exception as e:
        print(f"Error extracting audio: {e}")
        return None

class VideoEmotionDetector:
    def __init__(self, model: YOLO, window_size: float = 2.5, frame_rate: int = 30):
        self.model = model
        self.window_size = window_size
        self.frame_rate = frame_rate
        self.frames_per_window = int(window_size * frame_rate)
        self.slide_step = int(frame_rate)
        self.emotions = ['surprise', 'happy', 'contempt', 'fear', 'sadness', 'disgust', 'anger']

    def parse_results(self, results):
        try:
            if not results or not hasattr(results[0], 'probs'):
                return {emotion: 0.0 for emotion in self.emotions}
            probs_tensor = results[0].probs.data
            probs_numpy = probs_tensor.cpu().numpy()
            return {emotion: float(prob) for emotion, prob in zip(self.emotions, probs_numpy)}
        except Exception as e:
            print(f"Error parsing results: {str(e)}")
            traceback.print_exc()
            return {emotion: 0.0 for emotion in self.emotions}

    def predict_emotion_for_segment(self, segment_frames: list[np.ndarray], start_frame: int):
        try:
            accumulated_probs = {emotion: 0.0 for emotion in self.emotions}
            valid_frames = 0
            for frame in segment_frames:
                if frame is None:
                    continue
                frame = cv2.resize(frame, (640, 640))
                frame = frame.astype(np.uint8)
                with torch.no_grad():
                    results = self.model(frame, verbose=False)
                probs = self.parse_results(results)
                if any(probs.values()):
                    for emotion in self.emotions:
                        accumulated_probs[emotion] += probs[emotion]
                    valid_frames += 1
            if valid_frames == 0:
                return {emotion: 0.0 for emotion in self.emotions}
            avg_probs = {emotion: prob / valid_frames for emotion, prob in accumulated_probs.items()}
            probs_tensor = torch.tensor([avg_probs[e] for e in self.emotions])
            softmax_probs = F.softmax(probs_tensor, dim=0).numpy()
            return {emotion: float(prob) for emotion, prob in zip(self.emotions, softmax_probs)}
        except Exception as e:
            print(f"Error in emotion prediction: {str(e)}")
            traceback.print_exc()
            return {emotion: 0.0 for emotion in self.emotions}

    def process_video(self, video_path: str):
        print(f"Starting to process video: {video_path}")
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found at {video_path}")
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video file: {video_path}")
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            duration = total_frames / fps
            expected_segments = (total_frames - self.frames_per_window) // self.slide_step + 1
            print(f"Video properties: Total frames: {total_frames}, FPS: {fps}, Duration: {duration:.2f}s, Expected segments: {expected_segments}")
            frames = []
            emotion_predictions = []
            current_frame = 0
            start_time = time.time()
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
                current_frame += 1
                if len(frames) == self.frames_per_window:
                    emotion_probs = self.predict_emotion_for_segment(frames, current_frame - self.frames_per_window)
                    emotion_predictions.append(emotion_probs)
                    progress = (current_frame / total_frames) * 100
                    elapsed_time = time.time() - start_time
                    print(f"Progress: {progress:.1f}% - Segment {len(emotion_predictions)}/{expected_segments}, Time: {elapsed_time:.1f}s")
                    frames = frames[self.slide_step:]
                    if len(emotion_predictions) >= expected_segments:
                        break
            cap.release()
            total_time = time.time() - start_time
            print(f"Processing complete: {len(emotion_predictions)} segments in {total_time:.2f}s")
            return emotion_predictions
        except Exception as e:
            print(f"Error processing video: {str(e)}")
            traceback.print_exc()
            raise
        finally:
            if 'cap' in locals():
                cap.release()

def get_face(video_path: str, window_size: float = 2.5):
    try:
        detector = VideoEmotionDetector(model=facemod, window_size=window_size)
        return [max(pred.items(), key=lambda x: x[1])[0] for pred in detector.process_video(video_path)]
    except Exception as e:
        print(f"Error in get_face: {str(e)}")
        traceback.print_exc()
        return []

def get_aud(audio_path: str, window_size: int = 2):
    aud_class.eval()
    y, sr = librosa.load(audio_path, sr=16000)
    duration = int(librosa.get_duration(y=y, sr=sr))
    ems = []
    for i in range(0, duration, window_size):
        speech, sr = librosa.load(audio_path, sr=16000, duration=window_size, offset=i)
        max_len = 44100
        if len(speech) > max_len:
            speech = speech[:max_len]
        else:
            speech = np.pad(speech, (0, max_len - len(speech)), 'constant')
        x = aud_processor(speech, sampling_rate=16000, return_tensors='pt', padding=True, truncate=True, max_length=max_len)
        with torch.no_grad():
            logits = aud_class(x.input_values).logits
        idx = torch.argmax(logits, dim=-1).item()
        classes = ['fear', 'anger', 'disgust', 'neutral', 'sad', 'surprise', 'happy']
        ems.append(classes[idx])
    return ems

def get_s2t(audio_path: str):
    try:
        return whisper(audio_path)['text']
    except Exception as e:
        print(f"Error in get_s2t: {str(e)}")
        return ""

def get_sent(text: str):
    try:
        outputs = sent_class(text)[0]
        return max(outputs, key=lambda x: x['score'])['label']
    except Exception as e:
        print(f"Error in get_sent: {str(e)}")
        return "unknown"

template = """
Given are the detected facial emotion, speech emotion, the spoken context, and the overall sentiment for a video clip involving a single individual. Compare and contrast how the facial and speech emotions relate to each other and what this combination reveals about the emotional tone of the clip. Then, describe how the overall sentiment emerges from these signals. Present the analysis in a reflective paragraph format that also illustrates the reasoning behind the inference. Avoid using lists or tables. At the end, include a clear disclaimer noting that model predictions may not reflect true human emotions.

Inputs:
Facial Emotion: {face_emotions}
Speech Emotion: {speech_emotion}
Spoken Context: {context}
Detected Sentiment: {sentiment}

Response Format:

"The analysis for the patient interview:" (Put in bold characters)

Begin with a reflection on the facial emotion and what it visually suggests about the person's emotional state. Then, consider the speech emotion and how the tone or content of the words either supports or contrasts with the facial expression. Reflect on how these two emotional signals interact—do they reinforce each other, or do they create ambiguity or tension? After examining both, transition into an explanation of how the overall sentiment arises from the interplay between facial cues and vocal expression, supported by the spoken context. Conclude with a brief interpretation of what the combined emotions might indicate about the person's state of mind.

Insert Disclaimer here:

“Disclaimer: This analysis is based on model-generated predictions and may not accurately represent the true emotions or intent of the individual. Interpretations should be made with caution and in consideration of broader context.”

Response:
"""
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

def process_files(video_path: str):
    """
    Process a video file (only) and return facial emotions, speech emotions,
    context, sentiment, and LLM analysis.
    """
    try:
        # Extract audio from video
        audio_path = extract_audio(video_path)
        if audio_path is None:
            raise RuntimeError("Audio extraction failed")

        # Get signals
        face_emotions = get_face(video_path)
        speech_emotions = get_aud(audio_path)
        context = get_s2t(audio_path)
        sentiment = get_sent(context)

        # Prepare prompt
        prompt_input = {
            "face_emotions": ", ".join(face_emotions),
            "speech_emotion": ", ".join(speech_emotions),
            "context": context,
            "sentiment": sentiment
        }

        # Stream LLM analysis
        analysis = ""
        for chunk in chain.stream(prompt_input):
            analysis += chunk

        return {
            "face_emotions": face_emotions,
            "speech_emotions": speech_emotions,
            "context": context,
            "sentiment": sentiment,
            "analysis": analysis
        }
    except Exception as e:
        print(f"Error in process_files: {str(e)}")
        traceback.print_exc()
        return {"face_emotions": [], "speech_emotions": [], "context": "", "sentiment": "unknown", "analysis": "Error occurred."}