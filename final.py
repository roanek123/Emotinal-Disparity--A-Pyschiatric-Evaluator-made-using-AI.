from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from transformers import pipeline
from ultralytics import YOLO
import torch.nn.functional as F

import time
# import streamlit as st
# from streamlit_video_player import VideoPlayer
import cv2
import numpy as np
from PIL import Image
import io
import base64
import os
from moviepy.editor import VideoFileClip
from transformers import Wav2Vec2Model, Wav2Vec2Processor, Wav2Vec2ForSequenceClassification, pipeline
import torch
import librosa
import traceback

class VideoEmotionDetector:
    def __init__(self, model_path: str, window_size: float = 2.5, frame_rate: int = 30):
        print("Initializing VideoEmotionDetector...")
        self.model_path = model_path
        self.window_size = window_size
        self.frame_rate = frame_rate
        self.frames_per_window = int(window_size * frame_rate)  # 2.5s * 30fps = 75 frames
        self.slide_step = int(frame_rate)  # Slide by one second (30 frames)
        self.emotions = [
            'surprise', 'happy', 'contempt', 'fear', 
            'sadness', 'disgust', 'anger'
        ]
        
        # Check if model file exists
        if not os.path.exists(model_path):
            print(f"ERROR: Model file not found at {model_path}")
            raise FileNotFoundError(f"Model file not found at {model_path}")
            
        self.model = self._load_model(model_path)
        
    def _load_model(self, model_path: str) -> YOLO:
        print(f"Loading model from {model_path}...")
        try:
            model = YOLO(model_path)
            print("Model loaded successfully")
            return model
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise

    def parse_results(self, results) -> dict[str, float]:
        try:
            if not results or not hasattr(results[0], 'probs'):
                return {emotion: 0.0 for emotion in self.emotions}
            
            # Get probabilities as numpy array
            probs_tensor = results[0].probs.data
            probs_numpy = probs_tensor.cpu().numpy()
            
            # Create dictionary mapping emotions to their probabilities
            return {
                emotion: float(prob) 
                for emotion, prob in zip(self.emotions, probs_numpy)
            }
            
        except Exception as e:
            print(f"Error parsing results: {str(e)}")
            traceback.print_exc()
            return {emotion: 0.0 for emotion in self.emotions}

    def predict_emotion_for_segment(self, segment_frames: list[np.ndarray], start_frame: int) -> dict[str, float]:
        try:
            accumulated_probs = {emotion: 0.0 for emotion in self.emotions}
            valid_frames = 0
            
            for i, frame in enumerate(segment_frames):
                if frame is None:
                    continue
                    
                # Preprocess frame
                frame = cv2.resize(frame, (640, 640))
                frame = frame.astype(np.uint8)
                
                # Get predictions
                with torch.no_grad():
                    results = self.model(frame, verbose=False)
                
                # Parse results
                probs = self.parse_results(results)
                if any(probs.values()):  # Check if we got valid probabilities
                    for emotion in self.emotions:
                        accumulated_probs[emotion] += probs[emotion]
                    valid_frames += 1
            
            if valid_frames == 0:
                return {emotion: 0.0 for emotion in self.emotions}
                
            # Calculate average probabilities
            avg_probs = {
                emotion: prob/valid_frames 
                for emotion, prob in accumulated_probs.items()
            }
            
            # Convert to tensors for softmax
            probs_tensor = torch.tensor([avg_probs[e] for e in self.emotions])
            softmax_probs = F.softmax(probs_tensor, dim=0).numpy()
            
            # Create final probability dictionary
            final_probs = {
                emotion: float(prob) 
                for emotion, prob in zip(self.emotions, softmax_probs)
            }
            
            # Get the dominant emotion
            dominant_emotion = max(final_probs.items(), key=lambda x: x[1])
            
            print(f"\nSegment starting at frame {start_frame}:")
            print(f"Dominant emotion: {dominant_emotion[0]} ({dominant_emotion[1]:.3f})")
            print("All probabilities:")
            for emotion, prob in final_probs.items():
                print(f"{emotion}: {prob:.3f}")
                
            return final_probs
            
        except Exception as e:
            print(f"Error in emotion prediction: {str(e)}")
            traceback.print_exc()
            return {emotion: 0.0 for emotion in self.emotions}

    def process_video(self, video_path: str) -> list[dict[str, float]]:
        print(f"Starting to process video: {video_path}")

        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found at {video_path}")

        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video file: {video_path}")

            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            duration = total_frames / fps

            # Calculate expected segments based on sliding window behavior
            expected_segments = (total_frames - self.frames_per_window) // self.slide_step + 1

            print(f"Video properties:")
            print(f"Total frames: {total_frames}")
            print(f"FPS: {fps}")
            print(f"Duration: {duration:.2f} seconds")
            print(f"Expected segments: {expected_segments}")

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

                # Process when we have enough frames for a window
                if len(frames) == self.frames_per_window:
                    # Predict emotion for current segment
                    segment_start_frame = current_frame - self.frames_per_window
                    emotion_probs = self.predict_emotion_for_segment(frames, segment_start_frame)
                    emotion_predictions.append(emotion_probs)

                    # Progress update
                    progress = (current_frame / total_frames) * 100
                    elapsed_time = time.time() - start_time
                    print(f"Progress: {progress:.1f}% - Segment {len(emotion_predictions)}/{expected_segments}")
                    print(f"Time elapsed: {elapsed_time:.1f}s")

                    # Slide window forward by one second
                    frames = frames[self.slide_step:]

                    # Break the loop if we've reached the expected number of segments
                    if len(emotion_predictions) >= expected_segments:
                        break

            cap.release()

            total_time = time.time() - start_time
            print(f"\nProcessing complete:")
            print(f"Processed {len(emotion_predictions)} segments in {total_time:.2f} seconds")
            print(f"Average time per segment: {total_time/len(emotion_predictions):.2f} seconds")

            return emotion_predictions

        except Exception as e:
            print(f"Error processing video: {str(e)}")
            traceback.print_exc()
            raise
        finally:
            if 'cap' in locals():
                cap.release()


template = """
Given are the detected facial emotion, speech emotion, the spoken context, and the overall sentiment for a video clip involving a single individual. Compare and contrast how the facial and speech emotions relate to each other and what this combination reveals about the emotional tone of the clip. Then, describe how the overall sentiment emerges from these signals. Present the analysis in a reflective paragraph format that also illustrates the reasoning behind the inference. Avoid using lists or tables. At the end, include a clear disclaimer noting that model predictions may not reflect true human emotions.

Inputs:
Facial Emotion: {face_emotions}
Speech Emotion: {speech_emotion}
Spoken Context: {context}
Detected Sentiment: {sentiment}

Response Format:

'The analysis for the patient interview:'

Begin with a reflection on the facial emotion and what it visually suggests about the person's emotional state. Then, consider the speech emotion and how the tone or content of the words either supports or contrasts with the facial expression. Reflect on how these two emotional signals interact—do they reinforce each other, or do they create ambiguity or tension? After examining both, transition into an explanation of how the overall sentiment arises from the interplay between facial cues and vocal expression, supported by the spoken context. Conclude with a brief interpretation of what the combined emotions might indicate about the person's state of mind.

Insert Disclaimer here:

“Disclaimer: This analysis is based on model-generated predictions and may not accurately represent the true emotions or intent of the individual. Interpretations should be made with caution and in consideration of broader context.”

Response:

"""

# Initialize Ollama
#model = OllamaLLM(model="llama3.1")
model = OllamaLLM(model="deepseek-r1:1.5b")
# facemod=YOLO('best.pt')
whisper=pipeline("automatic-speech-recognition", model="whisper-small/") 
sent_class=pipeline(
    task="text-classification",
    model="roberta-base-go_emotions/",
    tokenizer="roberta-base-go_emotions/",
    top_k=None
)
# aud_processor=Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base')
# aud_class=Wav2Vec2ForSequenceClassification.from_pretrained('final/')
aud_processor=Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base')
aud_class=Wav2Vec2ForSequenceClassification.from_pretrained('facebook/wav2vec2-base')
aud_class.eval()

def get_face(path, window_size=2):
    try:
        # Define paths
        model_path = "best.pt"
        video_path = path
        
        # Initialize detector
        detector = VideoEmotionDetector(
            model_path=model_path,
            window_size=window_size,
            frame_rate=30
        )
        
        # Process video
        predictions = detector.process_video(video_path)
        ems=[]
        # Print final summary
        print("\nFinal Results Summary:")
        for i, prediction in enumerate(predictions):
            dominant_emotion = max(prediction.items(), key=lambda x: x[1])
            ems.append(dominant_emotion[0])
            print(f"Segment {i+1}/{len(predictions)}: {dominant_emotion[0]} ({dominant_emotion[1]:.3f})")
        return ems
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        traceback.print_exc()

def get_aud(path='temp.wav', window_size=2):
    aud_class.eval()
    y,sr=librosa.load(path,sr=16000)
    d=int(librosa.get_duration(y=y,sr=sr))
    ems=[]
    for i in range(0,d,window_size):
        speech,sr=librosa.load(path,sr=16000, duration=window_size, offset=i)
        max_length=44100
        if len(speech) > max_length:
            speech=speech[:max_length]
        else:
            speech=np.pad(speech, (0,max_length-len(speech)), 'constant')
        x=aud_processor(speech,sampling_rate=16000, return_tensors='pt',
                                padding=True,
                                truncate=True,
                                max_length=44100
            )
        xr=x.input_values
        with torch.no_grad():
            o=aud_class(xr)
        p = torch.argmax(o.logits, dim=-1).detach().cpu().numpy()
        classes=['fear','anger','disgust','neutral','sad', 'surprise','happy']
        ems.append(classes[p.item()])
    return ems

def extract_audio(input_video_path, output_audio_path='temp.wav'): # use of this function??
    try:
        # Load the video file
        video_clip = VideoFileClip(input_video_path)
        
        # Extract the audio from the video clip
        audio_clip = video_clip.audio
        
        # Write the audio to a separate file
        audio_clip.write_audiofile(output_audio_path)
        
        print(f"Audio extraction successful! Audio saved to {output_audio_path}")
        
        # Close the video and audio clips
        video_clip.close()
        audio_clip.close()
    
    except Exception as e:
        print(f"An error occurred during audio extraction: {str(e)}")

def get_s2t(path='temp.wav'):
    r=whisper(path)
    return r['text']

def get_sent(text):
    model_outputs=sent_class(text)[0]
    top_output = max(model_outputs, key=lambda x: x['score'])
    return top_output['label']

path="Q1.mp4"
prompt=ChatPromptTemplate.from_template(template)
chain = prompt | model
face=get_face(path)
aud=extract_audio(path) # what is the use of this??
speech=get_aud()
c=get_s2t()
s=get_sent(c)
print({"face_emotions": face, "speech_emotion":speech, "context":c, "sentiment":s})
for result in chain.stream({"face_emotions": face, "speech_emotion":speech, "context":c, "sentiment":s}):
    print(result, end="", flush=True)

print()