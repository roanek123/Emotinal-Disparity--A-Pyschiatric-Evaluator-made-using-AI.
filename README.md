
# 🎓EMOTIONAL DISPARITY: PYSCHIATRIC EVALUATION ASSISTANT.

This project is the realisation of an AI-assisted emotion analysis system that processes video interviews to extract and analyze facial emotions, speech-based emotions, spoken content, and the overall sentiment. By fusing these modalities, it generates a reflective, LLM-generated summary describing the emotional tone of the interaction, tailored for applications such as healthcare or psychological analysis.

The backend utilizes the following AI models:- 
 - YOLO for facial emotion detection.
 - Wav2Vec2 and Whisper for audio and speech analysis.
 - RoBERTa for sentiment detection.
 - An LLM (like DeepSeek, Llama) for generating a comprehensive narrative. 
 
The frontend is built using Streamlit for ease of use and interaction.

(For any issues for running this system in your system, please refer to the requirements.txt)
## Screenshots

![App Screenshot](https://via.placeholder.com/468x300?text=App+Screenshot+Here)


## 🔍 Features
Our system provides a multitude of features,focusing on acheiving patient pyschiatric evaluation:

- **LLM-Powered Emotional Reasoning**:
  A pre-defined template prompt is sent to the `deepseek-r1:1.5b` LLM using LangChain. This synthesizes results from visual, auditory, and linguistic inputs into a cohesive narrative about the emotional state of the individual.

- **Integrated Streaming UI with Streamlit**:
  A user-friendly Streamlit-based web interface that allows users to upload video files and receive real-time predictions and emotional analysis in a paragraph format.

- **Modular Design**:
  Backend functions are cleanly separated from the frontend logic to ensure better scalability, testing, and reuse.

- **Graceful Error Handling**:
  Designed to catch and log exceptions throughout the pipeline for improved debugging and robustness.



## 🧠 Key Components
This system is composed of several critical modules and components working together:

- **`OllamaLLM` via LangChain**:
  Acts as the LLM reasoning engine, transforming multimodal predictions into human-like interpretive text using structured prompting.

- **YOLOv8 Face Emotion Model (`best.pt`)**:
  A custom object detection model trained to classify facial expressions in individual frames of a video stream, enabling time-based analysis.

- **Whisper Speech Recognition Pipeline**:
  Converts spoken words into text using the open-source Whisper model with a lightweight `pipeline` interface.

- **RoBERTa Go Emotions Classifier**:
  A transformer-based classifier trained to detect fine-grained emotions and sentiments from natural language input (spoken context).

- **Wav2Vec2 Audio Emotion Model**:
  Processes audio signal waveforms and applies sequence classification to identify emotional tone directly from speech.

- **MoviePy and Librosa for Audio Extraction**:
  Extracts audio from uploaded videos and processes it into chunks for emotion and transcription analysis.

- **Streamlit Frontend**:
  Offers an interactive interface that lets users upload video files, see progress updates, and receive both raw and analyzed emotional insights.

- **End-to-End Pipeline (`process_files`)**:
  The orchestration function that chains together all models—extracting audio, classifying facial and vocal emotion, transcribing speech, detecting sentiment, and generating interpretive output through an LLM.

This combination of real-time video processing, audio and text analytics, and natural language reasoning makes the project well-suited for healthcare interviews, patient monitoring, or any scenario where understanding human emotional cues across modalities is critical.
## 🛠️ Installation Guide

Follow the steps below to set up the environment for running the multimodal emotion and sentiment analysis system.

---

###  Prerequisites

- Python 3.10
- Git
- pip
- Sufficient memory (RAM) or use smaller models for limited systems

---

### 📦 Step 1: Clone the Repository

```bash
git clone https://github.com/your-repo/multimodal-emotion-analysis.git
cd multimodal-emotion-analysis
```

---

###  Step 2: Install Python Dependencies

Install the required packages using pip:

```bash
pip install langchain langchain-ollama transformers ultralytics torch opencv-python numpy pillow librosa
pip install moviepy==1.0.3
```

---

###  Step 3: Set Up Ollama and LLaMA Models

1. **Install Ollama:**  
   Follow instructions at: [https://ollama.com/download](https://ollama.com/download)

2. **Run the Ollama server:**

   ```bash
   ollama serve
   ```

3. **(if needed) Fix DNS error:**

   - **Windows:** Change DNS to:
     - Preferred: `8.8.8.8`
     - Alternate: `1.1.1.1`
   - **Linux:**
     ```bash
     sudo nano /etc/resolv.conf
     # Add:
     nameserver 8.8.8.8
     nameserver 1.1.1.1
     ```

   - **macOS:** Go to System Settings > Network > DNS and add:
     - `8.8.8.8`, `1.1.1.1`

4. **Rename LLaMA model (if needed):**
   ```bash
   ollama cp llama3.1:8b llama3.1
   ```

###  Step 4: Install and Configure FFmpeg

1. Download from:  
   [https://www.gyan.dev/ffmpeg/builds/](https://www.gyan.dev/ffmpeg/builds/)

2. Extract the ZIP and locate the `/bin` folder inside it.

3. Add FFmpeg to system PATH:

   - Open **Environment Variables** → **System Variables** → Edit `Path` → Add the full path to `ffmpeg/bin`.

4. Restart VSCode or your terminal.


###  Step 5: Prepare Model Folders

- Download and extract:
  - **Whisper model folder** → into the `whisper/` directory.
  - **Roberta model folder** → into the `roberta/` directory.

Ensure both folders exist and contain the correct HuggingFace files.

---

###  Step 6: Run the Application

Make sure Ollama server is running:

```bash
ollama serve
```

Then launch the Streamlit app:

```bash
streamlit run finals_app.py
```

---

### ⚠️ Notes

- For better performance, keep memory usage minimal or opt for smaller models.
- Ensure that `llamaAPI` folder (created via LangChain Ollama) exists in your working directory.

## 🧪Run Locally
Download the following repository in your local environment.

```bash
git clone https://github.com/your-username/emotion-analysis-app.git
cd emotion-analysis-app
```

install the following packages specified in the "installation guide". Then follow the below steps:
- Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
- run the following command in directory terminal.
```bash
streamlit run finals_app.py
```

Do the following in the http localhost.
- Upload a video file (e.g., .mp4).

- Wait for processing (video and audio will be analyzed).

- Get facial emotion breakdown, speech emotion results, transcribed context, sentiment score, and LLM-based emotional summary.
## 🎉Demo

You can look at the full working of this system through this link.


## 🗺️ WorkFlow

The following describes the workflow of this system:

- Video Input → User uploads video via Streamlit interface.

- Audio Extraction → Extracts audio from the uploaded video.

- Emotion Detection:

    - Facial Emotion: Runs YOLO on video frames.

    - Speech Emotion: Wav2Vec2 on audio segments.

- Transcription & Sentiment:

    - Speech-to-Text: Whisper model.

    - Sentiment: RoBERTa classification on the text.

- LLM Summary → All extracted signals are combined and passed to DeepSeek via LangChain to generate a reflective analysis.



## 🗃️ Usage Example:

Input:
<User Video>

Output:
The analysis for the patient interview:

The facial emotion detected was predominantly “sadness”, which visually indicates a subdued emotional state. The vocal tone, however, reflected a mix of “neutral” and “surprise”, suggesting that while the person maintained a calm voice, there were moments of unexpected emotional expression. These contrasting signals hint at internal emotional complexity. The sentiment analysis classifies the spoken content as “sad”, aligning more with the facial expression. This synthesis of facial and vocal signals reveals a deep emotional burden beneath composed speech.

Disclaimer: This analysis is based on model-generated predictions and may not accurately represent the true emotions or intent of the individual. Interpretations should be made with caution and in consideration of broader context.


## Contributing
Contributions are always welcome!

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

#### ⚠️⚠️⚠️The following system is a realisation of hypothesis of an helper assistant for a real pyshciatrics. As our aim is to help the pschiastrist gain deeper insight into the patient. So that adequete help can be provided to the patient.


