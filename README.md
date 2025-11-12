PoetLens: Emotional Poetry Generator

Developer: Althameez

PoetLens is an intelligent, multi-modal AI system that analyzes an image, detects its dominant emotion, and generates a haiku-style poem reflecting that emotional tone.
It combines DeepFace, CLIP, BLIP, and GPT-2 to interpret both facial and scene-level emotions and translate them into expressive poetry.

Key Features

Multi-Modal Emotion Detection

Detects facial emotions using DeepFace

Identifies scene-based emotions using CLIP (OpenAI)

Fuses results for a more context-aware emotional understanding

AI-Powered Poetry Generation

Generates 3–4 line haiku-style poems using GPT-2 Medium

Each poem reflects the emotion and visual tone of the image

Image Captioning

Uses BLIP to describe the image context for better poetry coherence

Emotion Spectrum Output

Displays emotion probabilities across multiple categories (joyful, melancholic, mysterious, etc.)

Automatic Save

Saves the generated output as a .txt file alongside the image

Tech Stack
Module	Purpose
Python 3.8+	Core language
DeepFace	Facial emotion detection
Transformers (Hugging Face)	Models for CLIP, BLIP, and GPT-2
Torch	Deep learning backend
Pillow (PIL)	Image handling
NumPy	Numerical computations
OpenCV	Image pre-processing and analysis
⚙ Setup Instructions

Clone the Repository
git clone https://github.com/althameez-01/poetlens.git
cd poetlens

Create and Activate Virtual Environment
python -m venv venv
venv\Scripts\activate       # On Windows
source venv/bin/activate    # On macOS/Linux

Install Dependencies
pip install -r requirements.txt


(If requirements.txt not yet created, use below command to install manually)

pip install torch torchvision torchaudio transformers deepface pillow opencv-python numpy

Run PoetLens
python poetlens.py

Input Example
Enter path to your image: C:\Users\YourName\Pictures\portrait.jpg

How It Works

Facial Emotion Detection

If a face is present, DeepFace analyzes and maps it to a corresponding emotional tone (e.g., happy → joyful).

Scene Emotion Detection

CLIP compares the image against emotion prompts like “This image feels joyful.”

It scores each emotion and selects the top three.

Caption Generation

BLIP creates a concise caption describing the visual scene.

Poem Generation

GPT-2 (fine-tuned for creative writing) generates a short poem reflecting the dominant emotion.

Formatted Output

Outputs emotion spectrum, poem, and caption beautifully in the console.

Automatically saves a .txt file alongside the image.

Sample Output
╔══════════════════════════════════════════════════════════════╗
║                    POETLENS - EMOTIONAL POETRY               ║
╠══════════════════════════════════════════════════════════════╣
║  Image: portrait_of_smile                                    ║
║  Emotion: Joyful                                             ║
║  Spectrum: Joyful (78%), Hopeful (12%), Serene (10%)         ║
║  Generated: 2025-11-12 21:45:23                              ║
╚══════════════════════════════════════════════════════════════╝

Scene: A woman smiling warmly under sunlight

─── Radiant Moments ───

Laughter spills like golden light,  
Hearts dance in summer’s embrace,  
Joy blooms endless and bright.  

╔══════════════════════════════════════════════════════════════╗
║                    Created with PoetLens                      ║
╚══════════════════════════════════════════════════════════════╝

File Structure
poetlens/
│
├── models/                   # Cached pretrained models (auto-downloaded)
├── poetlens.py               # Main script
├── requirements.txt          # Dependency list
└── README.md                 # Documentation (this file)

Developer

Name: Althameez
Role: AI Developer / Research Enthusiast
Focus: Vision-Language Models, Emotion Recognition, and Generative AI

Future Enhancements

Web UI using Flask or Streamlit

Voice narration of generated poems

Multilingual emotion-poem synthesis

📊 Visualization dashboard for emotion spectra

🪪 License

This project is licensed under the MIT License.
You are free to use, modify, and distribute it with attribution.
