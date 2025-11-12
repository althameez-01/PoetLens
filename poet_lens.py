import os
import warnings
from datetime import datetime
import numpy as np
from PIL import Image
import cv2
import deepface
import torch
import torch.nn.functional as F
from transformers import (
    CLIPModel,
    CLIPProcessor,
    BlipForConditionalGeneration,
    BlipProcessor,
    GPT2LMHeadModel,
    GPT2Tokenizer,
)

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

USE_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODELS_DIR, exist_ok=True)

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
BLIP_MODEL_NAME = "Salesforce/blip-image-captioning-base"
POETRY_MODEL_NAME = "gpt2-medium"

EMOTIONS = [
    "joyful", "sad", "angry", "fearful", 
    "surprised", "disgusted", "neutral",
    "serene", "melancholic", "mysterious",
    "hopeful", "contemplative", "romantic"
]
DEEPFACE_TO_EMOTION = {
    "happy": "joyful",
    "sad": "melancholic",
    "angry": "angry",
    "fear": "fearful",
    "surprise": "surprised",
    "disgust": "disgusted",
    "neutral": "contemplative"
}
EMOTION_STYLES = {
    "joyful": "bright, uplifting imagery with light and warmth",
    "angry": "intense, sharp imagery with fire and storms",
    "melancholic": "soft, wistful imagery with fading light and rain",
    "fearful": "dark, tense imagery with shadows and cold",
    "serene": "calm, peaceful imagery with stillness and nature",
    "contemplative": "quiet, thoughtful imagery with depth and silence",
    "mysterious": "enigmatic, veiled imagery with mist and secrets",
    "hopeful": "gentle, forward-looking imagery with dawn and growth",
    "romantic": "tender, passionate imagery with hearts and roses",
    "surprised": "sudden, vivid imagery with sparks and wonder",
    "disgusted": "harsh, repulsive imagery with decay",
    "neutral": "balanced, observational imagery"
}

print("Loading models...")
device = torch.device(USE_DEVICE)

clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME, cache_dir=MODELS_DIR).to(device)
clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME, cache_dir=MODELS_DIR)

blip_processor = BlipProcessor.from_pretrained(BLIP_MODEL_NAME, cache_dir=MODELS_DIR)
blip_model = BlipForConditionalGeneration.from_pretrained(BLIP_MODEL_NAME, cache_dir=MODELS_DIR).to(device)

poetry_tokenizer = GPT2Tokenizer.from_pretrained(POETRY_MODEL_NAME, cache_dir=MODELS_DIR)
poetry_model = GPT2LMHeadModel.from_pretrained(POETRY_MODEL_NAME, cache_dir=MODELS_DIR).to(device)

if poetry_tokenizer.pad_token is None:
    poetry_tokenizer.pad_token = poetry_tokenizer.eos_token

try:
    from deepface import DeepFace
    HAVE_DEEPFACE = True
    print("DeepFace loaded successfully for facial emotion detection")
except Exception as e:
    HAVE_DEEPFACE = False
    print("DeepFace not available, using CLIP-only detection")

print("Models loaded!\n")


def detect_facial_emotion(image_path: str):
    """Detect facial emotion using DeepFace."""
    if not HAVE_DEEPFACE:
        return None, {}
    
    try:
        result = DeepFace.analyze(
            img_path=image_path,
            actions=['emotion'],
            enforce_detection=False,
            detector_backend='opencv'
        )
        
        if isinstance(result, list):
            result = result[0]
        
        emotions = result.get('emotion', {})
        dominant = result.get('dominant_emotion', 'neutral')
        
        mapped_emotion = DEEPFACE_TO_EMOTION.get(dominant, dominant)
        
        return mapped_emotion, emotions
    except Exception as e:
        print(f"[facial detection error] {e}")
        return None, {}


def generate_caption(image: Image.Image) -> str:
    """Generate image caption using BLIP."""
    try:
        inputs = blip_processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = blip_model.generate(**inputs, max_length=50)
        caption = blip_processor.decode(outputs[0], skip_special_tokens=True)
        return caption
    except Exception as e:
        print(f"[caption error] {e}")
        return "An indescribable scene"


def detect_scene_emotion(image: Image.Image, caption: str):
    """Detect scene-level emotion using CLIP."""
    emotion_prompts = [f"This image feels {e}" for e in EMOTIONS]
    clip_inputs = clip_processor(
        text=emotion_prompts, 
        images=image, 
        return_tensors="pt", 
        padding=True
    ).to(device)

    with torch.no_grad():
        outputs = clip_model(**clip_inputs)
        visual_logits = outputs.logits_per_image.squeeze(0)
        visual_probs = visual_logits.softmax(dim=0).cpu().numpy()

    emotion_probs = list(zip(EMOTIONS, visual_probs.tolist()))
    emotion_probs.sort(key=lambda x: x[1], reverse=True)
    
    return emotion_probs


def detect_emotion(image_path: str):
    """
    Multi-modal emotion detection:
    1. Facial emotion (if face detected) - primary
    2. Scene emotion via CLIP - secondary/context
    3. Fusion for final result
    """
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"[image error] {e}")
        return "neutral", "Unable to load image", [("neutral", 1.0)]
    
    caption = generate_caption(image)
    print(f"Caption: {caption}")

    facial_emotion, facial_scores = detect_facial_emotion(image_path)

    scene_emotions = detect_scene_emotion(image, caption)
    
    if facial_emotion and facial_scores:
        print(f"Face detected! Dominant emotion: {facial_emotion}")
        print(f"Facial scores: {facial_scores}")
        
        emotion_dict = dict(scene_emotions)
        if facial_emotion in emotion_dict:
            for i, (emo, score) in enumerate(scene_emotions):
                if emo == facial_emotion:
                    scene_emotions[i] = (emo, score * 0.3 + 0.7)
                else:
                    scene_emotions[i] = (emo, score * 0.3)
        
        total = sum(s for _, s in scene_emotions)
        scene_emotions = [(e, s/total) for e, s in scene_emotions]
        scene_emotions.sort(key=lambda x: x[1], reverse=True)
        
        primary_emotion = scene_emotions[0][0]
        top_three = scene_emotions[:3]
    else:
        print("No face detected, using scene-based emotion")
        primary_emotion = scene_emotions[0][0]
        top_three = scene_emotions[:3]

    return primary_emotion, caption, top_three


def generate_poem(emotion: str, caption: str, top_emotions: list) -> str:
    """Generate concise 3-4 line poem matching the emotion."""
    
    style = EMOTION_STYLES.get(emotion, "evocative imagery")
    
    prompt = f"""Write a haiku-style short poem (3-4 lines only).

Subject: {caption}
Emotion: {emotion}
Style: {style}

Rules:
- Exactly 3 or 4 lines
- Each line: 6-10 words maximum
- Use {emotion} tone throughout
- No explanations, just the poem

Poem:
"""

    inputs = poetry_tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    temp_map = {
        "angry": 0.85,
        "joyful": 0.8,
        "fearful": 0.75,
        "melancholic": 0.7,
        "serene": 0.65,
        "contemplative": 0.65
    }
    temp = temp_map.get(emotion, 0.75)
    
    try:
        with torch.no_grad():
            outputs = poetry_model.generate(
                inputs,
                max_length=inputs.shape[1] + 80, 
                min_length=inputs.shape[1] + 30,
                temperature=temp,
                top_p=0.85,
                top_k=40,
                do_sample=True,
                num_return_sequences=1,
                no_repeat_ngram_size=3,
                repetition_penalty=1.2,
                pad_token_id=poetry_tokenizer.eos_token_id,
            )
        generated = poetry_tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        print(f"[generation error] {e}")
        fallbacks = {
            "angry": "Thunder in eyes, lightning in veins,\nFury rises like storm-tossed waves,\nSilence breaks with clenched-fist rains.",
            "joyful": "Laughter spills like golden light,\nHearts dance in summer's embrace,\nJoy blooms endless and bright.",
            "melancholic": "Shadows linger where light once lived,\nMemories drift like autumn leaves,\nSilent tears the heart has given.",
            "serene": "Stillness whispers through the pines,\nPeace settles soft as morning mist,\nCalm waters mirror endless skies."
        }
        return fallbacks.get(emotion, "Moments captured in time,\nEmotions painted in light,\nSilent stories sublime.")
    if "Poem:" in generated:
        poem_text = generated.split("Poem:", 1)[1].strip()
    else:
        poem_text = generated[len(prompt):].strip()

    lines = [ln.strip() for ln in poem_text.split('\n') if ln.strip()]
    
    cleaned_lines = []
    skip_keywords = ['write', 'poem', 'line', 'haiku', 'rules', 'subject', 
                     'emotion', 'style', 'note', 'example', '*', '-', 'preface']
    
    for line in lines:
        if any(kw in line.lower() for kw in skip_keywords):
            continue
        if len(line.split()) < 3:
            continue
        if line[0].isdigit() or line.startswith(('•', '>', '#')):
            continue
        cleaned_lines.append(line)
    
    final_lines = cleaned_lines[:4]
    
    if len(final_lines) < 3:
        sentences = poem_text.replace('\n', ' ').split('.')
        final_lines = [s.strip() for s in sentences if s.strip() and len(s.split()) >= 3][:4]
    
    polished = []
    for line in final_lines:
        if line:
            line = line[0].upper() + line[1:] if len(line) > 1 else line.upper()
            line = ' '.join(['I' if word == 'i' else word for word in line.split()])
            line = line.rstrip('.,;:')
            polished.append(line)
    
    poem = '\n'.join(polished[:4]) if polished else "Emotions captured in silence,\nMoments frozen in time,\nStories written in light."
    
    return poem


def format_output(emotion: str, caption: str, top_emotions: list, poem: str, image_path: str):
    """Format the final output."""
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    emotion_text = ", ".join([f"{e.capitalize()} ({p:.0%})" for e, p in top_emotions])
    
    titles = {
        "joyful": "Radiant Moments",
        "angry": "Storm Within",
        "melancholic": "Fading Light",
        "fearful": "Shadow's Edge",
        "serene": "Perfect Stillness",
        "mysterious": "Veiled Whispers",
        "contemplative": "Silent Depths",
        "hopeful": "Dawn's Promise",
        "romantic": "Heart's Echo",
        "surprised": "Lightning Strike",
        "disgusted": "Bitter Truth",
        "neutral": "Quiet Observation"
    }
    title = titles.get(emotion, f"{emotion.capitalize()}")

    output = f"""
╔══════════════════════════════════════════════════════════════╗
║                    POETLENS - EMOTIONAL POETRY               ║
╠══════════════════════════════════════════════════════════════╣
║  Image: {image_name:<53}║
║  Emotion: {emotion.capitalize():<51}║
║  Spectrum: {emotion_text:<49}║
║  Generated: {timestamp:<48}║
╚══════════════════════════════════════════════════════════════╝

Scene: {caption}

─── {title} ───

{poem}

╔══════════════════════════════════════════════════════════════╗
║                    Created with PoetLens                      ║
╚══════════════════════════════════════════════════════════════╝
"""
    return output


def main():
    print("=== PoetLens - Emotional Poetry Generator ===\n")
    image_path = input("Enter path to your image: ").strip()
    
    if not os.path.exists(image_path):
        print("Error: File not found.")
        return
    
    if not image_path.lower().endswith((".png", ".jpg", ".jpeg")):
        print("Error: Use PNG/JPG/JPEG files only.")
        return

    print("\n" + "="*60)
    print("Analyzing image...")
    print("="*60)
    
    emotion, caption, top_emotions = detect_emotion(image_path)
    
    print(f"\n✓ Primary emotion: {emotion}")
    print(f"✓ Top 3: {', '.join([e for e, _ in top_emotions[:3]])}")
    
    print("\n" + "="*60)
    print("Generating poem...")
    print("="*60)
    
    poem = generate_poem(emotion, caption, top_emotions)
    
    formatted = format_output(emotion, caption, top_emotions, poem, image_path)
    
    print("\n" + formatted)
    
    output_path = os.path.splitext(image_path)[0] + "_poem.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(formatted)
    print(f"\n✓ Saved to: {output_path}\n")


if __name__ == "__main__":
    main()