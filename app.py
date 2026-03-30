"""
Rice Crop Stage Detector — Gradio App
======================================
Model  : MobileNetV2 (transfer learning) trained on 4 rice crop stages
Input  : 224x224 RGB image
Classes: flowering, germination, noise, tillering
"""

import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image

# ─── CONFIG ──────────────────────────────────────────────────
MODEL_PATH = "rice_stage_model_v2_with_noise.keras"
IMG_SIZE = 224
CONFIDENCE_THRESHOLD = 0.70

# Class labels (must match the order from training: alphabetical by folder name)
CLASS_LABELS = ["flowering", "germination", "noise", "tillering"]

# Telugu translations for each stage
TELUGU_LABELS = {
    "flowering":   "పూత దశ (Flowering)",
    "germination": "మొలకెత్తడం (Germination)",
    "tillering":   "పిలకలు వేయడం (Tillering)",
    "noise":       "గుర్తించలేని చిత్రం (Unrecognized)"
}

# Detailed stage descriptions
STAGE_INFO = {
    "flowering": {
        "description": "The rice plant is in the flowering/heading stage. Panicles have emerged and pollination is occurring.",
        "description_te": "వరి మొక్క పూత దశలో ఉంది. కంకులు బయటకు వచ్చి పరాగసంపర్కం జరుగుతోంది.",
        "advisory": "Ensure adequate water supply. Avoid pesticide spraying during active flowering. Monitor for neck blast disease.",
        "advisory_te": "తగినంత నీటి సరఫరా ఉండేలా చూడండి. పూత సమయంలో పురుగుమందులు పిచికారీ చేయకండి. మెడవిరుపు తెగులు కోసం గమనించండి."
    },
    "germination": {
        "description": "The rice seeds are in the germination/seedling stage. Young shoots are emerging from the soil.",
        "description_te": "వరి విత్తనాలు మొలకెత్తే దశలో ఉన్నాయి. చిన్న మొక్కలు మట్టి నుండి బయటకు వస్తున్నాయి.",
        "advisory": "Maintain thin water layer (2-3cm). Watch for seedling blight and case worm. Ensure proper nursery management.",
        "advisory_te": "సన్నని నీటి పొర (2-3 సెం.మీ) ఉంచండి. మొలక తెగులు మరియు కేస్ వార్మ్ కోసం గమనించండి."
    },
    "tillering": {
        "description": "The rice plant is actively producing tillers (side shoots). This is a critical growth phase.",
        "description_te": "వరి మొక్క పిలకలు వేస్తోంది. ఇది కీలకమైన పెరుగుదల దశ.",
        "advisory": "Apply nitrogen fertilizer to promote tillering. Maintain 5cm water depth. Scout for stem borer and leaf folder.",
        "advisory_te": "పిలకలు పెరగడానికి నత్రజని ఎరువు వేయండి. 5 సెం.మీ నీటి లోతు ఉంచండి. కాండం తొలుచు పురుగు కోసం గమనించండి."
    },
    "noise": {
        "description": "The uploaded image does not appear to be a recognizable rice crop stage.",
        "description_te": "అప్‌లోడ్ చేసిన చిత్రం గుర్తించదగిన వరి పంట దశగా కనిపించడం లేదు.",
        "advisory": "Please upload a clear photo of your rice crop taken from close range.",
        "advisory_te": "దయచేసి మీ వరి పంట యొక్క స్పష్టమైన ఫోటోను దగ్గరి నుండి తీసి అప్‌లోడ్ చేయండి."
    }
}

# ─── LOAD MODEL ──────────────────────────────────────────────
print("Loading rice stage classification model...")
model = tf.keras.models.load_model(MODEL_PATH)
print(f"Model loaded successfully. Input shape: {model.input_shape}")


# ─── PREDICTION FUNCTION ────────────────────────────────────
def classify_crop_stage(input_image):
    """
    Takes a PIL Image, preprocesses it, runs inference,
    and returns structured results.
    """
    if input_image is None:
        return "No image provided", "", "", "", {}

    # Preprocess
    img = input_image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    predictions = model.predict(img_array, verbose=0)
    class_idx = int(np.argmax(predictions[0]))
    confidence = float(np.max(predictions[0]))

    # Build confidence dict for all classes
    confidence_scores = {
        TELUGU_LABELS.get(cls, cls): float(predictions[0][i])
        for i, cls in enumerate(CLASS_LABELS)
    }

    # Check threshold
    if confidence < CONFIDENCE_THRESHOLD or CLASS_LABELS[class_idx] == "noise":
        stage = "noise"
        result_text = (
            f"**Result:** Not a recognized rice crop stage\n\n"
            f"**Confidence:** {confidence*100:.1f}% (below {CONFIDENCE_THRESHOLD*100:.0f}% threshold)\n\n"
            f"Please upload a clear photo of your rice crop."
        )
    else:
        stage = CLASS_LABELS[class_idx]
        info = STAGE_INFO[stage]
        result_text = (
            f"**Detected Stage:** {TELUGU_LABELS[stage]}\n\n"
            f"**Confidence:** {confidence*100:.1f}%\n\n"
            f"---\n\n"
            f"**Description:**\n{info['description']}\n\n"
            f"**వివరణ:**\n{info['description_te']}\n\n"
            f"---\n\n"
            f"**Advisory:**\n{info['advisory']}\n\n"
            f"**సలహా:**\n{info['advisory_te']}"
        )

    return result_text, confidence_scores


# ─── GRADIO INTERFACE ────────────────────────────────────────
with gr.Blocks(
    title="Rice Crop Stage Detector",
    theme=gr.themes.Soft(primary_hue="green")
) as demo:

    gr.Markdown(
        """
        # 🌾 Rice Crop Stage Detector
        ### వరి పంట దశ గుర్తింపు

        Upload a photo of your rice crop to identify its current growth stage.
        మీ వరి పంట ఫోటోను అప్‌లోడ్ చేసి ప్రస్తుత దశను తెలుసుకోండి.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(
                type="pil",
                label="Upload Rice Crop Image / వరి పంట చిత్రం అప్‌లోడ్ చేయండి",
                sources=["upload", "webcam"],
                height=350
            )
            submit_btn = gr.Button("🔍 Detect Stage / దశ గుర్తించండి", variant="primary", size="lg")

        with gr.Column(scale=1):
            result_text = gr.Markdown(label="Detection Result")
            confidence_chart = gr.Label(
                label="Confidence Scores / నమ్మకం స్కోర్లు",
                num_top_classes=4
            )

    submit_btn.click(
        fn=classify_crop_stage,
        inputs=[input_image],
        outputs=[result_text, confidence_chart]
    )

    gr.Markdown(
        """
        ---
        **Model:** MobileNetV2 (Transfer Learning) · **Classes:** Flowering, Germination, Tillering, Noise
        · **Threshold:** 70% confidence · **Input:** 224×224 RGB
        """
    )

# ─── LAUNCH ──────────────────────────────────────────────────
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
