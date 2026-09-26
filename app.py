# import os
# os.environ["USE_TF"] = "0"

# import streamlit as st
# import torch
# import torch.nn as nn
# import librosa
# import numpy as np
# from PIL import Image
# from transformers import AutoImageProcessor, SiglipForImageClassification
# import matplotlib.pyplot as plt


# # ============================================================
# # CONFIGURATION
# # ============================================================

# BINARY_MODEL_PATH = r"C:\Users\gonth\Downloads\siglip2_asthma_healthy_final.pt"
# MULTICLASS_MODEL_PATH = r"C:\Users\gonth\Downloads\siglip2_asthma_multiclass_final.pt"

# MODEL_ID = "google/siglip2-base-patch16-224"

# IMG_SIZE = 224
# TARGET_SR = 22050
# FIXED_DURATION = 6.0
# N_FFT = 2048
# HOP_LENGTH = 512
# DB_FLOOR = -80.0

# BINARY_CLASSES = ["Healthy", "Asthma"]

# # IMPORTANT:
# # This follows LabelEncoder().fit(categories) from the multiclass notebook.
# # Alphabetical order is therefore:
# MULTICLASS_CLASSES = [
#     "Bronchial",
#     "asthma",
#     "copd",
#     "healthy",
#     "pneumonia",
# ]

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# # ============================================================
# # PAGE CONFIG
# # ============================================================

# st.set_page_config(
#     page_title="Respiratory Sound Classification",
#     page_icon="🫁",
#     layout="wide",
# )

# st.title("🫁 Respiratory Sound Classification")
# st.caption("SigLIP-2 + STFT representation")

# st.sidebar.header("Configuration")

# task = st.sidebar.radio(
#     "Select classification task",
#     [
#         "Binary — Asthma vs Healthy",
#         "Multiclass — Lung Disease",
#     ],
# )

# if task.startswith("Binary"):
#     model_path = BINARY_MODEL_PATH
#     class_names = BINARY_CLASSES
#     num_labels = 1
# else:
#     model_path = MULTICLASS_MODEL_PATH
#     class_names = MULTICLASS_CLASSES
#     num_labels = 5

# st.sidebar.markdown("### Selected model")
# st.sidebar.code(model_path)

# st.sidebar.markdown(f"**Device:** `{DEVICE}`")


# # ============================================================
# # PREPROCESSING — EXACTLY MATCHING THE NOTEBOOK
# # ============================================================

# def fix_length(y, sr, duration=FIXED_DURATION):
#     target_len = int(sr * duration)

#     if len(y) >= target_len:
#         start = (len(y) - target_len) // 2
#         return y[start:start + target_len]

#     pad = target_len - len(y)
#     return np.pad(
#         y,
#         (pad // 2, pad - pad // 2),
#         mode="constant"
#     )


# def audio_to_stft_image(y, sr, img_size=IMG_SIZE):
#     """
#     Same preprocessing used in both notebooks.

#     WAV
#       -> resample to 22050 Hz
#       -> center crop / zero pad to 6 seconds
#       -> STFT
#       -> log magnitude dB
#       -> [-80, 0] clipping and normalization
#       -> delta
#       -> delta-delta
#       -> 3-channel image
#       -> resize 224x224
#       -> float32 [0,1]
#     """

#     if sr != TARGET_SR:
#         y = librosa.resample(
#             y,
#             orig_sr=sr,
#             target_sr=TARGET_SR
#         )
#         sr = TARGET_SR

#     y = fix_length(y, sr)

#     stft = librosa.stft(
#         y,
#         n_fft=N_FFT,
#         hop_length=HOP_LENGTH
#     )

#     log_mag = librosa.amplitude_to_db(
#         np.abs(stft),
#         ref=np.max
#     )

#     log_mag_n = (
#         np.clip(log_mag, DB_FLOOR, 0.0) / (-DB_FLOOR)
#     ) + 1.0

#     delta = librosa.feature.delta(log_mag)
#     delta2 = librosa.feature.delta(
#         log_mag,
#         order=2
#     )

#     d_clip = 6.0

#     delta_n = (
#         np.clip(delta, -d_clip, d_clip) + d_clip
#     ) / (2 * d_clip)

#     delta2_n = (
#         np.clip(delta2, -d_clip, d_clip) + d_clip
#     ) / (2 * d_clip)

#     img = np.stack(
#         [log_mag_n, delta_n, delta2_n],
#         axis=-1
#     ).astype(np.float32)

#     img_pil = Image.fromarray(
#         (img * 255).astype(np.uint8)
#     ).resize(
#         (img_size, img_size),
#         Image.BILINEAR
#     )

#     return np.array(img_pil).astype(np.float32) / 255.0


# # ============================================================
# # MODEL LOADING
# # ============================================================

# @st.cache_resource
# def load_processor():
#     return AutoImageProcessor.from_pretrained(MODEL_ID)


# def clean_state_dict(state_dict):
#     """
#     Removes common wrappers such as:
#       module.
#       model.
#     when the saved checkpoint contains a state_dict.
#     """

#     cleaned = {}

#     for key, value in state_dict.items():

#         new_key = key

#         if new_key.startswith("module."):
#             new_key = new_key[len("module."):]

#         # If the checkpoint was saved from the notebook's
#         # VFMBinaryClassifier / VFMClassifier wrapper.
#         if new_key.startswith("model."):
#             new_key = new_key[len("model."):]

#         cleaned[new_key] = value

#     return cleaned


# @st.cache_resource
# def load_model(model_path, num_labels):
#     """
#     Reconstruct the exact SigLIP-2 image-classification model
#     and load the fine-tuned checkpoint.

#     Supports common .pt formats:
#       1. raw state_dict
#       2. {"state_dict": ...}
#       3. {"model_state_dict": ...}
#     """

#     model = SiglipForImageClassification.from_pretrained(
#         MODEL_ID,
#         num_labels=num_labels,
#         ignore_mismatched_sizes=True,
#         attn_implementation="eager",
#     )

#     checkpoint = torch.load(
#         model_path,
#         map_location="cpu",
#         weights_only=False
#     )

#     if isinstance(checkpoint, nn.Module):
#         model = checkpoint

#     elif isinstance(checkpoint, dict):

#         if "state_dict" in checkpoint:
#             state_dict = checkpoint["state_dict"]

#         elif "model_state_dict" in checkpoint:
#             state_dict = checkpoint["model_state_dict"]

#         else:
#             # Assume the dictionary itself is the state dict.
#             state_dict = checkpoint

#         state_dict = clean_state_dict(state_dict)

#         missing, unexpected = model.load_state_dict(
#             state_dict,
#             strict=False
#         )

#         if missing:
#             st.warning(
#                 f"Checkpoint loaded with {len(missing)} missing keys."
#             )

#         if unexpected:
#             st.warning(
#                 f"Checkpoint contained {len(unexpected)} unexpected keys."
#             )

#     else:
#         raise ValueError(
#             "Unsupported .pt checkpoint format. "
#             "Expected a state_dict, checkpoint dictionary, or nn.Module."
#         )

#     model.to(DEVICE)
#     model.eval()

#     return model


# # ============================================================
# # SPECTROGRAM DISPLAY
# # ============================================================

# def show_spectrogram(img):
#     fig, ax = plt.subplots(figsize=(10, 4))

#     ax.imshow(
#         img[..., 0],
#         origin="lower",
#         aspect="auto",
#         cmap="magma"
#     )

#     ax.set_title("Log-Magnitude STFT")
#     ax.set_xlabel("Time")
#     ax.set_ylabel("Frequency")

#     plt.tight_layout()

#     return fig


# # ============================================================
# # INFERENCE
# # ============================================================

# def predict(model, processor, img, binary=False):

#     img_uint8 = (img * 255).astype(np.uint8)

#     inputs = processor(
#         images=img_uint8,
#         return_tensors="pt"
#     )

#     pixel_values = inputs["pixel_values"].to(DEVICE)

#     with torch.no_grad():

#         outputs = model(
#             pixel_values=pixel_values
#         )

#         logits = outputs.logits

#     if binary:

#         # Notebook:
#         # sigmoid(logit) >= 0.5 -> Asthma (1)
#         asthma_probability = torch.sigmoid(
#             logits.squeeze(-1)
#         ).item()

#         probabilities = np.array([
#             1.0 - asthma_probability,
#             asthma_probability
#         ])

#         predicted_idx = int(
#             asthma_probability >= 0.5
#         )

#     else:

#         probabilities = torch.softmax(
#             logits,
#             dim=-1
#         )[0].cpu().numpy()

#         predicted_idx = int(
#             np.argmax(probabilities)
#         )

#     return predicted_idx, probabilities


# # ============================================================
# # MAIN UI
# # ============================================================

# st.markdown("---")

# uploaded_file = st.file_uploader(
#     "Upload a respiratory audio recording (.wav)",
#     type=["wav"]
# )

# if uploaded_file is None:

#     st.info(
#         "Upload a .wav file to run the selected classifier."
#     )

#     st.markdown(
#         """
#         **Preprocessing used**

#         - Sampling rate → 22,050 Hz
#         - Duration → 6 seconds
#         - STFT → `n_fft=2048`, `hop_length=512`
#         - Log-magnitude dB → clipped at -80 dB
#         - Delta + delta-delta
#         - 3-channel spectrogram
#         - Resize → 224 × 224
#         - SigLIP-2 image processor
#         """
#     )

# else:

#     try:

#         # ----------------------------------------------------
#         # Load model and processor
#         # ----------------------------------------------------

#         processor = load_processor()

#         with st.spinner("Loading model..."):
#             model = load_model(
#                 model_path,
#                 num_labels
#             )

#         # ----------------------------------------------------
#         # Read WAV
#         # ----------------------------------------------------

#         audio_bytes = uploaded_file.read()

#         import io

#         y, sr = librosa.load(
#             io.BytesIO(audio_bytes),
#             sr=None
#         )

#         duration_original = len(y) / sr

#         # ----------------------------------------------------
#         # STFT preprocessing
#         # ----------------------------------------------------

#         with st.spinner("Processing audio..."):

#             stft_img = audio_to_stft_image(
#                 y,
#                 sr
#             )

#         # ----------------------------------------------------
#         # Prediction
#         # ----------------------------------------------------

#         with st.spinner("Running inference..."):

#             predicted_idx, probabilities = predict(
#                 model,
#                 processor,
#                 stft_img,
#                 binary=(num_labels == 1)
#             )

#         predicted_class = class_names[predicted_idx]

#         # ----------------------------------------------------
#         # RESULTS
#         # ----------------------------------------------------

#         st.success("Analysis complete")

#         col1, col2 = st.columns([1, 1])

#         with col1:

#             st.subheader("Prediction")

#             st.metric(
#                 "Predicted condition",
#                 predicted_class
#             )

#             if num_labels == 1:

#                 st.metric(
#                     "Asthma probability",
#                     f"{probabilities[1] * 100:.2f}%"
#                 )

#             else:

#                 confidence = probabilities[predicted_idx]

#                 st.metric(
#                     "Model confidence",
#                     f"{confidence * 100:.2f}%"
#                 )

#         with col2:

#             st.subheader("Recording")

#             st.write(
#                 f"**Original sampling rate:** {sr:,} Hz"
#             )

#             st.write(
#                 f"**Original duration:** {duration_original:.2f} s"
#             )

#             st.write(
#                 "**Model input:** 6.00 s / 22,050 Hz"
#             )

#         # ----------------------------------------------------
#         # PROBABILITIES
#         # ----------------------------------------------------

#         st.markdown("---")
#         st.subheader("Class probabilities")

#         prob_cols = st.columns(len(class_names))

#         for i, (name, prob) in enumerate(
#             zip(class_names, probabilities)
#         ):

#             with prob_cols[i]:

#                 st.metric(
#                     name.capitalize(),
#                     f"{prob * 100:.2f}%"
#                 )

#         # ----------------------------------------------------
#         # PROBABILITY BAR CHART
#         # Uses Matplotlib instead of st.bar_chart/Altair/Pandas.
#         # This avoids NumPy/Pandas/Altair binary compatibility
#         # problems in existing Anaconda environments.
#         # ----------------------------------------------------

#         st.markdown("---")
#         st.subheader("Probability distribution")

#         fig_prob, ax_prob = plt.subplots(figsize=(9, 4.5))

#         labels = [name.capitalize() for name in class_names]
#         values = probabilities * 100.0

#         bars = ax_prob.barh(labels, values)

#         ax_prob.set_xlabel("Probability (%)")
#         ax_prob.set_xlim(0, 100)
#         ax_prob.set_title("Model Class Probabilities")
#         ax_prob.grid(axis="x", alpha=0.25)

#         # Put the highest-probability class at the top.
#         ax_prob.invert_yaxis()

#         for bar, value in zip(bars, values):
#             ax_prob.text(
#                 min(value + 1.0, 96.0),
#                 bar.get_y() + bar.get_height() / 2,
#                 f"{value:.2f}%",
#                 va="center"
#             )

#         plt.tight_layout()

#         st.pyplot(
#             fig_prob,
#             clear_figure=True
#         )

#         # ----------------------------------------------------
#         # SPECTROGRAM
#         # ----------------------------------------------------

#         st.markdown("---")
#         st.subheader("Processed spectrogram")

#         fig = show_spectrogram(stft_img)

#         st.pyplot(
#             fig,
#             clear_figure=True
#         )

#         # ----------------------------------------------------
#         # AUDIO PLAYER
#         # ----------------------------------------------------

#         st.markdown("---")
#         st.subheader("Uploaded audio")

#         st.audio(
#             audio_bytes,
#             format="audio/wav"
#         )

#     except Exception as e:

#         st.error(
#             "An error occurred while loading the model, "
#             "processing the audio, or running inference."
#         )

#         st.exception(e)

import os
os.environ["USE_TF"] = "0"

import streamlit as st
import torch
import torch.nn as nn
import librosa
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, SiglipForImageClassification
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

BINARY_MODEL_PATH = "siglip2_asthma_healthy_final.pt"
MULTICLASS_MODEL_PATH = "siglip2_asthma_multiclass_final.pt"

MODEL_ID = "google/siglip2-base-patch16-224"

IMG_SIZE = 224
TARGET_SR = 22050
FIXED_DURATION = 6.0
N_FFT = 2048
HOP_LENGTH = 512
DB_FLOOR = -80.0

BINARY_CLASSES = ["Healthy", "Asthma"]

# IMPORTANT:
# This follows LabelEncoder().fit(categories) from the multiclass notebook.
# Alphabetical order is therefore:
MULTICLASS_CLASSES = [
    "Bronchial",
    "asthma",
    "copd",
    "healthy",
    "pneumonia",
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Respiratory Sound Classification",
    page_icon="🫁",
    layout="wide",
)

st.title("🫁 Respiratory Sound Classification")
st.caption("SigLIP-2 + STFT representation")

st.sidebar.header("Configuration")

task = st.sidebar.radio(
    "Select classification task",
    [
        "Binary — Asthma vs Healthy",
        "Multiclass — Lung Disease",
    ],
)

if task.startswith("Binary"):
    model_path = BINARY_MODEL_PATH
    class_names = BINARY_CLASSES
    num_labels = 1
else:
    model_path = MULTICLASS_MODEL_PATH
    class_names = MULTICLASS_CLASSES
    num_labels = 5

st.sidebar.markdown("### Selected model")
st.sidebar.code(
    "siglip2_asthma_healthy_final.pt"
    if task.startswith("Binary")
    else "siglip2_asthma_multiclass_final.pt"
)

st.sidebar.markdown(f"**Device:** `{DEVICE}`")

# The checkpoint contains both model.* and backbone_module.* entries.
# The loader intentionally uses only model.* because that is the
# SigLIP-2 model state used by the training wrapper's forward pass.


# ============================================================
# PREPROCESSING — EXACTLY MATCHING THE NOTEBOOK
# ============================================================

def fix_length(y, sr, duration=FIXED_DURATION):
    target_len = int(sr * duration)

    if len(y) >= target_len:
        start = (len(y) - target_len) // 2
        return y[start:start + target_len]

    pad = target_len - len(y)
    return np.pad(
        y,
        (pad // 2, pad - pad // 2),
        mode="constant"
    )


def audio_to_stft_image(y, sr, img_size=IMG_SIZE):
    """
    Same preprocessing used in both notebooks.

    WAV
      -> resample to 22050 Hz
      -> center crop / zero pad to 6 seconds
      -> STFT
      -> log magnitude dB
      -> [-80, 0] clipping and normalization
      -> delta
      -> delta-delta
      -> 3-channel image
      -> resize 224x224
      -> float32 [0,1]
    """

    if sr != TARGET_SR:
        y = librosa.resample(
            y,
            orig_sr=sr,
            target_sr=TARGET_SR
        )
        sr = TARGET_SR

    y = fix_length(y, sr)

    stft = librosa.stft(
        y,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )

    log_mag = librosa.amplitude_to_db(
        np.abs(stft),
        ref=np.max
    )

    log_mag_n = (
        np.clip(log_mag, DB_FLOOR, 0.0) / (-DB_FLOOR)
    ) + 1.0

    delta = librosa.feature.delta(log_mag)
    delta2 = librosa.feature.delta(
        log_mag,
        order=2
    )

    d_clip = 6.0

    delta_n = (
        np.clip(delta, -d_clip, d_clip) + d_clip
    ) / (2 * d_clip)

    delta2_n = (
        np.clip(delta2, -d_clip, d_clip) + d_clip
    ) / (2 * d_clip)

    img = np.stack(
        [log_mag_n, delta_n, delta2_n],
        axis=-1
    ).astype(np.float32)

    img_pil = Image.fromarray(
        (img * 255).astype(np.uint8)
    ).resize(
        (img_size, img_size),
        Image.BILINEAR
    )

    return np.array(img_pil).astype(np.float32) / 255.0


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource
def load_processor():
    return AutoImageProcessor.from_pretrained(MODEL_ID)


def extract_trained_state_dict(checkpoint):
    """
    The checkpoints were saved from the training wrappers as:

        {
            "model_state_dict": ...,
            ...
        }

    The state_dict contains two groups:

        model.*
        backbone_module.*

    The `model.*` group is the actual SigLIP-2 model used for
    forward inference. `backbone_module.*` is a duplicate
    reference to the vision backbone stored by the training
    wrapper and must NOT be loaded into the standalone
    SiglipForImageClassification object.

    Selecting only `model.*` removes the previous
    "208 unexpected keys" warning.
    """

    if not isinstance(checkpoint, dict):
        raise ValueError(
            "Expected a checkpoint dictionary."
        )

    if "model_state_dict" not in checkpoint:
        raise ValueError(
            "Checkpoint does not contain 'model_state_dict'."
        )

    full_state_dict = checkpoint["model_state_dict"]

    trained_state_dict = {}

    for key, value in full_state_dict.items():

        if key.startswith("model."):
            # Convert:
            # model.vision_model....
            #       ↓
            # vision_model....
            new_key = key[len("model."):]
            trained_state_dict[new_key] = value

    if not trained_state_dict:
        raise ValueError(
            "No 'model.*' parameters were found in the checkpoint."
        )

    return trained_state_dict


@st.cache_resource
def load_model(model_path, num_labels):
    """
    Load the exact SigLIP-2 architecture used by the notebooks.

    The saved checkpoints contain:
        model_state_dict
        model_choice
        model_id
        preprocessing metadata
        class_names

    Only the `model.*` portion of model_state_dict is loaded.
    The duplicate `backbone_module.*` entries are intentionally
    ignored because they belong to the training wrapper.
    """

    checkpoint = torch.load(
        model_path,
        map_location="cpu",
        weights_only=False
    )

    if not isinstance(checkpoint, dict):
        raise ValueError(
            "Unexpected checkpoint format. "
            "Expected the dictionary produced by the training notebook."
        )

    # Use the model identifier stored in the checkpoint.
    checkpoint_model_id = checkpoint.get(
        "model_id",
        MODEL_ID
    )

    checkpoint_model_choice = checkpoint.get(
        "model_choice",
        "siglip2"
    )

    if checkpoint_model_choice.lower() != "siglip2":
        raise ValueError(
            f"This Streamlit application expects a SigLIP-2 checkpoint, "
            f"but the checkpoint reports model_choice='{checkpoint_model_choice}'."
        )

    model = SiglipForImageClassification.from_pretrained(
        checkpoint_model_id,
        num_labels=num_labels,
        ignore_mismatched_sizes=False,
        attn_implementation="eager",
    )

    state_dict = extract_trained_state_dict(checkpoint)

    # Strict loading is intentional here. If this fails, the
    # checkpoint architecture does not match the inference model
    # and the application should NOT silently continue.
    missing, unexpected = model.load_state_dict(
        state_dict,
        strict=False
    )

    if missing or unexpected:
        raise RuntimeError(
            "Checkpoint/model architecture mismatch.\n\n"
            f"Missing keys: {len(missing)}\n"
            f"Unexpected keys: {len(unexpected)}\n\n"
            "The checkpoint was not loaded safely."
        )

    # Validate the classifier output dimension.
    classifier_weight = state_dict.get(
        "classifier.weight"
    )

    if classifier_weight is None:
        raise RuntimeError(
            "classifier.weight was not found in the checkpoint."
        )

    checkpoint_num_labels = classifier_weight.shape[0]

    if checkpoint_num_labels != num_labels:
        raise RuntimeError(
            f"Checkpoint has {checkpoint_num_labels} output units, "
            f"but the selected task expects {num_labels}."
        )

    model.to(DEVICE)
    model.eval()

    return model


# ============================================================
# SPECTROGRAM DISPLAY
# ============================================================

def show_spectrogram(img):
    fig, ax = plt.subplots(figsize=(10, 4))

    ax.imshow(
        img[..., 0],
        origin="lower",
        aspect="auto",
        cmap="magma"
    )

    ax.set_title("Log-Magnitude STFT")
    ax.set_xlabel("Time")
    ax.set_ylabel("Frequency")

    plt.tight_layout()

    return fig


# ============================================================
# INFERENCE
# ============================================================

def predict(model, processor, img, binary=False):

    img_uint8 = (img * 255).astype(np.uint8)

    inputs = processor(
        images=img_uint8,
        return_tensors="pt"
    )

    pixel_values = inputs["pixel_values"].to(DEVICE)

    with torch.no_grad():

        outputs = model(
            pixel_values=pixel_values
        )

        logits = outputs.logits

    if binary:

        # Notebook:
        # sigmoid(logit) >= 0.5 -> Asthma (1)
        asthma_probability = torch.sigmoid(
            logits.squeeze(-1)
        ).item()

        probabilities = np.array([
            1.0 - asthma_probability,
            asthma_probability
        ])

        predicted_idx = int(
            asthma_probability >= 0.5
        )

    else:

        probabilities = torch.softmax(
            logits,
            dim=-1
        )[0].cpu().numpy()

        predicted_idx = int(
            np.argmax(probabilities)
        )

    return predicted_idx, probabilities


# ============================================================
# MAIN UI
# ============================================================

st.markdown("---")

uploaded_file = st.file_uploader(
    "Upload a respiratory audio recording (.wav)",
    type=["wav"]
)

if uploaded_file is None:

    st.info(
        "Upload a .wav file to run the selected classifier."
    )

    st.markdown(
        """
        **Preprocessing used**

        - Sampling rate → 22,050 Hz
        - Duration → 6 seconds
        - STFT → `n_fft=2048`, `hop_length=512`
        - Log-magnitude dB → clipped at -80 dB
        - Delta + delta-delta
        - 3-channel spectrogram
        - Resize → 224 × 224
        - SigLIP-2 image processor
        """
    )

else:

    try:

        # ----------------------------------------------------
        # Load model and processor
        # ----------------------------------------------------

        processor = load_processor()

        with st.spinner("Loading model..."):
            model = load_model(
                model_path,
                num_labels
            )

        # ----------------------------------------------------
        # Read WAV
        # ----------------------------------------------------

        audio_bytes = uploaded_file.read()

        import io

        y, sr = librosa.load(
            io.BytesIO(audio_bytes),
            sr=None
        )

        duration_original = len(y) / sr

        # ----------------------------------------------------
        # STFT preprocessing
        # ----------------------------------------------------

        with st.spinner("Processing audio..."):

            stft_img = audio_to_stft_image(
                y,
                sr
            )

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        with st.spinner("Running inference..."):

            predicted_idx, probabilities = predict(
                model,
                processor,
                stft_img,
                binary=(num_labels == 1)
            )

        predicted_class = class_names[predicted_idx]

        # ----------------------------------------------------
        # RESULTS
        # ----------------------------------------------------

        st.success("Analysis complete")

        col1, col2 = st.columns([1, 1])

        with col1:

            st.subheader("Prediction")

            st.metric(
                "Predicted condition",
                predicted_class
            )

            if num_labels == 1:

                st.metric(
                    "Asthma probability",
                    f"{probabilities[1] * 100:.2f}%"
                )

            else:

                confidence = probabilities[predicted_idx]

                st.metric(
                    "Model confidence",
                    f"{confidence * 100:.2f}%"
                )

        with col2:

            st.subheader("Recording")

            st.write(
                f"**Original sampling rate:** {sr:,} Hz"
            )

            st.write(
                f"**Original duration:** {duration_original:.2f} s"
            )

            st.write(
                "**Model input:** 6.00 s / 22,050 Hz"
            )

        # ----------------------------------------------------
        # PROBABILITIES
        # ----------------------------------------------------

        st.markdown("---")
        st.subheader("Class probabilities")

        prob_cols = st.columns(len(class_names))

        for i, (name, prob) in enumerate(
            zip(class_names, probabilities)
        ):

            with prob_cols[i]:

                st.metric(
                    name.capitalize(),
                    f"{prob * 100:.2f}%"
                )

        # ----------------------------------------------------
        # PROBABILITY BAR CHART
        # Uses Matplotlib instead of st.bar_chart/Altair/Pandas.
        # This avoids NumPy/Pandas/Altair binary compatibility
        # problems in existing Anaconda environments.
        # ----------------------------------------------------

        st.markdown("---")
        st.subheader("Probability distribution")

        fig_prob, ax_prob = plt.subplots(figsize=(9, 4.5))

        labels = [name.capitalize() for name in class_names]
        values = probabilities * 100.0

        bars = ax_prob.barh(labels, values)

        ax_prob.set_xlabel("Probability (%)")
        ax_prob.set_xlim(0, 100)
        ax_prob.set_title("Model Class Probabilities")
        ax_prob.grid(axis="x", alpha=0.25)

        # Put the highest-probability class at the top.
        ax_prob.invert_yaxis()

        for bar, value in zip(bars, values):
            ax_prob.text(
                min(value + 1.0, 96.0),
                bar.get_y() + bar.get_height() / 2,
                f"{value:.2f}%",
                va="center"
            )

        plt.tight_layout()

        st.pyplot(
            fig_prob,
            clear_figure=True
        )

        # ----------------------------------------------------
        # SPECTROGRAM
        # ----------------------------------------------------

        st.markdown("---")
        st.subheader("Processed spectrogram")

        fig = show_spectrogram(stft_img)

        st.pyplot(
            fig,
            clear_figure=True
        )

        # ----------------------------------------------------
        # AUDIO PLAYER
        # ----------------------------------------------------

        st.markdown("---")
        st.subheader("Uploaded audio")

        st.audio(
            audio_bytes,
            format="audio/wav"
        )

    except Exception as e:

        st.error(
            "An error occurred while loading the model, "
            "processing the audio, or running inference."
        )

        st.exception(e)
