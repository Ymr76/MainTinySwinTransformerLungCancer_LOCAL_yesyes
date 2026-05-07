import streamlit as st
import torch
import torchvision.transforms as transforms
import timm
import cv2
import numpy as np
from PIL import Image
import math
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import os

# --- Page Config ---
st.set_page_config(page_title="Lung Cancer Detector", page_icon="🫁", layout="wide")
st.title("Swin Transformer Lung Cancer Detector")
st.markdown("Upload a lung scan to classify it and visualize the region of interest.")

# --- Constants & Classes ---
NUM_CLASSES = 4
# ImageFolder (used during training) only reads directories — it skipped the .txt file.
# So the actual class indices from training are:
# 0: 'Bengin cases'
# 1: 'Malignant cases'
# 2: 'Normal cases'
# 3: (4th output neuron, never targeted during training)
class_names = ['Bengin cases', 'Malignant cases', 'Normal cases', 'Unknown (Not in dataset)']

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ApplyCLAHE(object):
    """
    CLAHE (Contrast Limited Adaptive Histogram Equalization)
    """
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, img):
        img_np = np.array(img)
        if len(img_np.shape) == 2:  # Grayscale
            clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
            img_clahe = clahe.apply(img_np)
            img_out = Image.fromarray(img_clahe)
        else:  # RGB
            lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
            l_clahe = clahe.apply(l)
            lab_clahe = cv2.merge((l_clahe, a, b))
            img_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)
            img_out = Image.fromarray(img_clahe)
        return img_out

def swin_reshape_transform(tensor):
    """
    Reshape transform for Swin Transformer for Grad-CAM
    """
    batch_size, seq_len, channels = tensor.shape
    size = int(math.sqrt(seq_len))
    reshaped_tensor = tensor.reshape(batch_size, size, size, channels)
    return reshaped_tensor.permute(0, 3, 1, 2)

# --- Model Loading ---
@st.cache_resource
def load_model():
    model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=False, num_classes=NUM_CLASSES)
    model_path = os.path.join("deployed_models", "swin_lung_cancer_state.pt")
    
    if not os.path.exists(model_path):
        st.error(f"Model file not found at {model_path}. Please ensure it was exported.")
        return None
        
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    return model

model = load_model()

# --- Preprocessing Pipeline ---
data_transforms = transforms.Compose([
    ApplyCLAHE(clip_limit=2.0, tile_grid_size=(8, 8)),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- UI Layout ---
uploaded_file = st.file_uploader("Choose a lung scan image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model is not None:
    try:
        # Open Image
        original_img = Image.open(uploaded_file).convert('RGB')
        
        # Two columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            st.image(original_img, use_container_width=True)
        
        # Predict & Visualize
        with st.spinner("Analyzing image..."):
            img_tensor = data_transforms(original_img)
            img_input = img_tensor.unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                output = model(img_input)
                probs = torch.nn.functional.softmax(output, dim=1)
                confidence, preds = torch.max(probs, 1)
                
            predicted_class_idx = preds.item()
            predicted_class = class_names[predicted_class_idx]
            conf_score = confidence.item() * 100
            
            # --- Grad-CAM Logic ---
            target_layers = [model.layers[-1].blocks[-1].norm2]
            cam = GradCAM(model=model, target_layers=target_layers, reshape_transform=swin_reshape_transform)
            
            targets = [ClassifierOutputTarget(predicted_class_idx)]
            grayscale_cam = cam(input_tensor=img_input, targets=targets)[0, :]
            
            # Un-normalize for display
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            rgb_img = img_tensor.cpu().numpy().transpose((1, 2, 0))
            rgb_img = std * rgb_img + mean
            rgb_img = np.clip(rgb_img, 0, 1)
            
            # Red Box logic (Background Masking)
            gray_img_for_mask = cv2.cvtColor((rgb_img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            _, lung_mask = cv2.threshold(gray_img_for_mask, 15, 1, cv2.THRESH_BINARY)
            
            masked_cam = grayscale_cam * lung_mask
            max_cam_val = np.max(masked_cam)
            if max_cam_val > 0:
                masked_cam = masked_cam / max_cam_val
                
            visualization = show_cam_on_image(rgb_img, masked_cam, use_rgb=True)
            visualization = np.ascontiguousarray(visualization)
            
            # Draw Red Box if it's Malignant or if we just want to show ROI
            heatmap_thresh = (masked_cam > 0.7).astype(np.uint8) * 255
            contours, _ = cv2.findContours(heatmap_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)
                cv2.rectangle(visualization, (x, y), (x+w, y+h), (255, 0, 0), 3) # Red box
                
        with col2:
            st.subheader(f"Prediction: {predicted_class}")
            st.write(f"**Confidence:** {conf_score:.2f}%")
            if 'malignant' in predicted_class.lower():
                st.error("⚠️ Alert: Potential Malignancy Detected.")
            elif 'bengin' in predicted_class.lower() or 'benign' in predicted_class.lower():
                st.warning("🟡 Benign case detected.")
            else:
                st.success("✅ Normal case detected.")

            # --- Probability Distribution for all 3 real classes ---
            st.markdown("**Class Probabilities:**")
            probs_np = probs.cpu().numpy()[0]
            display_classes = ['Bengin cases', 'Malignant cases', 'Normal cases']
            for i, cls in enumerate(display_classes):
                st.progress(float(probs_np[i]), text=f"{cls}: {probs_np[i]*100:.1f}%")

            st.caption(
                "⚠️ **Model Bias Warning:** Trained on imbalanced data "
                "(Malignant ~1190 imgs vs Normal ~55 imgs) for only 5 epochs. "
                "The model tends to over-predict Malignant. Consult a radiologist."
            )

            st.image(visualization, caption="Heatmap & ROI (Red Box)", use_container_width=True)

    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
