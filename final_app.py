# -*- coding: utf-8 -*-
import streamlit as st
import numpy as np
import cv2
from PIL import Image
import pandas as pd
import time
from datetime import datetime
import io

st.set_page_config(
    page_title="CytoAnalyzer Pro",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS with animations
st.markdown("""
<style>
    @keyframes fadeIn { from {opacity: 0; transform: translateY(20px);} to {opacity: 1; transform: translateY(0);} }
    @keyframes pulse { 0% {transform: scale(1);} 50% {transform: scale(1.05);} 100% {transform: scale(1);} }
    @keyframes spin { 0% {transform: rotate(0deg);} 100% {transform: rotate(360deg);} }
    
    .main-header {
        animation: fadeIn 1s ease-in;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem; border-radius: 15px; color: white; text-align: center; margin-bottom: 2rem;
    }
    .developer-card {
        animation: fadeIn 1.5s ease-out;
        background: #f8f9fa; padding: 1.5rem; border-radius: 10px;
        border-left: 5px solid #667eea; margin: 1rem 0;
    }
    .analyze-btn { animation: pulse 2s infinite; }
    .loading-spinner {
        display: inline-block; width: 20px; height: 20px;
        border: 3px solid #f3f3f3; border-top: 3px solid #667eea;
        border-radius: 50%; animation: spin 1s linear infinite;
    }
    .success-animation {
        animation: fadeIn 0.5s ease-in; background: #d4edda;
        border: 1px solid #c3e6cb; color: #155724;
        padding: 1rem; border-radius: 5px; margin: 1rem 0;
    }
    .stButton > button {
        transition: all 0.3s ease; border-radius: 25px;
    }
    .stButton > button:hover {
        transform: scale(1.05); box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

def classify_object(contour, gray_roi):
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
    rect = cv2.minAreaRect(contour)
    (w, h) = rect[1]
    aspect_ratio = max(w, h) / max(min(w, h), 1e-6)
    
    if aspect_ratio > 3.0 and circularity < 0.6:
        return "Chromosome", 0.85
    elif circularity > 0.7 and area > 1000:
        return "Nucleus", 0.80
    elif circularity < 0.5:
        return "DNA_Fragment", 0.75
    elif circularity > 0.6 and area < 800:
        return "RNA_Granule", 0.70
    else:
        return "Unknown", 0.50

def analyze_image(img_rgb, min_area=200, progress_bar=None):
    if progress_bar:
        progress_bar.progress(20)
    
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    if binary.mean() > 127:
        binary = 255 - binary
    
    if progress_bar:
        progress_bar.progress(50)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if progress_bar:
        progress_bar.progress(80)
    
    results = []
    annotated = img_rgb.copy()
    colors = {"Chromosome": (255, 0, 0), "Nucleus": (0, 255, 0), "DNA_Fragment": (0, 0, 255), 
              "RNA_Granule": (255, 255, 0), "Unknown": (128, 128, 128)}
    
    for i, contour in enumerate(contours, 1):
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect).astype(int)
        (cx, cy), (w, h), angle = rect
        major_px = max(w, h)
        
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.fillPoly(mask, [contour], 255)
        roi = cv2.bitwise_and(gray, mask)
        
        obj_class, confidence = classify_object(contour, roi)
        color = colors.get(obj_class, (128, 128, 128))
        
        cv2.drawContours(annotated, [box], 0, color, 2)
        cv2.putText(annotated, str(i), (int(cx), int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        results.append({
            "ID": i, "Class": obj_class, "Confidence": round(confidence, 2),
            "Size_px": int(major_px), "Area_px": int(area)
        })
    
    if progress_bar:
        progress_bar.progress(100)
    
    return results, annotated

# Header
st.markdown("""
<div class="main-header">
    <h1>🧬 CytoAnalyzer Pro</h1>
    <h3>Professional Microscopy Analysis Platform</h3>
    <p>AI-Powered • Real-Time • Mobile Ready</p>
</div>
""", unsafe_allow_html=True)

# Developer info
st.markdown("""
<div class="developer-card">
    <h3>👨💻 Development Team</h3>
    <p><strong>Developer:</strong> Mujahid | BS Bioinformatics</p>
    <p><strong>University:</strong> Hazara University Mansehra</p>
    <p><strong>Supervisor:</strong> Assistant Professor Sajid ul Ghafoor</p>
    <p><strong>Version:</strong> Pro v2.0 - Cloud Ready</p>
</div>
""", unsafe_allow_html=True)

# Settings
with st.expander("⚙️ Settings", expanded=False):
    col1, col2 = st.columns(2)
    with col1:
        min_area = st.slider("Min Object Size", 100, 1000, 200)
    with col2:
        known_um = st.number_input("Scale (μm)", value=10.0)

# Image input
st.subheader("📸 Image Input")
tab1, tab2 = st.tabs(["📁 Upload", "📷 Camera"])

with tab1:
    uploaded_file = st.file_uploader("Choose image", type=['png', 'jpg', 'jpeg', 'tiff'])

with tab2:
    camera_image = st.camera_input("Take photo")

# Process image
image = None
if camera_image:
    image = Image.open(camera_image).convert('RGB')
    st.markdown('<div class="success-animation">✅ Photo captured!</div>', unsafe_allow_html=True)
elif uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.markdown('<div class="success-animation">✅ Image uploaded!</div>', unsafe_allow_html=True)

if image:
    img_array = np.array(image)
    st.image(image, caption="Source Image", use_column_width=True)
    
    st.markdown('<div class="analyze-btn">', unsafe_allow_html=True)
    if st.button("🔬 Analyze Image", type="primary"):
        st.markdown('<div class="loading-spinner"></div> Analyzing...', unsafe_allow_html=True)
        
        progress_bar = st.progress(0)
        time.sleep(0.5)
        
        start_time = time.time()
        results, annotated = analyze_image(img_array, min_area, progress_bar)
        processing_time = time.time() - start_time
        
        progress_bar.empty()
        st.markdown('<div class="success-animation">🎉 Analysis Complete!</div>', unsafe_allow_html=True)
        
        st.image(annotated, caption="Analysis Results", use_column_width=True)
        
        if results:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Objects", len(results))
            with col2:
                st.metric("Time", f"{processing_time:.1f}s")
            with col3:
                avg_conf = sum(r['Confidence'] for r in results) / len(results)
                st.metric("Confidence", f"{avg_conf:.2f}")
            
            df = pd.DataFrame(results)
            st.dataframe(df, use_container_width=True)
            
            col_exp1, col_exp2 = st.columns(2)
            with col_exp1:
                csv = df.to_csv(index=False)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                st.download_button("📊 CSV", csv, f"analysis_{timestamp}.csv", "text/csv")
            
            with col_exp2:
                img_buffer = io.BytesIO()
                Image.fromarray(annotated).save(img_buffer, format='PNG')
                st.download_button("🖼️ PNG", img_buffer.getvalue(), f"result_{timestamp}.png", "image/png")
        else:
            st.warning("No objects detected. Try adjusting settings.")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Installation guide
st.markdown("---")
st.subheader("📱 Mobile Installation")
st.info("""
**Install on Android/iPhone:**
1. Open this link in Chrome/Safari
2. Tap menu (⋮) → "Add to Home screen"
3. App installs like native mobile app
4. Use phone camera for microscope images!
""")

# Footer
st.markdown("""
<div style="text-align: center; margin-top: 2rem; animation: fadeIn 3s ease-in;">
    <h4>🧬 CytoAnalyzer Pro v2.0</h4>
    <p>Developed by <strong>Mujahid</strong> | BS Bioinformatics</p>
    <p>Hazara University Mansehra</p>
    <p>Supervised by <strong>Assistant Professor Sajid ul Ghafoor</strong></p>
</div>
""", unsafe_allow_html=True)