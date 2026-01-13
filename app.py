import streamlit as st
import cv2
import numpy as np
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import io

# --- CONFIGURATION ---
REAL_MARKER_SIZE = 45.0  # mm
ARUCO_DICT_TYPE = cv2.aruco.DICT_4X4_50

# --- CONFIG: Increase Streamlit Upload Limit ---
# (This is handled in .streamlit/config.toml, but we handle memory here)

def detect_aruco_and_get_ratios(image_array):
    img = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_TYPE)
    parameters = cv2.aruco.DetectorParameters()
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX 
    
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    corners, ids, rejected = detector.detectMarkers(gray)

    ratio_x = None
    ratio_y = None
    
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(img, corners, ids)
        c = corners[0][0]
        
        # Sub-Pixel Refinement
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
        gray_float = np.float32(gray)
        c_refined = cv2.cornerSubPix(gray, c, (5, 5), (-1, -1), criteria)
        c = c_refined

        width_top = np.linalg.norm(c[0] - c[1])
        width_bot = np.linalg.norm(c[3] - c[2])
        avg_width_px = (width_top + width_bot) / 2.0
        
        height_left = np.linalg.norm(c[0] - c[3])
        height_right = np.linalg.norm(c[1] - c[2])
        avg_height_px = (height_left + height_right) / 2.0
        
        ratio_x = avg_width_px / REAL_MARKER_SIZE
        ratio_y = avg_height_px / REAL_MARKER_SIZE
        
        label = f"Precision Mode | X:{ratio_x:.1f} Y:{ratio_y:.1f}"
        cv2.putText(img, label, (int(c[0][0]), int(c[0][1] - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB), ratio_x, ratio_y

st.title("📏 Unified Health Measurement Tool")
st.markdown(f"""
**Instructions:**
1. Marker size: **{int(REAL_MARKER_SIZE)}mm**.
2. **Place marker ON TOP of the object** (same height).
3. Take photo directly from above.
""")

uploaded_file = st.file_uploader("Upload Photo", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    # --- MEMORY FIX: RESIZE HUGE IMAGES ---
    # Phones take 4000px+ images. This crashes the free cloud tier.
    # We resize to max 2500px, which is plenty for sub-millimeter accuracy.
    max_dimension = 2500
    if image.width > max_dimension or image.height > max_dimension:
        image.thumbnail((max_dimension, max_dimension))
    
    try:
        from PIL import ImageOps
        image = ImageOps.exif_transpose(image)
    except:
        pass
        
    img_array = np.array(image)
    
    processed_img, px_per_mm_x, px_per_mm_y = detect_aruco_and_get_ratios(img_array)

    if px_per_mm_x and px_per_mm_y:
        st.success(f"Active! Scales: X={px_per_mm_x:.2f}, Y={px_per_mm_y:.2f}")
        
        bg_image = Image.fromarray(processed_img)
        
        max_width = 800
        original_width, original_height = bg_image.size
        
        if original_width > max_width:
            display_scale = max_width / original_width
            canvas_width = max_width
            canvas_height = int(original_height * display_scale)
        else:
            display_scale = 1.0
            canvas_width = original_width
            canvas_height = original_height

        st.write("Draw lines:")
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",
            stroke_width=2,
            stroke_color="#FF0000",
            background_image=bg_image,
            update_streamlit=True,
            height=canvas_height,
            width=canvas_width,
            drawing_mode="line",
            key="canvas",
        )

        if canvas_result.json_data is not None:
            objects = canvas_result.json_data["objects"]
            
            if len(objects) > 0:
                
                # --- VISUAL OVERLAY ---
                drawing_array = canvas_result.image_data.astype('uint8')
                drawing_pil = Image.fromarray(drawing_array)
                drawing_pil_resized = drawing_pil.resize((original_width, original_height), resample=Image.NEAREST)
                
                final_pil = bg_image.copy()
                final_pil.paste(drawing_pil_resized, (0, 0), drawing_pil_resized)
                
                final_img = np.array(final_pil)
                final_img = cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR)

                # --- CALCULATION ---
                for obj in objects:
                    w = obj["width"] * obj["scaleX"]
                    h = obj["height"] * obj["scaleY"]
                    ang = np.deg2rad(obj["angle"])
                    
                    dx_screen_px = abs(w * np.cos(ang) - h * np.sin(ang))
                    dy_screen_px = abs(w * np.sin(ang) + h * np.cos(ang))
                    
                    dx_original_px = dx_screen_px / display_scale
                    dy_original_px = dy_screen_px / display_scale
                    
                    dx_mm = dx_original_px / px_per_mm_x
                    dy_mm = dy_original_px / px_per_mm_y
                    
                    real_length_mm = np.sqrt(dx_mm**2 + dy_mm**2)

                    # Labels
                    center_x = obj["left"] + (obj["width"] * obj["scaleX"] / 2)
                    center_y = obj["top"] + (obj["height"] * obj["scaleY"] / 2)
                    text_x = int(center_x / display_scale)
                    text_y = int(center_y / display_scale)

                    label = f"{real_length_mm:.1f} mm"
                    text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
                    text_w, text_h = text_size
                    cv2.rectangle(final_img, (text_x, text_y - 40), (text_x + text_w + 10, text_y - 5), (255, 255, 255), -1)
                    cv2.putText(final_img, label, (text_x + 5, text_y - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

                # Latest Metric
                last_obj = objects[-1]
                lw = last_obj["width"] * last_obj["scaleX"]
                lh = last_obj["height"] * last_obj["scaleY"]
                la = np.deg2rad(last_obj["angle"])
                ldx = abs(lw * np.cos(la) - lh * np.sin(la)) / display_scale
                ldy = abs(lw * np.sin(la) + lh * np.cos(la)) / display_scale
                last_mm = np.sqrt((ldx/px_per_mm_x)**2 + (ldy/px_per_mm_y)**2)
                st.metric(label="Latest Measurement", value=f"{last_mm:.1f} mm")

                # Download
                final_img = cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB)
                final_pil_save = Image.fromarray(final_img)
                buf = io.BytesIO()
                final_pil_save.save(buf, format="JPEG", quality=95)
                byte_im = buf.getvalue()

                st.download_button(
                    label="💾 Download Record",
                    data=byte_im,
                    file_name="measurement_record.jpg",
                    mime="image/jpeg"
                )
    else:
        st.error("⚠️ ArUco marker not detected.")
        st.image(processed_img, use_container_width=True)