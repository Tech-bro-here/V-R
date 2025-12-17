import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import os
import time
from PIL import Image
import io
import datetime

try:
    from rembg import remove
except ImportError:
    st.error("Module 'rembg' not found. Install with `pip install rembg`")

if "capture_triggered" not in st.session_state:
    st.session_state.capture_triggered = False

# --- Constants ---
CATEGORIES = ["Dresses", "T-shirts", "Hats", "Glasses"]
FOLDER_MAP = {c: c.replace(" ", "").replace("-", "") for c in CATEGORIES}
for folder in FOLDER_MAP.values():
    os.makedirs(folder, exist_ok=True)
SNAPSHOT_DIR = "snapshots"
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh

# --- Utility functions for preprocessing ---
def bg_remove(image: Image.Image) -> Image.Image:
    buf = io.BytesIO()
    image.convert("RGBA").save(buf, format="PNG")
    byte_img = buf.getvalue()
    output_bytes = remove(byte_img)
    return Image.open(io.BytesIO(output_bytes))

def crop_transparent(img: Image.Image, threshold=2) -> Image.Image:
    arr = np.array(img)
    if arr.shape[2] == 4:
        alpha = arr[:, :, 3]
        rows = np.any(alpha > threshold, axis=1)
        cols = np.any(alpha > threshold, axis=0)
        if rows.any() and cols.any():
            ymin, ymax = np.where(rows)[0][[0, -1]]
            xmin, xmax = np.where(cols)[0][[0, -1]]
            arr_cropped = arr[ymin:ymax+1, xmin:xmax+1]
            return Image.fromarray(arr_cropped)
    return img

def save_preprocessed_image(final_img, uploaded_file, category):
    save_folder = FOLDER_MAP[category]
    base_name = os.path.splitext(uploaded_file.name)[0]
    save_path = os.path.join(save_folder, base_name + ".png")
    final_img.save(save_path, format="PNG")
    return save_path

def resize_to_fit(overlay, target_w, target_h):
    oh, ow = overlay.shape[:2]
    if ow == 0 or oh == 0:
        return overlay, 0, 0
    scale_w = target_w / ow
    scale_h = target_h / oh
    scale = min(scale_w, scale_h)
    new_w, new_h = max(1, int(ow * scale)), max(1, int(oh * scale))
    overlay_resized = cv2.resize(overlay, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return overlay_resized, new_w, new_h

def overlay_image(frame, overlay_rgba):
    if overlay_rgba is None or frame is None:
        return frame
    alpha = overlay_rgba[..., 3:4] / 255.0
    overlay_rgb = overlay_rgba[..., :3]
    inv_alpha = 1.0 - alpha
    h, w = overlay_rgba.shape[:2]
    fh, fw = frame.shape[:2]
    if h > fh or w > fw:
        h, w = min(h, fh), min(w, fw)
        overlay_rgba = overlay_rgba[:h, :w]
        frame = frame[:h, :w]
    frame[:] = (alpha * overlay_rgb + inv_alpha * frame).astype(np.uint8)
    return frame

def warp_overlay_perspective(frame, overlay_rgba, dst_pts):
    fh, fw = frame.shape[:2]
    oh, ow = overlay_rgba.shape[:2]
    src = np.array([[0, 0], [ow - 1, 0], [ow - 1, oh - 1], [0, oh - 1]], dtype=np.float32)
    dst = np.array(dst_pts, dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(overlay_rgba, M, (fw, fh), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    return overlay_image(frame, warped)

def warp_overlay_affine(frame, overlay_rgba, dst_pts_3):
    fh, fw = frame.shape[:2]
    oh, ow = overlay_rgba.shape[:2]
    src = np.array([[0, oh // 2], [ow - 1, oh // 2], [ow // 2, 0]], dtype=np.float32)
    dst = np.array(dst_pts_3, dtype=np.float32)
    M = cv2.getAffineTransform(src, dst)
    warped = cv2.warpAffine(overlay_rgba, M, (fw, fh), borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))
    return overlay_image(frame, warped)

def ensure_alpha(img: np.ndarray) -> np.ndarray:
    if img.shape[2] == 3:
        alpha_channel = np.full(img.shape[:2], 255, dtype=img.dtype)
        return cv2.merge((*cv2.split(img), alpha_channel))
    return img

# --- Virtual Try-On Function ---
def virtual_try_on(category: str, product_path: str):
    # Fixed parameters (no sliders)
    y_offset = -45
    roi_scale_top = 1.0
    roi_scale_bottom = 1.07
    roi_height_factor = 1.08
    top_width_padding = 120
    bottom_width_padding = 120

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Webcam not accessible.")
        return

    img_raw = cv2.imread(product_path, cv2.IMREAD_UNCHANGED)
    if img_raw is None:
        st.error("Failed to load image. Make sure the PNG is present!")
        return
    img_raw = ensure_alpha(img_raw)

    show_landmarks = st.sidebar.checkbox("Show Landmarks & ROI", value=True)

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose, \
         mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True) as face_mesh:

        placeholder = st.empty()
        col1, col2 = st.columns(2)
        snap_btn = col1.button("Capture Snapshot")
        stop_btn = col2.button("Stop Camera")
        st.session_state.capture_triggered = False

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            fh, fw, _ = frame.shape
            display_frame = frame.copy()

            # === DRESSES / T-SHIRTS (UNCHANGED) ===
                        # === DRESSES / T-SHIRTS (UPDATED – T-shirt stops at hip) ===
                        # === DRESSES / T-SHIRTS (T-shirt: hip | Dress: full length to bottom) ===
                        # === DRESSES / T-SHIRTS (FULLY CORRECTED: T-shirt to hip | Dress to knee) ===
                        # === DRESSES / T-SHIRTS (SAMPLE-STYLE OVERLAY + YOUR ROI BOX) ===
            if category in ["Dresses", "T-shirts"]:
                results = pose.process(frame_rgb)
                if results.pose_landmarks:
                    lms = results.pose_landmarks.landmark
                    ls = np.array([lms[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * fw,
                                   lms[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * fh])
                    rs = np.array([lms[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * fw,
                                   lms[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * fh])
                    lh = np.array([lms[mp_pose.PoseLandmark.LEFT_HIP.value].x * fw,
                                   lms[mp_pose.PoseLandmark.LEFT_HIP.value].y * fh])
                    rh = np.array([lms[mp_pose.PoseLandmark.RIGHT_HIP.value].x * fw,
                                   lms[mp_pose.PoseLandmark.RIGHT_HIP.value].y * fh])

                    # --- Your ROI Logic (Kept 100% same) ---
                    shoulder_vector = rs - ls
                    top_middle = (ls + rs) / 2 + np.array([0, y_offset])

                    shoulder_norm = np.linalg.norm(shoulder_vector)
                    delta_shoulder = (shoulder_vector / (shoulder_norm + 1e-8)) * (shoulder_norm * roi_scale_top + top_width_padding) / 2
                    tl_adj = top_middle - delta_shoulder
                    tr_adj = top_middle + delta_shoulder

                    hip_middle = (lh + rh) / 2
                    hip_vector = rh - lh
                    hip_norm = np.linalg.norm(hip_vector)
                    hip_dir = hip_vector / (hip_norm + 1e-8)

                    if category == "T-shirts":
                        hip_dist = (hip_norm * roi_scale_bottom + bottom_width_padding) / 2
                        bl = hip_middle - hip_dir * hip_dist
                        br = hip_middle + hip_dir * hip_dist
                        bottom_center = hip_middle
                    else:
                        lk = np.array([lms[mp_pose.PoseLandmark.LEFT_KNEE.value].x * fw,
                                       lms[mp_pose.PoseLandmark.LEFT_KNEE.value].y * fh])
                        rk = np.array([lms[mp_pose.PoseLandmark.RIGHT_KNEE.value].x * fw,
                                       lms[mp_pose.PoseLandmark.RIGHT_KNEE.value].y * fh])
                        knee_middle = (lk + rk) / 2
                        bottom_center = knee_middle

                        dress_bottom_width_factor = 1.1
                        bottom_width = hip_norm * roi_scale_bottom * dress_bottom_width_factor + bottom_width_padding
                        bottom_half_width = bottom_width / 2
                        perp_dir = np.array([-hip_dir[1], hip_dir[0]])
                        bl = bottom_center - perp_dir * bottom_half_width
                        br = bottom_center + perp_dir * bottom_half_width

                    # --- SAMPLE PROGRAM OVERLAY LOGIC (No Warp!) ---
                    center_x = int(top_middle[0])
                    shoulders_y = int(top_middle[1])
                    bottom_y = int(max(bl[1], br[1]))

                    dress_width = int(shoulder_norm * roi_scale_top + top_width_padding)
                    dress_height = max(1, int(bottom_y - shoulders_y))

                    if dress_width > 0 and dress_height > 0:
                        # Resize dress to fit ROI width × height
                        resized_dress = cv2.resize(img_raw, (dress_width, dress_height), interpolation=cv2.INTER_AREA)
                        resized_dress = ensure_alpha(resized_dress)

                        # Position: center on shoulders, offset up by 1/3 of height
                        dress_x = center_x - dress_width // 2
                        dress_y = int(shoulders_y)

                        # Clip to frame
                        x1 = max(0, dress_x)
                        y1 = max(0, dress_y)
                        x2 = min(fw, dress_x + dress_width)
                        y2 = min(fh, dress_y + dress_height)

                        if x1 < x2 and y1 < y2:
                            crop_w = x2 - x1
                            crop_h = y2 - y1
                            dress_crop = resized_dress[
                                (y1 - dress_y):(y1 - dress_y + crop_h),
                                (x1 - dress_x):(x1 - dress_x + crop_w)
                            ]
                            frame_roi = display_frame[y1:y2, x1:x2]

                            alpha = dress_crop[:, :, 3] / 255.0
                            overlay_rgb = dress_crop[:, :, :3]

                            blended = (alpha[..., None] * overlay_rgb + (1 - alpha[..., None]) * frame_roi).astype(np.uint8)
                            display_frame[y1:y2, x1:x2] = blended

                    # --- Draw your red ROI box (for debug) ---
                    if show_landmarks:
                        for pt in [ls, rs, lh, rh]:
                            cv2.circle(display_frame, tuple(np.int32(pt)), 10, (0, 255, 0), -1)
                        if category == "Dresses":
                            lk_pt = np.array([lms[mp_pose.PoseLandmark.LEFT_KNEE.value].x * fw,
                                              lms[mp_pose.PoseLandmark.LEFT_KNEE.value].y * fh])
                            rk_pt = np.array([lms[mp_pose.PoseLandmark.RIGHT_KNEE.value].x * fw,
                                              lms[mp_pose.PoseLandmark.RIGHT_KNEE.value].y * fh])
                            cv2.circle(display_frame, tuple(np.int32(lk_pt)), 8, (255, 0, 255), -1)
                            cv2.circle(display_frame, tuple(np.int32(rk_pt)), 8, (255, 0, 255), -1)

                        # Draw ROI box
                        pts = np.array([tl_adj, tr_adj, br, bl], dtype=np.int32)
                        cv2.polylines(display_frame, [pts], isClosed=True, color=(255, 0, 0), thickness=3)

            # === HATS (FIXED: ROBUST LANDMARKS & ROI) ===
                        # === HATS (UPDATED – matches the robust logic from the sample) ===
                        # === HATS (UPDATED – hat sits higher on the head) ===
            elif category == "Hats":
                face_results = face_mesh.process(frame_rgb)
                if face_results.multi_face_landmarks:
                    lms = face_results.multi_face_landmarks[0].landmark

                    # ---- Landmark selection (same as sample) ----
                    left  = lms[234]      # left side of head (near ear)
                    right = lms[454]      # right side of head (near ear)
                    top   = lms[10]       # top of forehead

                    x1, y1 = int(left.x  * fw), int(left.y  * fh)
                    x2, y2 = int(right.x * fw), int(right.y * fh)
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    dx, dy = x2 - x1, y2 - y1
                    angle  = np.degrees(np.arctan2(dy, dx))

                    # ---- Scaling (same default as sample) ----
                    eye_dist = ((dx ** 2 + dy ** 2) ** 0.5)
                    scale_factor = 1.45
                    target_width = int(eye_dist * scale_factor)

                    overlay_resized = cv2.resize(img_raw,
                                                 (target_width,
                                                  int(img_raw.shape[0] * target_width / img_raw.shape[1])),
                                                 interpolation=cv2.INTER_AREA)

                    # ---- Rotate to match head tilt ----
                    M = cv2.getRotationMatrix2D((overlay_resized.shape[1] // 2,
                                                 overlay_resized.shape[0] // 2),
                                                angle, 1.0)
                    rotated = cv2.warpAffine(overlay_resized, M,
                                             (overlay_resized.shape[1], overlay_resized.shape[0]),
                                             flags=cv2.INTER_AREA,
                                             borderMode=cv2.BORDER_CONSTANT,
                                             borderValue=(0, 0, 0, 0))

                    # ---- Position – lift the hat a little higher (≈ 12 % of its height) ----
                    x_offset = cx - rotated.shape[1] // 2
                    y_offset = int(top.y * fh) - int(rotated.shape[0] * 0.88)   # changed from 0.75 → 0.88

                    # Clamp to frame borders
                    x_offset = max(0, min(x_offset, fw - rotated.shape[1]))
                    y_offset = max(0, min(y_offset, fh - rotated.shape[0]))

                    # ---- Alpha‑blend overlay ----
                    for i in range(rotated.shape[0]):
                        for j in range(rotated.shape[1]):
                            if (0 <= y_offset + i < fh) and (0 <= x_offset + j < fw):
                                alpha = rotated[i, j, 3] / 255.0
                                if alpha > 0:
                                    display_frame[y_offset + i, x_offset + j] = (
                                        (1 - alpha) * display_frame[y_offset + i, x_offset + j] +
                                        alpha * rotated[i, j, :3]
                                    ).astype(np.uint8)

                    # ---- Debug landmarks (optional) ----
                    if show_landmarks:
                        for pt in [(x1, y1), (x2, y2), (int(top.x * fw), int(top.y * fh))]:
                            cv2.circle(display_frame, pt, 7, (0, 255, 0), -1)
                        cv2.line(display_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

            # === GLASSES (UNCHANGED) ===
                        # === GLASSES (UPDATED – identical to the sample) ===
            elif category == "Glasses":
                face_results = face_mesh.process(frame_rgb)
                if face_results.multi_face_landmarks:
                    lms = face_results.multi_face_landmarks[0].landmark

                    # ---- Landmark selection (same as sample) ----
                    left  = lms[33]       # left eye outer corner
                    right = lms[263]      # right eye outer corner
                    top   = lms[168]      # nose bridge (center)

                    x1, y1 = int(left.x  * fw), int(left.y  * fh)
                    x2, y2 = int(right.x * fw), int(right.y * fh)
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    dx, dy = x2 - x1, y2 - y1
                    angle  = np.degrees(np.arctan2(dy, dx))
                    angle = -angle

                    # ---- Scaling (default from sample) ----
                    eye_dist = ((dx ** 2 + dy ** 2) ** 0.5)
                    scale_factor = 1.45
                    target_width = int(eye_dist * scale_factor)

                    overlay_resized = cv2.resize(img_raw,
                                                 (target_width,
                                                  int(img_raw.shape[0] * target_width / img_raw.shape[1])),
                                                 interpolation=cv2.INTER_AREA)

                    # ---- Rotate ----
                    M = cv2.getRotationMatrix2D((overlay_resized.shape[1] // 2,
                                                 overlay_resized.shape[0] // 2),
                                                angle, 1.0)
                    rotated = cv2.warpAffine(overlay_resized, M,
                                             (overlay_resized.shape[1], overlay_resized.shape[0]),
                                             flags=cv2.INTER_AREA,
                                             borderMode=cv2.BORDER_CONSTANT,
                                             borderValue=(0, 0, 0, 0))

                    # ---- Position (center on eyes) ----
                    x_offset = cx - rotated.shape[1] // 2
                    y_offset = cy - rotated.shape[0] // 2

                    x_offset = max(0, min(x_offset, fw - rotated.shape[1]))
                    y_offset = max(0, min(y_offset, fh - rotated.shape[0]))

                    # ---- Alpha‑blend overlay ----
                    for i in range(rotated.shape[0]):
                        for j in range(rotated.shape[1]):
                            if (0 <= y_offset + i < fh) and (0 <= x_offset + j < fw):
                                alpha = rotated[i, j, 3] / 255.0
                                if alpha > 0:
                                    display_frame[y_offset + i, x_offset + j] = (
                                        (1 - alpha) * display_frame[y_offset + i, x_offset + j] +
                                        alpha * rotated[i, j, :3]
                                    ).astype(np.uint8)

                    # ---- Debug landmarks (optional) ----
                    if show_landmarks:
                        for pt in [(x1, y1), (x2, y2), (int(top.x * fw), int(top.y * fh))]:
                            cv2.circle(display_frame, pt, 7, (0, 255, 0), -1)
                        cv2.line(display_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

            placeholder.image(display_frame, channels="BGR", use_container_width=True)

            if snap_btn and not st.session_state.capture_triggered:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                filename = os.path.join(SNAPSHOT_DIR, f"snapshot_{timestamp}.png")
                cv2.imwrite(filename, display_frame)
                st.toast(f"Saved: `{os.path.basename(filename)}`")
                st.session_state.capture_triggered = True

            if stop_btn:
                break

            time.sleep(0.03)

        cap.release()

# --- Streamlit App ---
st.set_page_config(page_title="Virtual Clothing Try-On", layout="wide")
st.title("Virtual Clothing, Hat & Glasses Try-On System")
mode = st.sidebar.radio("Select Mode", ["Image Preparation", "Try-On"])

if mode == "Image Preparation":
    st.header("Upload and Preprocess Product Images")
    category = st.selectbox("Select Clothing Category", CATEGORIES)
    uploaded_file = st.file_uploader("Choose Clothing Image (PNG/JPG)", type=["png", "jpg", "jpeg"])
    threshold = st.slider("Cropping threshold (lower = more aggressive)", min_value=1, max_value=50, value=2, step=1)

    if uploaded_file is not None:
        pil_img = Image.open(uploaded_file).convert("RGBA")
        st.image(pil_img, caption="Step 1: Uploaded Image", use_container_width=True)

        if st.button("Remove Background"):
            with st.spinner("Removing background..."):
                prepped_img = bg_remove(pil_img)
            st.session_state.prepped_img = prepped_img
            st.image(prepped_img, caption="Step 2: Background Removed", use_container_width=True)

        if st.button("Crop Transparent"):
            img_to_crop = st.session_state.get("prepped_img", pil_img)
            cropped_img = crop_transparent(img_to_crop, threshold=threshold)
            st.session_state.cropped_img = cropped_img
            st.image(cropped_img, caption="Step 3: Cropped Image Preview", use_container_width=True)

        if st.session_state.get("cropped_img", None) is not None:
            st.image(st.session_state.cropped_img, caption="Final Preview", use_container_width=True)
            if st.button("Save Final Image"):
                saved_path = save_preprocessed_image(st.session_state.cropped_img, uploaded_file, category)
                st.success(f"Image preprocessed and saved to `{saved_path}`")

elif mode == "Try-On":
    st.header("Select Product for Virtual Try-On")
    category = st.selectbox("Select Product Category", CATEGORIES)
    folder = FOLDER_MAP[category]
    products = [f for f in os.listdir(folder) if f.lower().endswith('.png')]

    if len(products) == 0:
        st.warning(f"No preprocessed images found in folder `{folder}`. Please use **Image Preparation** mode first.")
    else:
        cols = st.columns(3)
        for i, prod in enumerate(products):
            path = os.path.join(folder, prod)
            with cols[i % 3]:
                st.image(path, use_container_width=True)
                if st.button(f"Try On {prod}", key=prod):
                    st.session_state.selected = path
                    st.session_state.category = category

        if "selected" in st.session_state:
            st.header("Live Preview Area")
            virtual_try_on(st.session_state.category, st.session_state.selected)
