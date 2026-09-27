import cv2
import numpy as np
import streamlit as st

st.set_page_config(page_title="Passport Photo Alignment", page_icon="📸", layout="wide")

# Standard portrait frame
OUT_W, OUT_H = 413, 531

FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
EYE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)

def detect(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    faces = FACE_CASCADE.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
    )
    if len(faces) == 0:
        return None, None

    face = max(faces, key=lambda r: r[2] * r[3])
    x, y, w, h = face

    # Eye search only inside upper face region
    roi = gray[y:y + int(h * 0.60), x:x + w]
    eyes = EYE_CASCADE.detectMultiScale(
        roi, scaleFactor=1.08, minNeighbors=6, minSize=(18, 18)
    )

    candidates = []
    for ex, ey, ew, eh in eyes:
        cx = x + ex + ew / 2
        cy = y + ey + eh / 2
        candidates.append((cx, cy, ew * eh))

    candidates.sort(key=lambda a: a[2], reverse=True)

    # Choose one eye from each half of the face
    left = [e for e in candidates if e[0] < x + w / 2]
    right = [e for e in candidates if e[0] >= x + w / 2]

    if left and right:
        l = max(left, key=lambda e: e[2])
        r = max(right, key=lambda e: e[2])
        eyes = np.float32([[l[0], l[1]], [r[0], r[1]]])
    else:
        eyes = None

    return face, eyes


def align(img, face, eyes):
    x, y, w, h = face

    # Center of detected face
    face_center = np.array(
        [x + w / 2.0, y + h / 2.0],
        dtype=np.float32
    )

    # -----------------------------------------
    # 1. ROTATION
    # -----------------------------------------
    angle = 0.0

    if eyes is not None:
        left_eye, right_eye = eyes

        # Angle of line connecting the eyes
        angle = -np.degrees(
            np.arctan2(
                right_eye[1] - left_eye[1],
                right_eye[0] - left_eye[0]
            )
        )

    # -----------------------------------------
    # 2. SCALING
    # -----------------------------------------
    # We want the face to occupy about 52% of
    # the passport-photo width.
    target_face_width = OUT_W * 0.52

    scale = target_face_width / float(w)

    # Prevent extreme scaling
    scale = np.clip(scale, 0.6, 2.5)

    # -----------------------------------------
    # 3. TARGET FACE POSITION
    # -----------------------------------------
    # Horizontal center
    target_x = OUT_W / 2

    # Slightly above the vertical center because
    # passport photos normally show more shoulders
    # below the face.
    target_y = OUT_H * 0.43

    target_center = np.array(
        [target_x, target_y],
        dtype=np.float32
    )

    # -----------------------------------------
    # 4. ROTATE + SCALE AROUND FACE CENTER
    # -----------------------------------------
    M = cv2.getRotationMatrix2D(
        (float(face_center[0]), float(face_center[1])),
        float(angle),
        float(scale)
    )

    # -----------------------------------------
    # 5. TRANSLATION
    # -----------------------------------------
    # After rotation/scaling, the face center
    # must move to target_center.
    M[0, 2] += target_center[0] - face_center[0]
    M[1, 2] += target_center[1] - face_center[1]

    # -----------------------------------------
    # 6. CREATE FINAL IMAGE
    # -----------------------------------------
    result = cv2.warpAffine(
        img,
        M,
        (OUT_W, OUT_H),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE
    )

    return result, "Face-centered: rotation + scaling + translation"


def mark(img, face, eyes):
    out = img.copy()
    x, y, w, h = face
    cv2.rectangle(out, (x, y), (x+w, y+h), (0, 255, 0), 2)

    if eyes is not None:
        for ex, ey in eyes.astype(int):
            cv2.circle(out, (ex, ey), 7, (255, 0, 0), -1)

    return out


st.title("📸 Passport Photo Alignment System")
st.write("Automatically aligns a portrait using translation, rotation and scaling.")

uploaded = st.file_uploader("Upload a clear front-facing photograph",
                            type=["jpg", "jpeg", "png"])

if uploaded:
    data = np.frombuffer(uploaded.getvalue(), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)

    if img is None:
        st.error("Could not read the image.")
        st.stop()

    face, eyes = detect(img)

    if face is None:
        st.error("No face detected. Try a clearer, front-facing photograph.")
        st.stop()

    result, method = align(img, face, eyes)

    if result is None:
        st.error("Alignment failed.")
        st.stop()

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Original / Detection")
        st.image(cv2.cvtColor(mark(img, face, eyes), cv2.COLOR_BGR2RGB),
                 use_container_width=True)

    with c2:
        st.subheader("Aligned Passport Photo")
        st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB),
                 use_container_width=True)

    st.success(method)

    ok, encoded = cv2.imencode(".jpg", result, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if ok:
        st.download_button(
            "⬇️ Download aligned photo",
            encoded.tobytes(),
            "aligned_passport_photo.jpg",
            "image/jpeg"
        )

st.divider()
st.markdown("""
### Algorithm
**Input → Face Detection → Eye Detection → Rotation → Scaling → Translation → Aligned Output**

The system uses OpenCV Haar Cascades for face/eye detection and an affine transformation for geometric alignment.

**Important:** This is a computer-vision mini-project prototype, not an official passport-photo compliance checker.
""")
