import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import av
import json
import os
import time
import requests

from datetime import datetime, time as dt_time
from collections import deque
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Employee Attendance AI",
    page_icon="👤",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("👤 Employee Face Recognition Based Attendance System")

st.write(
    "Camera → Face Recognition → Login / Logout → Google Sheet"
)

st.divider()


# ============================================================
# BASE DIRECTORY
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


# ============================================================
# MODEL PATH
# ============================================================

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "employee_recognition_model.keras"
)

CLASS_NAMES_PATH = os.path.join(
    BASE_DIR,
    "model",
    "class_names.json"
)


# ============================================================
# GOOGLE APPS SCRIPT URL
# ============================================================

GOOGLE_SCRIPT_URL = (
    "https://script.google.com/macros/s/"
    "AKfycby8eE2p2vQkUoImX1Lb80oDkPKv5LaNntHnWPGTiNvQDCYYqaeryEBegMe4AvniE6yd"
    "/exec"
)


# ============================================================
# SETTINGS
# ============================================================

IMAGE_SIZE = (160, 160)

CONFIDENCE_THRESHOLD = 80

CHECK_EVERY_N_FRAMES = 5

SMOOTHING_FRAMES = 5

COOLDOWN_SECONDS = 60

# Face crop around detected face
FACE_MARGIN = 0.20

# Minimum detected face size
MIN_FACE_SIZE = (80, 80)


# ============================================================
# CHECK MODEL
# ============================================================

if not os.path.isfile(MODEL_PATH):

    st.error(
        "❌ Model file not found:\n\n"
        + MODEL_PATH
    )

    st.stop()


# ============================================================
# CHECK CLASS NAMES
# ============================================================

if not os.path.isfile(CLASS_NAMES_PATH):

    st.error(
        "❌ class_names.json not found:\n\n"
        + CLASS_NAMES_PATH
    )

    st.stop()


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_employee_model():

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    return model


try:

    model = load_employee_model()

except Exception as e:

    st.error(
        "❌ Model could not be loaded.\n\n"
        + str(e)
    )

    st.stop()


# ============================================================
# LOAD CLASS NAMES
# ============================================================

try:

    with open(
        CLASS_NAMES_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        class_names = json.load(f)

except Exception as e:

    st.error(
        "❌ class_names.json could not be read.\n\n"
        + str(e)
    )

    st.stop()


# ============================================================
# MODEL INFORMATION
# ============================================================

st.success(
    f"✅ Model loaded successfully | "
    f"Employees: {len(class_names)}"
)


# ============================================================
# VERIFY MODEL OUTPUT
# ============================================================

try:

    model_output_classes = int(
        model.output_shape[-1]
    )

except Exception:

    model_output_classes = None


if model_output_classes is not None:

    if model_output_classes != len(class_names):

        st.error(
            "❌ MODEL / CLASS NAMES MISMATCH\n\n"
            f"Model output classes: {model_output_classes}\n"
            f"class_names count: {len(class_names)}"
        )

        st.stop()


# ============================================================
# VERIFY 17 CLASSES
# ============================================================

if len(class_names) != 17:

    st.warning(
        f"⚠️ Expected 17 classes but found "
        f"{len(class_names)} classes."
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ System Information")

    st.write(
        f"**Employees:** {len(class_names)}"
    )

    st.write(
        "**Input Size:** 160 × 160"
    )

    st.write(
        f"**Confidence:** {CONFIDENCE_THRESHOLD}%"
    )

    st.write(
        f"**Face Margin:** {int(FACE_MARGIN * 100)}%"
    )

    st.write(
        "**Login:** 06:30 AM onwards"
    )

    st.write(
        "**Logout:** 06:30 PM onwards"
    )

    st.divider()

    st.subheader("Employee Classes")

    for i, employee in enumerate(class_names):

        st.write(
            f"{i + 1}. {employee}"
        )


# ============================================================
# FACE DETECTOR
# ============================================================

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades
    + "haarcascade_frontalface_default.xml"
)


if face_detector.empty():

    st.error(
        "❌ Haar Cascade face detector could not be loaded."
    )

    st.stop()


# ============================================================
# ATTENDANCE ACTION
# ============================================================

def get_attendance_action():

    current_time = datetime.now().time()

    login_time = dt_time(
        6,
        30
    )

    logout_time = dt_time(
        18,
        30
    )

    # Before 6:30 AM
    if current_time < login_time:

        return "BEFORE_LOGIN"

    # 6:30 AM to before 6:30 PM
    elif current_time < logout_time:

        return "LOGIN"

    # 6:30 PM onwards
    else:

        return "LOGOUT"


# ============================================================
# FACE CROP WITH MARGIN
# ============================================================

def crop_face_with_margin(
    image,
    x,
    y,
    w,
    h
):

    image_height, image_width = image.shape[:2]

    margin_x = int(
        w * FACE_MARGIN
    )

    margin_y = int(
        h * FACE_MARGIN
    )

    x1 = max(
        0,
        x - margin_x
    )

    y1 = max(
        0,
        y - margin_y
    )

    x2 = min(
        image_width,
        x + w + margin_x
    )

    y2 = min(
        image_height,
        y + h + margin_y
    )

    face_crop = image[
        y1:y2,
        x1:x2
    ]

    return face_crop, x1, y1, x2, y2


# ============================================================
# IMAGE PREPARATION
# ============================================================

def prepare_face(face):

    if face is None:

        return None

    if face.size == 0:

        return None

    # Camera frame is BGR
    # Model input is RGB

    face_rgb = cv2.cvtColor(
        face,
        cv2.COLOR_BGR2RGB
    )

    # Resize to model input size

    face_rgb = cv2.resize(
        face_rgb,
        IMAGE_SIZE,
        interpolation=cv2.INTER_AREA
    )

    # Float32

    face_rgb = face_rgb.astype(
        np.float32
    )

    # Add batch dimension

    face_rgb = np.expand_dims(
        face_rgb,
        axis=0
    )

    # IMPORTANT:
    #
    # Do NOT divide by 255 here.
    #
    # If the trained Keras model contains:
    #
    # Rescaling(1./127.5, offset=-1)
    #
    # the model performs normalization internally.

    return face_rgb


# ============================================================
# MODEL PREDICTION
# ============================================================

def predict_employee(face):

    input_image = prepare_face(
        face
    )

    if input_image is None:

        return (
            "Unknown",
            0.0,
            -1
        )

    try:

        prediction = model.predict(
            input_image,
            verbose=0
        )[0]

    except Exception as e:

        print(
            "Prediction error:",
            e
        )

        return (
            "Unknown",
            0.0,
            -1
        )

    predicted_index = int(
        np.argmax(prediction)
    )

    confidence = float(
        prediction[predicted_index] * 100
    )

    # Check index

    if (
        predicted_index < 0
        or predicted_index >= len(class_names)
    ):

        return (
            "Unknown",
            confidence,
            predicted_index
        )

    employee_name = class_names[
        predicted_index
    ]

    return (
        employee_name,
        confidence,
        predicted_index
    )


# ============================================================
# SEND ATTENDANCE TO GOOGLE SHEET
# ============================================================

def send_attendance(
    employee_name,
    confidence,
    action
):

    now = datetime.now()

    data = {

        "date":
            now.strftime("%d-%m-%Y"),

        "time":
            now.strftime("%H:%M:%S"),

        "employee_name":
            employee_name,

        "confidence":
            round(
                confidence,
                2
            ),

        "action":
            action,

        "uniform":
            "NOT_CHECKED",

        "id_card":
            "NOT_CHECKED"
    }

    try:

        response = requests.post(
            GOOGLE_SCRIPT_URL,
            json=data,
            timeout=15
        )

        if response.status_code == 200:

            return (
                True,
                response.text
            )

        return (
            False,
            f"HTTP Error: {response.status_code}"
        )

    except Exception as e:

        return (
            False,
            str(e)
        )


# ============================================================
# VIDEO PROCESSOR
# ============================================================

class EmployeeRecognitionProcessor(
    VideoProcessorBase
):

    def __init__(self):

        self.frame_count = 0

        # Current result

        self.employee_name = "Scanning..."

        self.confidence = 0.0

        self.action = ""

        self.message = ""

        # Prediction history

        self.prediction_history = deque(
            maxlen=SMOOTHING_FRAMES
        )

        # Google Sheet cooldown

        self.last_submit_time = {}


    # ========================================================
    # CAMERA FRAME
    # ========================================================

    def recv(
        self,
        frame
    ):

        self.frame_count += 1

        # WebRTC → OpenCV

        img = frame.to_ndarray(
            format="bgr24"
        )


        # ====================================================
        # FACE DETECTION
        # ====================================================

        gray = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2GRAY
        )

        faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=MIN_FACE_SIZE
        )


        # ====================================================
        # NO FACE
        # ====================================================

        if len(faces) == 0:

            # IMPORTANT:
            # Clear old predictions.
            # Otherwise previous employee can
            # remain in prediction history.

            self.prediction_history.clear()

            self.employee_name = "No Face"

            self.confidence = 0.0

            self.action = ""

            self.message = (
                "Please show your face clearly"
            )

            cv2.putText(
                img,
                "FACE NOT DETECTED",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                2
            )

            return av.VideoFrame.from_ndarray(
                img,
                format="bgr24"
            )


        # ====================================================
        # SELECT LARGEST FACE
        # ====================================================

        x, y, w, h = max(
            faces,
            key=lambda box:
            box[2] * box[3]
        )


        # ====================================================
        # FACE CROP WITH MARGIN
        # ====================================================

        (
            face_crop,
            crop_x1,
            crop_y1,
            crop_x2,
            crop_y2
        ) = crop_face_with_margin(
            img,
            x,
            y,
            w,
            h
        )


        # ====================================================
        # CHECK FACE SIZE
        # ====================================================

        if face_crop is None:

            self.employee_name = "Unknown"

            self.confidence = 0.0

            self.action = "NOT VERIFIED"

            self.message = (
                "Invalid face crop"
            )

            return av.VideoFrame.from_ndarray(
                img,
                format="bgr24"
            )


        # ====================================================
        # DRAW FACE BOX
        # ====================================================

        cv2.rectangle(
            img,
            (crop_x1, crop_y1),
            (crop_x2, crop_y2),
            (0, 255, 0),
            3
        )


        # ====================================================
        # RUN MODEL
        # ====================================================

        if (
            self.frame_count
            % CHECK_EVERY_N_FRAMES
            == 0
        ):

            (
                predicted_name,
                predicted_confidence,
                predicted_index
            ) = predict_employee(
                face_crop
            )


            # =================================================
            # SAVE ONLY VALID PREDICTION
            # =================================================

            if (
                predicted_name != "Unknown"
                and predicted_index >= 0
            ):

                self.prediction_history.append(
                    (
                        predicted_name,
                        predicted_confidence,
                        predicted_index
                    )
                )


            # =================================================
            # STABLE PREDICTION
            # =================================================

            if len(
                self.prediction_history
            ) >= 3:

                names = [
                    p[0]
                    for p in
                    self.prediction_history
                ]


                # Most repeated name

                stable_name = max(
                    set(names),
                    key=names.count
                )


                # Predictions for stable employee

                stable_items = [
                    p
                    for p in
                    self.prediction_history
                    if p[0] == stable_name
                ]


                # Average confidence

                stable_confidence = float(
                    np.mean(
                        [
                            p[1]
                            for p in stable_items
                        ]
                    )
                )


                stable_index = (
                    stable_items[-1][2]
                )


                self.employee_name = (
                    stable_name
                )

                self.confidence = (
                    stable_confidence
                )


                # =============================================
                # HIGH CONFIDENCE
                # =============================================

                if (
                    stable_confidence
                    >= CONFIDENCE_THRESHOLD
                ):

                    action = (
                        get_attendance_action()
                    )

                    self.action = action


                    # =========================================
                    # BEFORE 6:30 AM
                    # =========================================

                    if action == "BEFORE_LOGIN":

                        self.message = (
                            "⏰ Login starts at 06:30 AM"
                        )


                    # =========================================
                    # LOGIN / LOGOUT
                    # =========================================

                    else:

                        current_time = (
                            time.time()
                        )

                        last_time = (
                            self.last_submit_time.get(
                                stable_name,
                                0
                            )
                        )


                        # =====================================
                        # COOLDOWN
                        # =====================================

                        if (
                            current_time
                            - last_time
                            >= COOLDOWN_SECONDS
                        ):

                            success, result = (
                                send_attendance(
                                    stable_name,
                                    stable_confidence,
                                    action
                                )
                            )


                            if success:

                                self.message = (
                                    f"✅ "
                                    f"{stable_name} "
                                    f"{action} recorded"
                                )

                                self.last_submit_time[
                                    stable_name
                                ] = current_time

                            else:

                                self.message = (
                                    "❌ Google Sheet error: "
                                    + str(result)
                                )


                # =============================================
                # LOW CONFIDENCE
                # =============================================

                else:

                    self.action = (
                        "NOT VERIFIED"
                    )

                    self.message = (
                        "❌ Low confidence - "
                        "look at camera clearly"
                    )


        # ====================================================
        # NAME DISPLAY
        # ====================================================

        if (
            self.employee_name
            not in [
                "Scanning...",
                "No Face",
                "Unknown"
            ]
        ):

            if (
                self.confidence
                >= CONFIDENCE_THRESHOLD
            ):

                text_color = (
                    0,
                    255,
                    0
                )

            else:

                text_color = (
                    0,
                    0,
                    255
                )


            label = (
                f"{self.employee_name} "
                f"{self.confidence:.1f}%"
            )


            # Black label background

            cv2.rectangle(
                img,
                (
                    crop_x1,
                    max(
                        0,
                        crop_y1 - 55
                    )
                ),
                (
                    crop_x2,
                    crop_y1
                ),
                (0, 0, 0),
                -1
            )


            cv2.putText(
                img,
                label,
                (
                    crop_x1 + 10,
                    max(
                        35,
                        crop_y1 - 18
                    )
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                text_color,
                2,
                cv2.LINE_AA
            )


        # ====================================================
        # ACTION DISPLAY
        # ====================================================

        if self.action:

            cv2.putText(
                img,
                f"Action: {self.action}",
                (30, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 255),
                2
            )


        # ====================================================
        # MESSAGE DISPLAY
        # ====================================================

        if self.message:

            cv2.putText(
                img,
                self.message,
                (30, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (255, 255, 255),
                2
            )


        # ====================================================
        # RETURN CAMERA
        # ====================================================

        return av.VideoFrame.from_ndarray(
            img,
            format="bgr24"
        )


# ============================================================
# CAMERA SECTION
# ============================================================

st.header(
    "📷 Employee Face Recognition"
)

st.info(
    "Camera ON करा आणि चेहरा camera समोर "
    "स्पष्ट ठेवा. Recognition 80% किंवा "
    "त्यापेक्षा जास्त झाल्यावर attendance process होईल."
)


# ============================================================
# START CAMERA
# ============================================================

webrtc_streamer(
    key="employee-attendance",

    video_processor_factory=(
        EmployeeRecognitionProcessor
    ),

    media_stream_constraints={
        "video": True,
        "audio": False
    },

    async_processing=True
)


# ============================================================
# ATTENDANCE RULES
# ============================================================

st.divider()

st.header(
    "🕒 Attendance Rules"
)

col1, col2, col3 = st.columns(3)


with col1:

    st.metric(
        "LOGIN",
        "06:30 AM+"
    )


with col2:

    st.metric(
        "LOGOUT",
        "06:30 PM+"
    )


with col3:

    st.metric(
        "Recognition",
        "80%+"
    )


st.caption(
    "Before 06:30 AM → Login record होणार नाही."
)


# ============================================================
# MODEL DEBUG INFORMATION
# ============================================================

with st.expander(
    "🔍 Model Information"
):

    st.write(
        "**Model Path:**"
    )

    st.code(
        os.path.abspath(
            MODEL_PATH
        )
    )


    st.write(
        "**Class Names:**"
    )

    st.json(
        class_names
    )


    st.write(
        "**Number of Classes:**"
    )

    st.write(
        len(class_names)
    )


    st.write(
        "**Model Input:**"
    )

    st.code(
        str(
            model.input_shape
        )
    )


    st.write(
        "**Model Output:**"
    )

    st.code(
        str(
            model.output_shape
        )
    )


    st.write(
        "**Face Margin:**"
    )

    st.write(
        f"{int(FACE_MARGIN * 100)}%"
    )


    st.write(
        "**Confidence Threshold:**"
    )

    st.write(
        f"{CONFIDENCE_THRESHOLD}%"
    )