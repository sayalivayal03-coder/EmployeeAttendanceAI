import os
import cv2

# ==========================================
# INPUT DATASET
# ==========================================

INPUT_DIR = r"C:\Users\Sayali\Downloads\archive (3)\Image_Train"

# ==========================================
# OUTPUT FACE DATASET
# ==========================================

OUTPUT_DIR = r"C:\Users\Sayali\Downloads\face_dataset"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# FACE DETECTOR
# ==========================================

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

# ==========================================
# CHECK INPUT FOLDER
# ==========================================

if not os.path.exists(INPUT_DIR):
    print("❌ INPUT folder not found:")
    print(INPUT_DIR)
    exit()

print("====================================")
print("FACE DATASET PREPARATION STARTED")
print("====================================")

# ==========================================
# PROCESS EACH EMPLOYEE
# ==========================================

employees = os.listdir(INPUT_DIR)

print("Employees found:", len(employees))
print()

for employee in employees:

    employee_path = os.path.join(
        INPUT_DIR,
        employee
    )

    if not os.path.isdir(employee_path):
        continue

    output_employee = os.path.join(
        OUTPUT_DIR,
        employee
    )

    os.makedirs(
        output_employee,
        exist_ok=True
    )

    count = 0

    print("Processing:", employee)

    # ======================================
    # PROCESS IMAGES
    # ======================================

    for filename in os.listdir(employee_path):

        image_path = os.path.join(
            employee_path,
            filename
        )

        image = cv2.imread(image_path)

        if image is None:
            continue

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(80, 80)
        )

        if len(faces) == 0:

            print(
                "  ❌ Face not found:",
                filename
            )

            continue

        # ==================================
        # GET LARGEST FACE
        # ==================================

        x, y, w, h = max(
            faces,
            key=lambda face: face[2] * face[3]
        )

        face = image[
            y:y+h,
            x:x+w
        ]

        # ==================================
        # SAVE FACE
        # ==================================

        output_path = os.path.join(
            output_employee,
            f"{count}.jpg"
        )

        cv2.imwrite(
            output_path,
            face
        )

        count += 1

    print(
        "  ✅",
        employee,
        "->",
        count,
        "faces saved"
    )

    print()

# ==========================================
# DONE
# ==========================================

print("====================================")
print("DONE!")
print("====================================")
print(
    "Face dataset created at:"
)
print(OUTPUT_DIR)