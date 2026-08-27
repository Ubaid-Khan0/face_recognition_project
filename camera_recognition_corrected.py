import os
import time
import sqlite3
from datetime import datetime

import cv2
from deepface import DeepFace
from ultralytics import YOLO


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CAMERA_INDEX = 0

KNOWN_FACES_DIR = os.path.join(BASE_DIR, "known_faces")
DATABASE_FILE = os.path.join(BASE_DIR, "attendance.db")

# YOLO11 nano
YOLO_MODEL = os.path.join(BASE_DIR, "yolo11n.pt")

# Run face recognition once every second
RECOGNITION_INTERVAL = 1.0

# DeepFace model
FACE_MODEL = "Facenet512"

# Detector used by DeepFace
# opencv is fast and works well for this webcam pipeline.
FACE_DETECTOR = "opencv"

# Used as an additional safety threshold.
# DeepFace's verified result is also required.
FACE_DISTANCE_THRESHOLD = 0.40

# Prevent the same recognized student from triggering the database
# action repeatedly.
ATTENDANCE_ACTION_COOLDOWN_SECONDS = 60

# YOLO confidence for person detection
YOLO_CONFIDENCE = 0.45


# ============================================================
# DATABASE
# ============================================================

def create_database():
    connection = sqlite3.connect(DATABASE_FILE)
    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS attendance_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_name TEXT NOT NULL,
            attendance_date TEXT NOT NULL,
            check_in TEXT NOT NULL,
            check_out TEXT,
            created_at TEXT NOT NULL,
            legacy_attendance_id INTEGER UNIQUE
        )
    """)

    # Preserve records from an older attendance table if one exists.
    cursor.execute("""
        SELECT name FROM sqlite_master
        WHERE type = 'table' AND name = 'attendance'
    """)

    if cursor.fetchone():
        cursor.execute("""
            INSERT OR IGNORE INTO attendance_sessions
            (
                student_name,
                attendance_date,
                check_in,
                check_out,
                created_at,
                legacy_attendance_id
            )
            SELECT
                student_name,
                attendance_date,
                check_in,
                check_out,
                created_at,
                id
            FROM attendance
            WHERE check_in IS NOT NULL
        """)

    connection.commit()
    connection.close()


# ============================================================
# CHECK-IN
# ============================================================

def check_in(student_name):
    now = datetime.now()

    date_text = now.strftime("%Y-%m-%d")
    time_text = now.strftime("%H:%M:%S")

    connection = sqlite3.connect(DATABASE_FILE)
    cursor = connection.cursor()

    # Do not create another active check-in for the same student.
    cursor.execute("""
        SELECT id
        FROM attendance_sessions
        WHERE student_name = ?
          AND attendance_date = ?
          AND check_out IS NULL
        ORDER BY id DESC
        LIMIT 1
    """, (student_name, date_text))

    record = cursor.fetchone()

    if record:
        connection.close()
        return False, f"{student_name} is already checked in. Check out first."

    cursor.execute("""
        INSERT INTO attendance_sessions
        (
            student_name,
            attendance_date,
            check_in,
            check_out,
            created_at
        )
        VALUES (?, ?, ?, ?, ?)
    """, (
        student_name,
        date_text,
        time_text,
        None,
        now.isoformat()
    ))

    connection.commit()
    connection.close()

    return True, f"{student_name} CHECK-IN successful at {time_text}"


# ============================================================
# CHECK-OUT
# ============================================================

def check_out(student_name):
    now = datetime.now()

    date_text = now.strftime("%Y-%m-%d")
    time_text = now.strftime("%H:%M:%S")

    connection = sqlite3.connect(DATABASE_FILE)
    cursor = connection.cursor()

    cursor.execute("""
        SELECT id, check_in
        FROM attendance_sessions
        WHERE student_name = ?
          AND attendance_date = ?
          AND check_out IS NULL
        ORDER BY id DESC
        LIMIT 1
    """, (student_name, date_text))

    record = cursor.fetchone()

    if not record:
        connection.close()
        return False, f"{student_name} has no active check-in today."

    cursor.execute("""
        UPDATE attendance_sessions
        SET check_out = ?
        WHERE id = ?
    """, (time_text, record[0]))

    connection.commit()
    connection.close()

    return True, f"{student_name} CHECK-OUT successful at {time_text}"


# ============================================================
# GET TODAY'S ATTENDANCE
# ============================================================

def get_today_attendance():
    today = datetime.now().strftime("%Y-%m-%d")

    connection = sqlite3.connect(DATABASE_FILE)
    cursor = connection.cursor()

    cursor.execute("""
        SELECT
            student_name,
            attendance_date,
            check_in,
            check_out
        FROM attendance_sessions
        WHERE attendance_date = ?
        ORDER BY check_in
    """, (today,))

    records = cursor.fetchall()
    connection.close()

    return records


# ============================================================
# LOAD KNOWN STUDENTS
# ============================================================

def load_known_students():
    students = {}

    if not os.path.exists(KNOWN_FACES_DIR):
        os.makedirs(KNOWN_FACES_DIR)

        print()
        print("=" * 70)
        print("known_faces folder was created.")
        print("Put student images inside this folder.")
        print("Example:")
        print("known_faces/ubaid_khan.jpg")
        print("=" * 70)
        print()

        return students

    for filename in os.listdir(KNOWN_FACES_DIR):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            image_path = os.path.join(KNOWN_FACES_DIR, filename)

            # Check that OpenCV can actually read the image.
            image = cv2.imread(image_path)

            if image is None:
                print(f"WARNING: Could not read image: {image_path}")
                continue

            student_name = os.path.splitext(filename)[0]
            student_name = student_name.replace("_", " ")

            students[student_name] = image_path

    return students


# ============================================================
# FACE RECOGNITION
# ============================================================

def recognize_student(face_crop, known_students):
    """
    Compare a YOLO-detected person/face crop against each registered
    student's reference image.

    Returns:
        (student_name, distance) if a match is found
        (None, None) otherwise
    """

    if face_crop is None or face_crop.size == 0:
        return None, None

    best_student = None
    best_distance = float("inf")

    for student_name, image_path in known_students.items():
        try:
            result = DeepFace.verify(
                img1_path=image_path,
                img2_path=face_crop,
                model_name=FACE_MODEL,
                detector_backend=FACE_DETECTOR,
                enforce_detection=True,
                align=True,
                normalization="base"
            )

            distance = float(result.get("distance", float("inf")))
            verified = bool(result.get("verified", False))

            if verified and distance <= FACE_DISTANCE_THRESHOLD:
                if distance < best_distance:
                    best_distance = distance
                    best_student = student_name

        except Exception as error:
            # Do not spam the terminal every frame.
            # A failed comparison simply means this candidate was not
            # successfully verified.
            print(f"Recognition warning for {student_name}: {error}")

    if best_student is not None:
        return best_student, best_distance

    return None, None


# ============================================================
# DATABASE DISPLAY
# ============================================================

def print_today_attendance():
    records = get_today_attendance()

    print()
    print("=" * 70)
    print("TODAY'S ATTENDANCE")
    print("=" * 70)

    if not records:
        print("No attendance records yet.")
    else:
        for record in records:
            student = record[0]
            check_in_time = record[2]
            check_out_time = record[3]

            print(
                f"{student:25} | "
                f"IN: {check_in_time or '--'} | "
                f"OUT: {check_out_time or '--'}"
            )

    print("=" * 70)
    print()


# ============================================================
# FIND THE BEST YOLO PERSON
# ============================================================

def get_largest_person_box(results, frame_width, frame_height):
    """
    Select the largest detected person.

    For a classroom attendance station, this makes the system focus
    on the person closest/largest in front of the camera.
    """

    best_box = None
    best_area = 0

    for result in results:
        if result.boxes is None:
            continue

        for box in result.boxes:
            confidence = float(box.conf[0])

            if confidence < YOLO_CONFIDENCE:
                continue

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0].tolist()
            )

            x1 = max(0, min(x1, frame_width - 1))
            y1 = max(0, min(y1, frame_height - 1))
            x2 = max(0, min(x2, frame_width - 1))
            y2 = max(0, min(y2, frame_height - 1))

            if x2 <= x1 or y2 <= y1:
                continue

            area = (x2 - x1) * (y2 - y1)

            if area > best_area:
                best_area = area
                best_box = (x1, y1, x2, y2, confidence)

    return best_box


# ============================================================
# CROP PERSON FOR DEEPFACE
# ============================================================

def crop_person(frame, person_box):
    """
    Crop the YOLO person box and add a small margin.

    DeepFace then searches for the face inside this smaller image
    instead of processing the entire webcam frame.
    """

    if person_box is None:
        return None, None

    x1, y1, x2, y2, confidence = person_box

    width = x2 - x1
    height = y2 - y1

    # Add a small margin around the person.
    margin_x = int(width * 0.08)
    margin_y = int(height * 0.08)

    x1 = max(0, x1 - margin_x)
    y1 = max(0, y1 - margin_y)
    x2 = min(frame.shape[1], x2 + margin_x)
    y2 = min(frame.shape[0], y2 + margin_y)

    crop = frame[y1:y2, x1:x2].copy()

    return crop, (x1, y1, x2, y2, confidence)


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():
    print()
    print("=" * 70)
    print("              AI FACE ATTENDANCE SYSTEM")
    print("=" * 70)
    print()

    # --------------------------------------------------------
    # Database
    # --------------------------------------------------------

    create_database()

    # --------------------------------------------------------
    # Load students
    # --------------------------------------------------------

    known_students = load_known_students()

    if not known_students:
        print("ERROR: No student images found.")
        print()
        print("Put images inside:")
        print(KNOWN_FACES_DIR)
        print()
        print("Example:")
        print(os.path.join(KNOWN_FACES_DIR, "ubaid_khan.jpg"))
        return

    print("Registered students:")

    for student in known_students:
        print(" -", student)

    print()

    # --------------------------------------------------------
    # Load YOLO11n
    # --------------------------------------------------------

    print("Loading YOLO11n...")

    try:
        yolo = YOLO(YOLO_MODEL)
    except Exception as error:
        print("ERROR: Could not load YOLO11n.")
        print(error)
        print()
        print("Make sure yolo11n.pt exists or that Ultralytics can download it.")
        return

    print("YOLO11n loaded successfully.")

    # --------------------------------------------------------
    # Start camera
    # --------------------------------------------------------

    camera = cv2.VideoCapture(CAMERA_INDEX)

    if not camera.isOpened():
        print("ERROR: Could not open camera.")
        return

    # Request a reasonable webcam resolution.
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print()
    print("=" * 70)
    print("CAMERA STARTED")
    print("=" * 70)
    print()
    print("I = CHECK-IN MODE")
    print("O = CHECK-OUT MODE")
    print("A = SHOW TODAY'S ATTENDANCE")
    print("Q = QUIT")
    print()

    # --------------------------------------------------------
    # Variables
    # --------------------------------------------------------

    mode = "CHECK-IN"

    last_recognition_time = 0

    last_student = None
    last_distance = None

    status_message = "Ready - face the camera"
    status_time = time.time()

    recognition_cooldown = {}

    # Current YOLO person box
    current_person_box = None

    # --------------------------------------------------------
    # Camera loop
    # --------------------------------------------------------

    while True:
        success, frame = camera.read()

        if not success:
            print("ERROR: Could not read camera.")
            break

        frame = cv2.flip(frame, 1)

        current_time = time.time()

        frame_height, frame_width = frame.shape[:2]

        # ====================================================
        # YOLO11n PERSON DETECTION
        # ====================================================

        results = yolo(
            frame,
            verbose=False,
            classes=[0],       # COCO class 0 = person
            conf=YOLO_CONFIDENCE
        )

        # Draw all person detections.
        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0].tolist()
                )

                confidence = float(box.conf[0])

                x1 = max(0, min(x1, frame_width - 1))
                y1 = max(0, min(y1, frame_height - 1))
                x2 = max(0, min(x2, frame_width - 1))
                y2 = max(0, min(y2, frame_height - 1))

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (255, 180, 0),
                    2
                )

                cv2.putText(
                    frame,
                    f"Person {confidence:.2f}",
                    (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 180, 0),
                    2
                )

        # Select the largest person.
        current_person_box = get_largest_person_box(
            results,
            frame_width,
            frame_height
        )

        # ====================================================
        # FACE RECOGNITION
        # ====================================================

        if current_time - last_recognition_time >= RECOGNITION_INTERVAL:
            last_recognition_time = current_time

            if current_person_box is None:
                last_student = "No Person"
                last_distance = None

            else:
                person_crop, display_box = crop_person(
                    frame,
                    current_person_box
                )

                try:
                    student_name, distance = recognize_student(
                        person_crop,
                        known_students
                    )

                    if student_name:
                        last_student = student_name
                        last_distance = distance

                        # Draw a stronger box around the recognized person.
                        x1, y1, x2, y2, confidence = display_box

                        cv2.rectangle(
                            frame,
                            (x1, y1),
                            (x2, y2),
                            (0, 255, 0),
                            3
                        )

                        cv2.putText(
                            frame,
                            f"IDENTIFIED: {student_name}",
                            (x1, max(30, y1 - 15)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.75,
                            (0, 255, 0),
                            2
                        )

                        # ------------------------------------------------
                        # Prevent repeated database actions
                        # ------------------------------------------------

                        last_action_time = recognition_cooldown.get(
                            student_name,
                            0
                        )

                        if (
                            current_time - last_action_time
                            >= ATTENDANCE_ACTION_COOLDOWN_SECONDS
                        ):
                            recognition_cooldown[student_name] = current_time

                            # --------------------------------------------
                            # CHECK-IN
                            # --------------------------------------------

                            if mode == "CHECK-IN":
                                success_action, message = check_in(
                                    student_name
                                )

                                status_message = message
                                status_time = time.time()
                                print(message)

                            # --------------------------------------------
                            # CHECK-OUT
                            # --------------------------------------------

                            elif mode == "CHECK-OUT":
                                success_action, message = check_out(
                                    student_name
                                )

                                status_message = message
                                status_time = time.time()
                                print(message)

                    else:
                        last_student = "Unknown"
                        last_distance = None
                        status_message = "Face detected, but student not recognized"
                        status_time = time.time()

                except Exception as error:
                    print("Recognition error:", error)

                    last_student = "Recognition Error"
                    last_distance = None
                    status_message = "Could not process face"
                    status_time = time.time()

        # ====================================================
        # DATE / TIME
        # ====================================================

        now = datetime.now()

        date_text = now.strftime("%d-%m-%Y")
        time_text = now.strftime("%I:%M:%S %p")

        # ====================================================
        # HEADER
        # ====================================================

        cv2.rectangle(
            frame,
            (0, 0),
            (frame.shape[1], 130),
            (30, 30, 30),
            -1
        )

        cv2.putText(
            frame,
            "AI FACE ATTENDANCE SYSTEM",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Date: {date_text}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Time: {time_text}",
            (20, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        # ====================================================
        # MODE
        # ====================================================

        mode_color = (
            (0, 255, 0)
            if mode == "CHECK-IN"
            else (0, 165, 255)
        )

        cv2.putText(
            frame,
            f"MODE: {mode}",
            (frame.shape[1] - 260, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            mode_color,
            2
        )

        # ====================================================
        # RECOGNIZED STUDENT
        # ====================================================

        if last_student:
            if last_student == "Unknown":
                student_color = (0, 0, 255)
            elif last_student in ("No Person", "Recognition Error"):
                student_color = (0, 255, 255)
            else:
                student_color = (0, 255, 0)

            display_text = f"Student: {last_student}"

            if last_distance is not None:
                display_text += f" | Distance: {last_distance:.3f}"

            cv2.putText(
                frame,
                display_text,
                (20, frame.shape[0] - 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                student_color,
                2
            )

        # ====================================================
        # STATUS MESSAGE
        # ====================================================

        if time.time() - status_time < 5:
            cv2.putText(
                frame,
                status_message,
                (20, frame.shape[0] - 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2
            )

        # ====================================================
        # CONTROLS
        # ====================================================

        cv2.putText(
            frame,
            "I: Check-In | O: Check-Out | A: Attendance | Q: Quit",
            (20, frame.shape[0] - 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

        # ====================================================
        # SHOW CAMERA
        # ====================================================

        cv2.imshow(
            "AI Face Attendance System",
            frame
        )

        # ====================================================
        # KEYBOARD
        # ====================================================

        key = cv2.waitKey(1) & 0xFF

        if key == ord("i"):
            mode = "CHECK-IN"
            status_message = "CHECK-IN MODE"
            status_time = time.time()
            print("\nMode changed to CHECK-IN")

        elif key == ord("o"):
            mode = "CHECK-OUT"
            status_message = "CHECK-OUT MODE"
            status_time = time.time()
            print("\nMode changed to CHECK-OUT")

        elif key == ord("a"):
            print_today_attendance()

        elif key == ord("q"):
            break

    # ========================================================
    # CLEANUP
    # ========================================================

    camera.release()
    cv2.destroyAllWindows()

    print()
    print("Camera stopped.")
    print_today_attendance()


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
