import cv2
import os
import time
import sqlite3
import threading
import datetime
import tkinter as tk

from PIL import Image, ImageTk
from deepface import DeepFace
from whatsapp_api_client_python import API


# ============================================================
# CONFIGURATION
# ============================================================

# ------------------------------------------------------------
# CAMERA
# ------------------------------------------------------------

CAMERA_INDEX = 0


# ------------------------------------------------------------
# FACE RECOGNITION
# ------------------------------------------------------------

MODEL_NAME = "Facenet"

DETECTOR_BACKEND = "opencv"

DISTANCE_METRIC = "cosine"

# Lower value = stricter recognition
FACE_DISTANCE_THRESHOLD = 0.45

# Run face recognition every X seconds
RECOGNITION_INTERVAL = 1.0

# Prevent duplicate attendance
COOLDOWN_SECONDS = 60


# ------------------------------------------------------------
# KNOWN FACES
# ------------------------------------------------------------

PICS_DIR = "known_faces"

if not os.path.exists(PICS_DIR):
    os.makedirs(PICS_DIR)


# ------------------------------------------------------------
# DATABASE
# ------------------------------------------------------------

DATABASE_FILE = "attendance.db"


# ------------------------------------------------------------
# WHATSAPP / GREEN API
# ------------------------------------------------------------

# IMPORTANT:
# Replace these with your NEW Green API credentials.

ID_INSTANCE = "710722742243"

API_TOKEN_INSTANCE = "6f726ee34e92495fbfe5170f01f4d530b3cd38138b224d068f"

# Example:
# Pakistani number:
# 03001234567
#
# Green API format:
# 923001234567@c.us

RECIPIENT_PHONE = "923397070799@c.us"


# Create Green API client
green_api = API.GreenApi(
    ID_INSTANCE,
    API_TOKEN_INSTANCE
)


# ============================================================
# DATABASE SETUP
# ============================================================

def initialize_database():

    connection = sqlite3.connect(DATABASE_FILE)

    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_name TEXT NOT NULL,
            action TEXT NOT NULL,
            date TEXT NOT NULL,
            time TEXT NOT NULL
        )
    """)

    connection.commit()
    connection.close()


initialize_database()


# ============================================================
# WHATSAPP MESSAGE
# ============================================================

def send_whatsapp_alert(student_name, action):

    def send_request():

        current_time = datetime.datetime.now()

        date_str = current_time.strftime("%d-%m-%Y")

        time_str = current_time.strftime("%I:%M:%S %p")

        message = (
            "🔔 *School Attendance Alert*\n\n"
            f"👤 *Student:* {student_name}\n"
            f"📌 *Action:* {action}\n"
            f"📅 *Date:* {date_str}\n"
            f"⏰ *Time:* {time_str}\n\n"
            "🤖 AI Face Recognition Attendance System"
        )

        try:

            response = green_api.sending.sendMessage(
                RECIPIENT_PHONE,
                message
            )

            if response.code == 200:

                print(
                    f"[WHATSAPP SUCCESS] "
                    f"{student_name} - {action}"
                )

            else:

                print(
                    f"[WHATSAPP ERROR] "
                    f"Status Code: {response.code}"
                )

        except Exception as error:

            print(
                f"[WHATSAPP ERROR] {error}"
            )

    # Run WhatsApp in background
    # so camera does not freeze
    threading.Thread(
        target=send_request,
        daemon=True
    ).start()


# ============================================================
# SAVE ATTENDANCE
# ============================================================

def save_attendance(student_name, action):

    current_time = datetime.datetime.now()

    date_str = current_time.strftime("%Y-%m-%d")

    time_str = current_time.strftime("%H:%M:%S")

    try:

        connection = sqlite3.connect(
            DATABASE_FILE
        )

        cursor = connection.cursor()

        cursor.execute(
            """
            INSERT INTO attendance
            (
                student_name,
                action,
                date,
                time
            )
            VALUES (?, ?, ?, ?)
            """,
            (
                student_name,
                action,
                date_str,
                time_str
            )
        )

        connection.commit()

        connection.close()

        print(
            f"[DATABASE] "
            f"{student_name} - {action} "
            f"{date_str} {time_str}"
        )

    except Exception as error:

        print(
            f"[DATABASE ERROR] {error}"
        )


# ============================================================
# LOAD KNOWN FACE IMAGES
# ============================================================

def get_known_face_images():

    known_faces = []

    if not os.path.exists(PICS_DIR):
        return known_faces

    for student_folder in os.listdir(PICS_DIR):

        student_path = os.path.join(
            PICS_DIR,
            student_folder
        )

        if not os.path.isdir(student_path):
            continue

        for file_name in os.listdir(student_path):

            if file_name.lower().endswith(
                (".jpg", ".jpeg", ".png")
            ):

                image_path = os.path.join(
                    student_path,
                    file_name
                )

                known_faces.append(
                    (
                        student_folder,
                        image_path
                    )
                )

    return known_faces


# ============================================================
# FACE RECOGNITION
# ============================================================

def match_face_with_deepface(frame):

    known_faces = get_known_face_images()

    if len(known_faces) == 0:

        print(
            "[WARNING] No student photos found."
        )

        print(
            "[INFO] Example:"
        )

        print(
            "known_faces/ubaid/ubaid.jpg"
        )

        return "Unknown"


    print(
        f"[INFO] Checking "
        f"{len(known_faces)} known face image(s)..."
    )


    for student_name, image_path in known_faces:

        try:

            result = DeepFace.verify(

                img1_path=frame,

                img2_path=image_path,

                model_name=MODEL_NAME,

                detector_backend=DETECTOR_BACKEND,

                distance_metric=DISTANCE_METRIC,

                enforce_detection=False

            )


            distance = result.get(
                "distance",
                1.0
            )

            verified = result.get(
                "verified",
                False
            )


            print(
                f"[FACE CHECK] "
                f"{student_name} | "
                f"distance={distance:.4f} | "
                f"verified={verified}"
            )


            if (
                verified
                or
                distance <= FACE_DISTANCE_THRESHOLD
            ):

                print(
                    f"[MATCH FOUND] "
                    f"{student_name}"
                )

                return student_name


        except Exception as error:

            print(
                f"[DEEPFACE ERROR] "
                f"{student_name}: {error}"
            )


    print(
        "[NO MATCH] Unknown student"
    )

    return "Unknown"


# ============================================================
# TKINTER APPLICATION
# ============================================================

class FaceAttendanceApp:

    def __init__(self, window):

        self.window = window

        self.window.title(
            "AI School Entry / Exit Attendance System"
        )

        self.window.geometry(
            "1000x800"
        )

        self.window.protocol(
            "WM_DELETE_WINDOW",
            self.quit_app
        )


        # ----------------------------------------------------
        # APPLICATION VARIABLES
        # ----------------------------------------------------

        self.cap = cv2.VideoCapture(
            CAMERA_INDEX
        )

        self.current_frame = None

        self.detected_name = "Unknown"

        self.is_processing = False

        self.last_recognition_time = 0

        self.last_student = None

        self.last_attendance_time = 0

        # Current mode
        self.mode = "ENTRY"


        # ----------------------------------------------------
        # TITLE
        # ----------------------------------------------------

        self.title_label = tk.Label(
            self.window,
            text="AI SCHOOL ATTENDANCE SYSTEM",
            font=("Arial", 24, "bold")
        )

        self.title_label.pack(
            pady=10
        )


        # ----------------------------------------------------
        # MODE BUTTON FRAME
        # ----------------------------------------------------

        self.mode_frame = tk.Frame(
            self.window
        )

        self.mode_frame.pack(
            pady=5
        )


        # ENTRY BUTTON
        self.entry_button = tk.Button(
            self.mode_frame,
            text="ENTRY",
            font=("Arial", 16, "bold"),
            width=12,
            height=2,
            command=self.set_entry_mode
        )

        self.entry_button.grid(
            row=0,
            column=0,
            padx=10
        )


        # EXIT BUTTON
        self.exit_button = tk.Button(
            self.mode_frame,
            text="EXIT",
            font=("Arial", 16, "bold"),
            width=12,
            height=2,
            command=self.set_exit_mode
        )

        self.exit_button.grid(
            row=0,
            column=1,
            padx=10
        )


        # ----------------------------------------------------
        # MODE LABEL
        # ----------------------------------------------------

        self.mode_label = tk.Label(
            self.window,
            text="MODE: ENTRY",
            font=("Arial", 20, "bold")
        )

        self.mode_label.pack(
            pady=8
        )


        # ----------------------------------------------------
        # STATUS LABEL
        # ----------------------------------------------------

        self.status_label = tk.Label(
            self.window,
            text="Starting camera...",
            font=("Arial", 16, "bold")
        )

        self.status_label.pack(
            pady=5
        )


        # ----------------------------------------------------
        # RESULT LABEL
        # ----------------------------------------------------

        self.result_label = tk.Label(
            self.window,
            text="Student: Unknown",
            font=("Arial", 20, "bold")
        )

        self.result_label.pack(
            pady=8
        )


        # ----------------------------------------------------
        # CAMERA
        # ----------------------------------------------------

        self.cam_label = tk.Label(
            self.window
        )

        self.cam_label.pack(
            pady=10
        )


        # ----------------------------------------------------
        # QUIT BUTTON
        # ----------------------------------------------------

        self.quit_button = tk.Button(
            self.window,
            text="QUIT",
            font=("Arial", 14, "bold"),
            width=12,
            command=self.quit_app
        )

        self.quit_button.pack(
            pady=10
        )


        # ----------------------------------------------------
        # SET INITIAL BUTTON STATE
        # ----------------------------------------------------

        self.update_mode_buttons()


        # ----------------------------------------------------
        # START CAMERA
        # ----------------------------------------------------

        self.update_video()


    # ========================================================
    # SET ENTRY MODE
    # ========================================================

    def set_entry_mode(self):

        self.mode = "ENTRY"

        self.last_student = None
        self.last_attendance_time = 0

        self.update_mode_buttons()

        self.status_label.config(
            text="Status: Entry mode activated"
        )

        self.result_label.config(
            text="Student: Unknown"
        )

        print(
            "[MODE] ENTRY mode activated"
        )


    # ========================================================
    # SET EXIT MODE
    # ========================================================

    def set_exit_mode(self):

        self.mode = "EXIT"

        self.last_student = None
        self.last_attendance_time = 0

        self.update_mode_buttons()

        self.status_label.config(
            text="Status: Exit mode activated"
        )

        self.result_label.config(
            text="Student: Unknown"
        )

        print(
            "[MODE] EXIT mode activated"
        )


    # ========================================================
    # UPDATE BUTTON APPEARANCE
    # ========================================================

    def update_mode_buttons(self):

        if self.mode == "ENTRY":

            self.entry_button.config(
                relief=tk.SUNKEN
            )

            self.exit_button.config(
                relief=tk.RAISED
            )

            self.mode_label.config(
                text="MODE: ENTRY"
            )

        else:

            self.entry_button.config(
                relief=tk.RAISED
            )

            self.exit_button.config(
                relief=tk.SUNKEN
            )

            self.mode_label.config(
                text="MODE: EXIT"
            )


    # ========================================================
    # VIDEO UPDATE
    # ========================================================

    def update_video(self):

        ret, frame = self.cap.read()


        if ret:

            self.current_frame = frame.copy()


            # ------------------------------------------------
            # FACE DETECTION
            # ------------------------------------------------

            gray = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2GRAY
            )


            faces = face_cascade.detectMultiScale(

                gray,

                scaleFactor=1.1,

                minNeighbors=5,

                minSize=(80, 80)

            )


            # ------------------------------------------------
            # DRAW FACE BOXES
            # ------------------------------------------------

            for (
                x,
                y,
                w,
                h
            ) in faces:

                if self.detected_name != "Unknown":

                    box_color = (
                        0,
                        255,
                        0
                    )

                else:

                    box_color = (
                        0,
                        0,
                        255
                    )


                cv2.rectangle(

                    frame,

                    (x, y),

                    (x + w, y + h),

                    box_color,

                    2

                )


                cv2.putText(

                    frame,

                    self.detected_name,

                    (x, y - 10),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.8,

                    box_color,

                    2

                )


            # ------------------------------------------------
            # SHOW MODE ON CAMERA
            # ------------------------------------------------

            cv2.putText(

                frame,

                f"MODE: {self.mode}",

                (20, 40),

                cv2.FONT_HERSHEY_SIMPLEX,

                1,

                (255, 255, 0),

                2

            )


            # ------------------------------------------------
            # RECOGNITION TIMER
            # ------------------------------------------------

            current_time = time.time()


            if (

                len(faces) > 0

                and

                not self.is_processing

                and

                (
                    current_time
                    -
                    self.last_recognition_time
                )
                >= RECOGNITION_INTERVAL

            ):

                self.is_processing = True

                self.last_recognition_time = current_time

                frame_copy = self.current_frame.copy()


                threading.Thread(

                    target=self.run_recognition_thread,

                    args=(frame_copy,),

                    daemon=True

                ).start()


            # ------------------------------------------------
            # STATUS
            # ------------------------------------------------

            self.status_label.config(

                text=(
                    f"Status: "
                    f"{self.detected_name}"
                )

            )


            self.result_label.config(

                text=(
                    f"Student: "
                    f"{self.detected_name}"
                )

            )


            # ------------------------------------------------
            # DISPLAY FRAME
            # ------------------------------------------------

            rgb_frame = cv2.cvtColor(

                frame,

                cv2.COLOR_BGR2RGB

            )


            image = Image.fromarray(
                rgb_frame
            )


            image_tk = ImageTk.PhotoImage(
                image=image
            )


            self.cam_label.imgtk = image_tk

            self.cam_label.configure(
                image=image_tk
            )


        else:

            self.status_label.config(
                text="Camera error"
            )


        # Keep updating camera
        self.window.after(
            20,
            self.update_video
        )


    # ========================================================
    # DEEPFACE THREAD
    # ========================================================

    def run_recognition_thread(
        self,
        frame
    ):

        try:

            name = match_face_with_deepface(
                frame
            )


            self.detected_name = name


            if name != "Unknown":

                self.process_attendance(
                    name
                )


        except Exception as error:

            print(
                f"[RECOGNITION ERROR] "
                f"{error}"
            )


        finally:

            self.is_processing = False


    # ========================================================
    # ATTENDANCE PROCESSING
    # ========================================================

    def process_attendance(
        self,
        student_name
    ):

        current_time = time.time()


        # ----------------------------------------------------
        # PREVENT DUPLICATES
        # ----------------------------------------------------

        if (

            self.last_student
            ==
            student_name

            and

            (
                current_time
                -
                self.last_attendance_time
            )
            <
            COOLDOWN_SECONDS

        ):

            print(
                f"[COOLDOWN] "
                f"{student_name} already processed."
            )

            return


        # ----------------------------------------------------
        # UPDATE LAST ATTENDANCE
        # ----------------------------------------------------

        self.last_student = student_name

        self.last_attendance_time = current_time


        # ----------------------------------------------------
        # CREATE ACTION
        # ----------------------------------------------------

        if self.mode == "ENTRY":

            action = "ENTERED SCHOOL"

        else:

            action = "LEFT SCHOOL"


        # ----------------------------------------------------
        # PRINT RESULT
        # ----------------------------------------------------

        print(
            "===================================="
        )

        print(
            f"[ATTENDANCE] "
            f"{student_name}"
        )

        print(
            f"[MODE] "
            f"{self.mode}"
        )

        print(
            f"[ACTION] "
            f"{action}"
        )

        print(
            "===================================="
        )


        # ----------------------------------------------------
        # SAVE TO DATABASE
        # ----------------------------------------------------

        save_attendance(
            student_name,
            action
        )


        # ----------------------------------------------------
        # SEND WHATSAPP
        # ----------------------------------------------------

        send_whatsapp_alert(
            student_name,
            action
        )


        # ----------------------------------------------------
        # UPDATE GUI
        # ----------------------------------------------------

        self.window.after(
            0,
            lambda: self.show_success(
                student_name,
                action
            )
        )


    # ========================================================
    # SHOW SUCCESS MESSAGE
    # ========================================================

    def show_success(
        self,
        student_name,
        action
    ):

        self.result_label.config(
            text=(
                f"{student_name}\n"
                f"{action}"
            )
        )

        self.status_label.config(
            text=(
                "Attendance recorded + "
                "WhatsApp notification sent"
            )
        )


    # ========================================================
    # QUIT APPLICATION
    # ========================================================

    def quit_app(self):

        print(
            "[SYSTEM] Closing..."
        )


        if self.cap.isOpened():

            self.cap.release()


        self.window.destroy()


# ============================================================
# HAAR CASCADE
# ============================================================

face_cascade = cv2.CascadeClassifier(

    cv2.data.haarcascades
    +
    "haarcascade_frontalface_default.xml"

)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print(
        "========================================"
    )

    print(
        " AI SCHOOL ENTRY / EXIT ATTENDANCE"
    )

    print(
        "========================================"
    )

    print(
        f"MODEL: {MODEL_NAME}"
    )

    print(
        f"THRESHOLD: {FACE_DISTANCE_THRESHOLD}"
    )

    print(
        "ENTRY / EXIT BUTTONS ENABLED"
    )

    print(
        "WHATSAPP NOTIFICATIONS ENABLED"
    )

    print(
        "========================================"
    )


    root = tk.Tk()


    app = FaceAttendanceApp(
        root
    )


    root.mainloop()