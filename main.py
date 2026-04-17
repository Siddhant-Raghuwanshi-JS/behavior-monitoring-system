import cv2
import mediapipe as mp
import time


import csv

log_file = open("log.csv", "a", newline="")
csv_writer = csv.writer(log_file)

mp_face_mesh = mp.solutions.face_mesh

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

last_seen_time = time.time()
looking_away_start = None

# 🧠 Suspicion system
suspicion_score = 0
last_score_update = time.time()

with mp_face_mesh.FaceMesh(refine_landmarks=True) as face_mesh:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        face_detected = False

        if results.multi_face_landmarks:
            face_detected = True
            last_seen_time = time.time()

            for face_landmarks in results.multi_face_landmarks:
                h, w, _ = frame.shape

                # Eye landmarks
                left_x = face_landmarks.landmark[33].x
                right_x = face_landmarks.landmark[263].x

                direction = "Center"

                if left_x < 0.4:
                    direction = "Left"
                elif right_x > 0.6:
                    direction = "Right"

                cv2.putText(frame, f"Looking {direction}", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 255, 0), 2)

                # 🧠 Attention Logic
                if direction != "Center":
                    if looking_away_start is None:
                        looking_away_start = time.time()
                    elif time.time() - looking_away_start > 3:
                        cv2.putText(frame, "Not Paying Attention!",
                                    (50, 150),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    1,
                                    (0, 0, 255),
                                    2)
                        suspicion_score += 2
                        csv_writer.writerow([time.time(), "Not Paying Attention", suspicion_score])
                else:
                    looking_away_start = None

        # Face missing logic
        if not face_detected:
            if time.time() - last_seen_time > 3:
                cv2.putText(frame, "Face not detected!",
                            (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 0, 255),
                            2)
                suspicion_score += 3
                csv_writer.writerow([time.time(), "Face Missing", suspicion_score])

        # 🧠 Score decay (reduces over time)
        if time.time() - last_score_update > 5:
            suspicion_score = max(0, suspicion_score - 1)
            last_score_update = time.time()

        # Show score
        cv2.putText(frame, f"Score: {suspicion_score}",
                    (50, 200),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 0),
                    2)

        # Final alert
        if suspicion_score > 5:
            cv2.putText(frame, "HIGH ALERT!",
                        (50, 250),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        3)

        cv2.imshow("Behavior Monitor", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

log_file.close()