import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
eyes_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)


def find_iris(eye_roi):
    gray_eye = cv2.cvtColor(eye_roi, cv2.COLOR_BGR2GRAY)
    gray_eye = cv2.GaussianBlur(gray_eye, (9, 9), 2)

    circles = cv2.HoughCircles(
        gray_eye,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=20,
        param1=100,
        param2=20,
        minRadius=5,
        maxRadius=min(gray_eye.shape) // 2,
    )

    if circles is None:
        return None

    circles = np.round(circles[0, :]).astype(int)
    best_circle = max(circles, key=lambda c: c[2])
    cx, cy, radius = best_circle
    return (cx, cy, radius)


def process_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

    if len(faces) == 0:
        cv2.putText(frame, "No face detected", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return frame

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        face_roi = frame[y : y + h, x : x + w]
        face_gray = gray[y : y + h, x : x + w]
        eyes = eyes_cascade.detectMultiScale(face_gray, scaleFactor=1.1, minNeighbors=8, minSize=(25, 25))

        if len(eyes) == 0:
            cv2.putText(frame, "Face detected - Eyes not found", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            continue

        for (ex, ey, ew, eh) in eyes:
            eye_roi = face_roi[ey : ey + eh, ex : ex + ew]
            iris = find_iris(eye_roi)

            if iris is not None:
                cx, cy, radius = iris
                center = (x + ex + cx, y + ey + cy)
                cv2.circle(frame, center, radius, (0, 255, 0), 2)
                cv2.putText(frame, "Retina detected", (x, y + h + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Eye found", (x + ex, y + ey - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

    return frame


def main():
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        raise RuntimeError("Webcam not found or cannot be opened.")

    while True:
        ret, frame = camera.read()
        if not ret:
            print("Failed to capture frame.")
            break

        frame = process_frame(frame)
        cv2.imshow("Human Retina Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
