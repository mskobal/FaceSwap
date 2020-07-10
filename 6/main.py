""" test for video camera """

import cv2


def main():
    cap = cv2.VideoCapture(0)
    cascade = cv2.CascadeClassifier("parameters.xml")

    while True:
        ret, frame = cap.read()
        rectangles = cascade.detectMultiScale(
            frame, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE)

        for x, y, w, h in rectangles:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == "__main__":
    main()
