""" Test face detection for static image """
import cv2


def main():
    cascade = cv2.CascadeClassifier("parameters.xml")

    frame = cv2.imread("test.jpeg")
    rectangles = cascade.detectMultiScale(
        frame, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE)

    for x, y, w, h in rectangles:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imwrite("out.jpeg", frame)


if __name__ == '__main__':
    main()
