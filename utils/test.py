import cv2

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use DirectShow backend instead of MSMF

if not cap.isOpened():
    print("Error: Could not open the camera.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Cannot read from the camera.")
        break

    cv2.imshow('Camera Test', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
