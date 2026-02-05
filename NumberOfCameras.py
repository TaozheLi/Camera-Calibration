import cv2

MAX_CAMERAS = 100  # test indices 0–9

available_cameras = []

for i in range(MAX_CAMERAS):
    cap = cv2.VideoCapture(i, cv2.CAP_V4L2)  # use V4L2 on Linux
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print(f"✅ Camera index {i} is available")
            available_cameras.append(i)
        else:
            print(f"⚠️ Camera index {i} opened but no frame")
        cap.release()
    else:
        print(f"❌ Camera index {i} cannot be opened")

print("\nAvailable camera indices:", available_cameras)
