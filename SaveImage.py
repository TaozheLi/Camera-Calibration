import os
import cv2

cap = cv2.VideoCapture(0)  # 0 = default camera

if not cap.isOpened():
    print("❌ Cannot open camera")
    exit()

count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Can't receive frame (stream end?)")
        break

    cv2.imshow("Camera", frame)
    output_dir = "saved_image"
    img_name = f"test.jpg".format(count)
    os.makedirs(output_dir, exist_ok=True)
    img_path = os.path.join(output_dir, img_name)
    _ = cv2.waitKey(1) & 0xFF
    if _ == ord("s"):
        print("save one image in path: {}".format(img_path))
        cv2.imwrite(img_name, frame)
        count += 1
    elif _ == ord('q'):
        break
    else:
        continue

cap.release()
cv2.destroyAllWindows()
