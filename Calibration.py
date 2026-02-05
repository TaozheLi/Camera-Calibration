import cv2
import numpy as np
import glob
import os

# ==============================
# Parameters (CHANGE THESE)
# ==============================
CHECKERBOARD = (13, 8)          # inner corners (columns, rows)
SQUARE_SIZE = 0.066          # meters (or any unit, but be consistent)
IMAGE_DIR = "./saved_image/"

# ==============================
# Prepare object points
# ==============================
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0],
                       0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

objpoints = []  # 3D points
imgpoints = []  # 2D points
error_threshold = 0.3
# ==============================
# Load calibration images
# ==============================
images = os.listdir(IMAGE_DIR)
# images = ["/home/simmons/Downloads/test.png"]
for fname in images:
    fpath = os.path.join(IMAGE_DIR, fname)
    img = cv2.imread(fpath)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(
        gray,
        CHECKERBOARD
    )
    cv2.imshow("img", gray)
    cv2.waitKey(0)
    print(gray)
    print(corners)
    if ret:
        print(objp)
        objpoints.append(objp)
        print("detected corners")
        corners2 = cv2.cornerSubPix(
            gray,
            corners,
            (11, 11),
            (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        )
        imgpoints.append(corners2)

        cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        cv2.imshow("Detected Corners", img)
        cv2.waitKey(0)

cv2.destroyAllWindows()

# ==============================
# Camera calibration
# ==============================
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    objpoints,
    imgpoints,
    gray.shape[::-1],
    None,
    None
)

print("Reprojection error:", ret)
print("\nCamera matrix:\n", camera_matrix)
print("\nDistortion coefficients:\n", dist_coeffs)

# ==============================
# Save calibration
# ==============================
np.savez(
    "No1_camera.npz",
    camera_matrix=camera_matrix,
    dist_coeffs=dist_coeffs
)
