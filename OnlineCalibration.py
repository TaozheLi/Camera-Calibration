import cv2
import numpy as np


# ==============================
# Parameters (CHANGE THESE)
# ==============================
CHECKERBOARD = (13, 8)          # inner corners (columns, rows)
SQUARE_SIZE = 0.066          # meters (or any unit, but be consistent)
cap = cv2.VideoCapture(0)
# ==============================
# Prepare object points
# ==============================
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0],
                       0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

error = float('inf')
min_error = float('inf')
error_threshold = 0.7
objpoints = []  # 3D points
imgpoints = []  # 2D points
n_frames = 0
while error > error_threshold:
    if n_frames < 100:
        n_frames += 1
        continue
    ret, frame = cap.read()
    if ret:
        cv2.imshow("frame", frame)
        cv2.waitKey(1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        success, corners = cv2.findChessboardCorners(
            gray,
            CHECKERBOARD
        )
        if success:
            # print("detected corners")
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(
                gray,
                corners,
        (13, 8),
        (-1, -1),
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            )
            imgpoints.append(corners2)
            cv2.drawChessboardCorners(frame, CHECKERBOARD, corners2, success)
            cv2.imshow("Detected Corners", frame)
            cv2.waitKey(1)

            error, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
                objpoints,
                imgpoints,
                gray.shape[::-1],
                None,
                None
            )
            if error < min_error:
                np.savez(
                    "No1_camera.npz",
                    camera_matrix=camera_matrix,
                    dist_coeffs=dist_coeffs,
                    revecs = rvecs,
                    tvecs = tvecs,
                )
                min_error = error

        print("error: ", error)
        n_frames += 1
    else:
        print("error, camera closed")

cv2.destroyAllWindows()

# ==============================
# Camera calibration
# ==============================


print("Reprojection error:", min_error)
print("\nCamera matrix:\n", camera_matrix)
print("\nDistortion coefficients:\n", dist_coeffs)
print("rvecs:\n", rvecs)
print("tvecs:\n", tvecs)

# ==============================
# Save calibration
# ==============================
