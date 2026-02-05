import cv2
import numpy as np
file = "No1_camera.npz"
z = np.load(file)
# print(z)
# print(z['camera_matrix'])
# print(z['dist_coeffs'])
# print(z['rvecs'])
# print(z['tvecs'])
camera_matrix = z["camera_matrix"]
dist_coeffs = z["dist_coeffs"]
# rvecs = z["rvecs"]
# tvecs = z["tvecs"]

# test_image = cv2.imread("test.jpg")
# # cv2.drawKeypoints(test_image, cv2.KeyPoint(100, 100, 1), test_image, color=
# cv2.circle(test_image, center=(405, 200), radius=2, color=(255, 0, 0), thickness=2)
# cv2.imshow("Camera", test_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
points = np.array([405, 200]).reshape(-1, 1, 2)
result = cv2.undistort(np.array(points), cameraMatrix=camera_matrix, distCoeffs=dist_coeffs)
print(result)
