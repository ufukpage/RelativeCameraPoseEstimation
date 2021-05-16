
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math

fx, fy = 763, 763
cx, cy = 960, 540
MIN_MATCH_COUNT = 8
FLANN_INDEX_KDTREE = 1


def rotationMatrixToEulerAngles(R):
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])


def getBestMatches(ref_des, q_des, ratio=0.8):
    bf = cv2.BFMatcher()
    # return  bf.match(ref_des, q_des)
    # index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    # search_params = dict(checks=50)
    # flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = bf.knnMatch(ref_des, q_des, k=2)  # first k best matches

    best_matches = []

    # from Lowe's
    for i, (m, n) in enumerate(matches):
        if m.distance < ratio * n.distance:
            best_matches.append(m)
    return best_matches


def getTransformation(ref_img, query_img):
    sift = cv2.SIFT_create()

    ref_kp, ref_des = sift.detectAndCompute(ref_img, None)
    q_kp, q_des = sift.detectAndCompute(query_img, None)
    best_matches = getBestMatches(ref_des, q_des, ratio=0.95)

    # img3 =cv2.drawMatches(ref_img, ref_kp, query_img, q_kp,best_matches[-10:0], None)
    # plt.imshow(img3), plt.show()

    if len(best_matches) > MIN_MATCH_COUNT:
        # Camera Intrisics matrix
        K = np.float64([[fx, 0, cx],
                      [0, fy, cy],
                      [0, 0, 1]])

        src_pts = np.float32([ref_kp[m.queryIdx].pt for m in best_matches])
        dst_pts = np.float32([q_kp[m.trainIdx].pt for m in best_matches])

        E, mask = cv2.findEssentialMat(src_pts, dst_pts, K, method=cv2.RANSAC,  prob=0.999)

        points, R_est, t_est, mask_pose = cv2.recoverPose(E, src_pts, dst_pts, K)

        # extrinsic matrix
        # ext_mat = np.hstack((R_est, t_est))

        print("Rotation:", rotationMatrixToEulerAngles(R_est))
        print("Transformation:", t_est)
    else:
        print("not enough matches")


def printRelativePose(ref_fname, query_fname):
    ref_img = cv2.imread(ref_fname)
    query_img = cv2.imread(query_fname)

    print("Ref image: ", ref_fname)
    print("Query image: ", query_fname)
    getTransformation(ref_img, query_img)


if "__main__" == __name__:

    ref_fname = "img1.png"
    query_fname = "img2.png"

    printRelativePose(ref_fname, query_fname)

    ref_fname = "img1.png"
    query_fname = "img3.png"

    printRelativePose(ref_fname, query_fname)
