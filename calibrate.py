import cv2
import numpy as np
import os
import glob

class Calibration(object):
    def __init__(self,size):
        self.size = size
        self.k = 0
        self.d = 0
        self._img_shape = None

    def Checkboard(self):
        CHECKERBOARD = (6,8)#(,9)
        subpix_criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
        calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv2.fisheye.CALIB_FIX_SKEW+cv2.fisheye.CALIB_CHECK_COND
        objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
        objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
        objpoints = [] # 3d point in real world space
        imgpoints = [] # 2d points in image plane.
        images = glob.glob('Chessboard/*.jpg')
        for fname in images:
            img = cv2.imread(fname)
            img = cv2.resize(img, (self.size[0],self.size[1]), interpolation = cv2.INTER_AREA)
            if self._img_shape == None:
                self._img_shape = img.shape[:2]
            else:
                assert self._img_shape == img.shape[:2], "All images must share the same size."
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)
            # If found, add object points, image points (after refining them)
            if ret == True:
                objpoints.append(objp)
                cv2.cornerSubPix(gray,corners,(3,3),(-1,-1),subpix_criteria)
                imgpoints.append(corners)

        N_OK = len(objpoints)
        self.k = np.zeros((3, 3))
        self.d = np.zeros((4, 1))
        rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
        tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
        rms, _, _, _, _ = \
            cv2.fisheye.calibrate(
                objpoints,
                imgpoints,
                gray.shape[::-1],
                self.k,
                self.d,
                rvecs,
                tvecs,
                calibration_flags,
                (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
            )
        self._img_shape = self._img_shape[::-1]
        # print("Found " + str(N_OK) + " valid images for calibration")
        # print("DIM=" + str(self._img_shape[::-1]))
        # print("K=np.array(" + str(self.k.tolist()) + ")")
        # print("D=np.array(" + str(self.d.tolist()) + ")")

    def Undistort(self,img, balance, dim2=None, dim3=None):
        if not dim2:
            dim2 = self._img_shape
        if not dim3:
            dim3 = self._img_shape

        new_k = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(self.k, self.d, dim2, np.eye(3), balance=balance)
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(self.k, self.d, np.eye(3), new_k, dim3, cv2.CV_16SC2)
        undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        return(undistorted_img)
