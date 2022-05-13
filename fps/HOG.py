import argparse
import os
import time

import cv2
import dlib


def detectFaceDlibHog(detector, frame, inHeight=300, inWidth=0):

    frameDlibHog = frame.copy()
    frameHeight = frameDlibHog.shape[0]
    frameWidth = frameDlibHog.shape[1]
    if not inWidth:
        inWidth = int((frameWidth / frameHeight) * inHeight)

    scaleHeight = frameHeight / inHeight
    scaleWidth = frameWidth / inWidth

    frameDlibHogSmall = cv2.resize(frameDlibHog, (inWidth, inHeight))

    frameDlibHogSmall = cv2.cvtColor(frameDlibHogSmall, cv2.COLOR_BGR2RGB)
    faceRects = detector(frameDlibHogSmall, 0)
    print(frameWidth, frameHeight, inWidth, inHeight)
    bboxes = []
    for faceRect in faceRects:

        cvRect = [
            int(faceRect.left() * scaleWidth),
            int(faceRect.top() * scaleHeight),
            int(faceRect.right() * scaleWidth),
            int(faceRect.bottom() * scaleHeight),
        ]
        bboxes.append(cvRect)
        cv2.rectangle(
            frameDlibHog,
            (cvRect[0], cvRect[1]),
            (cvRect[2], cvRect[3]),
            (0, 255, 0),
            int(round(frameHeight / 150)),
            4,
        )
    return frameDlibHog, bboxes


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Face detection")
    parser.add_argument("--video", type=str, default="", help="Path to video file")
    args = parser.parse_args()

    source = args.video
    hogFaceDetector = dlib.get_frontal_face_detector()
    
    cap = cv2.VideoCapture(0)

    hasFrame, frame = cap.read()


    frame_count = 0
    tt_dlibHog = 0

    while True:
        hasFrame, frame = cap.read()
        if not hasFrame:
            break

        frame_count += 1
        t = time.time()

        outDlibHog, bboxes = detectFaceDlibHog(hogFaceDetector, frame)
        tt_dlibHog += time.time() - t
        fpsDlibHog = frame_count / tt_dlibHog

        label = "DLIB HoG; FPS : {:.2f}".format(fpsDlibHog)
        cv2.putText(
            outDlibHog,
            label,
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.3,
            (0, 0, 255),
            3,
            cv2.LINE_AA,
        )

        cv2.imshow("Face Detection Comparison", outDlibHog)


        if frame_count == 1:
            tt_dlibHog = 0

        k = cv2.waitKey(5)
        if k == 27:
            break

    cv2.destroyAllWindows()