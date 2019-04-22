# import sys
# sys.path.append('ObjectTracking')

import build_bin as bi
import numpy as np
import cv2
FONT = cv2.FONT_HERSHEY_COMPLEX
def main(video_path):
    cap = cv2.VideoCapture(video_path)
    count = 0
    Total_Frame_Number = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    roi_flag = 0
    roi_flag_t = 0
    while count < Total_Frame_Number:
        #print("count",count)
        ret, img = cap.read(10)
        k = cv2.waitKey(10)
        if k == ord('q'):
            break
        if k == 32:  # pause when press SPACE
            cv2.waitKey(0)
        if k == ord('a'):
            # set up the ROI for tracking when press A
            roi_window = cv2.selectROI('ROI', img, fromCenter=False)

            #try rgb

            x, y, w, h = roi_window
            tx =x
            ty = y
            #print('first_center',x-w/2,y+h/2)
            roi = img[y:y + h, x:x + w]
            roi_hist = bi.color_hist(roi)
            roi_flag =1
            cv2.imshow("roi_img", roi)
            cv2.waitKey(0)
            cv2.destroyWindow('roi_img')
            cv2.destroyWindow('ROI')

        # get next frame:
        #print(roi_flag)
        if roi_flag==1:
            #roi_window = find_new_roi_window(roi_hist, roi_window, roi)
            x = tx
            y = ty
            roi_next = img[y:y + h, x:x + w]
            tx,ty=bi.meanshift_step(roi_next, roi_window, roi_hist,img)
            # 将roi区域画出来
            # draw roi
            img = cv2.rectangle(img, (tx, ty), (tx+w, ty+h), 255, 2)
        #print(roi_flag)
        if roi_flag ==2:
            roi_flag = 1

        # current frame index
        count += 1
        cv2.putText(img, str(count), (40, 40), FONT, 1, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.imshow('Video', img)

    cv2.destroyAllWindows()
    cap.release()



if __name__ == "__main__":
    video_path = '../data/video2.mp4'
    main(video_path)

    # img = cv2.imread('data/img.jpg')
    # hist = get_img_hist(img)
    # img_window =
    # find_new_roi_window(hist,img_window,img)
