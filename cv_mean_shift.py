# import sys
# sys.path.append('ObjectTracking')

import build_bin as bi
import numpy as np
import cv2
BIN_COUNTS = 130
FONT = cv2.FONT_HERSHEY_COMPLEX

# COLOR_LIST = getColorList()


def get_img_hist(img):
    """
    根据给定图片, 计算颜色概率分布图
    calculate color histogram of an image
    :param img:
    :return:

    :cv2.inRange():
        hsv指的是原图
        lower_red指的是图像中低于这个lower_red的值，图像值变为0
        upper_red指的是图像中高于这个upper_red的值，图像值变为0
        而在lower_red～upper_red之间的值变成255

    """
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # 过滤低亮度值和高亮度值(rgb), 返回过滤后的黑白二值图, 表示roi在白色区域
    mask = cv2.inRange(hsv_img, np.array((0., 60., 32.)), np.array((180., 255., 255.)))


    # kx = cv2.getGaussianKernel(img.shape[1], 3)  # 行数(个数, sigma)
    # ky = cv2.getGaussianKernel(img.shape[0], 3)  # 列数
    # kernel = np.matmul(kx, ky.T)


    # 提取绿色部分
    # mask = cv2.inRange(hsv_img, COLOR_LIST['green'][0], COLOR_LIST['green'][1])
    # mask = cv2.bitwise_not(mask)  # 取反色

    # cv2.imshow('mask', mask)
    # cv2.waitKey(0)
    # cv2.destroyWindow('mask')

    # 计算直方图, 只计算mask未过滤掉的部分的值

    img_hist = cv2.calcHist([hsv_img], [0], mask, [BIN_COUNTS], [0, BIN_COUNTS])

    print(img_hist)

    # 归一化
    cv2.normalize(img_hist, img_hist, 0, 256, cv2.NORM_MINMAX)

    return img_hist


def find_new_roi_window(roi_hist, roi_window, img):
    """
    找到与roi颜色出现频率相同的区域, 即查找目标位置(以颜色出现频率相同为标准)
    in img, find a place that has same color histogram as roiv
    :param roi_hist: color histogram of roi
    :param roi_window: coordinate of roi
    :param img: img of current frame
    :return:
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 会输出与输入图像大小相同的图像，每一个像素值代表了输入图像上对应点属于目标对象的概率，简言之，输出图像中像素值越高的点越可能代表想要查找的目标
    dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, BIN_COUNTS], 1)

    # apply meanshift to get the new location
    # stop do meanshift until reach one condition
    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 1000, 0.0001)

    # 利用meanshift将roi_window转移到目标位置
    ret, roi_window = cv2.meanShift(dst, roi_window, term_crit)



    return roi_window


def main(video_path):
    cap = cv2.VideoCapture(video_path)
    #cap = cv2.VideoCapture(0)
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
            #print('wh',x,y,w,h)
            #roi_hist = get_img_hist(roi)
            roi = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
            #print(roi)

            roi_hist = bi.get_hist(roi)
            roi_flag =1
            #print(roi_hist)

            cv2.imshow("roi_img", roi)
            cv2.waitKey(0)
            cv2.destroyWindow('roi_img')
            cv2.destroyWindow('ROI')

        # get next frame:
        #print(roi_flag)
        if roi_flag==1:
            #roi_window = find_new_roi_window(roi_hist, roi_window, roi)


            # get last center
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            #print('text,x,t',x,y)
            roi_next = img[y:y + h, x:x + w]
            #print('next',np.sum(roi_next))
            #roi_next = cv2.cvtColor(roi_next, cv2.COLOR_RGB2GRAY)
            tx,ty=bi.meanshift_step(roi_next, (x,y,w,h), roi_hist,img)
            x = tx
            y = ty
            # 将roi区域画出来
            # draw roi

            # if num_test == 100:
            #     cv2.destroyAllWindows()
            #     cap.release()
            #x, y, w, h = roi_window
            img = cv2.rectangle(img, (tx, ty), (tx+w, ty+h), 255, 2)
        #print(roi_flag)
        if roi_flag ==2:
            roi_flag = 1

        # 当前帧数
        # current frame index
        count += 1
        cv2.putText(img, str(count), (40, 40), FONT, 1, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.imshow('Video', img)

    cv2.destroyAllWindows()
    cap.release()



if __name__ == "__main__":
    video_path = 'data/video2.mp4'
    main(video_path)

    # img = cv2.imread('data/img.jpg')
    # hist = get_img_hist(img)
    # img_window =
    # find_new_roi_window(hist,img_window,img)
