import numpy as np
import cv2

# divide 256 gray scale into 64 bin from 0-63

def gray_64bin():
    bin = []
    count = 0

    for i in range(0,64):
        while(count<4):
            bin.append(i)

            count = count+1
        count = 0

    #print(bin)
    return bin
def gray_32bin():
    bin = []
    count = 0
    for i in range(0,32):
        while(count<8):
            bin.append(i)
            count = count+1
        count = 0
    return bin
def gray_16bin():
    bin = []
    count = 0
    for i in range(0,16):
        while(count<16):
            bin.append(i)
            count = count+1
        count = 0
    return bin
#print(gray_64bin())

def gray_bin():
    bin = []
    count = 0
    bin_num = 0
    for i in range(0,256):

        bin.append(i)
        #count = count+1
        #count = 0
        bin_num+=1
    #print(bin)
    return bin

def RGB_empty_hist():
    hist = []
    for i in range(0,4096):
        hist.append(0)
    #print(len(hist))
    return hist

#use rgb to get hist
def color_hist(img):
    h = img.shape[0]
    w = img.shape[1]
    hist = RGB_empty_hist()
    band = pow(1, 2) + pow(1, 2)
    wei_c = []
    for i in range(0,h):
        for j in range(0,w):
            qr = img[i][j][0]/16
            qg = img[i][j][1]/16
            qb = img[i][j][2]/16
            q_temp = qr*239+qg*16+qb
            #print(q_temp)
            q_temp = np.around(q_temp).astype(int)
            #print(i)

            dist = pow(i - 1, 2) + pow(j - 1, 2)
            wei = 1 - dist / band
            wei_c.append(wei)
            hist[q_temp]=hist[q_temp]+wei
    C = sum(wei_c)
    if C == 0:
        C = 1
    hist = [c_bin / C for c_bin in hist]
    return hist
#print(gray_bin())

def empty_hist():
    hist = []
    for i in range(0,16):
        hist.append(0)
    #print(len(hist))
    return hist

def get_hist(img):
    h = img.shape[0]
    w = img.shape[1]
    #print("wh in hist",w,h)

    bin = gray_16bin();
    hist = empty_hist()
    c_x = w/2
    c_y = h/2
    wei_c = []
    band = pow(c_x,2)+ pow(c_y,2)

    for col in range(0,h):
        for row in range(0,w):
            color = img[col][row]
            #print(color)
            color_bin = bin[color]
            #print(color_bin)
            dist = pow(col-c_y,2)+pow(row-c_x,2)
            wei = 1-dist/band
            wei_c.append(wei)
            hist[color_bin] = hist[color_bin] + wei
    C = sum(wei_c)
    #print('c',C)
    #normalize hist
    hist=[c_bin / C for c_bin in hist]
    #print(len(hist))
    return hist

def get_similarity(hist1,hist2):
    similar = []
    for i in (range(0,4096)):
        if hist2[i] != 0:
            temp = hist1[i]/hist2[i]
            simi = np.sqrt(temp)

            similar.append(simi)
        else :
            similar.append(0)
    #print(similar)
    return similar

def meanshift_step(roi,roi_window,hist1,img):
    # 1 calculate h2
    # 2 calculate similarity
    # 3 ca/culate new center
    box_cx, box_cy, box_w, box_h = roi_window
    #print(box_w,box_h)
    #len = len(roi)

    len = box_h*box_w

    #print('roiwindow',roi_window)
    num = 0
    sim = []
    # box_cx = box.x
    # box_cy = box.y
    # box_h = box.shape[0]
    # box_w = box.shape[1]
    #x_shift = 0
    #y_shift = 0
    sum_w = 0
    # calcuate the hist2

    # caculate 2 simularity
   # similarity = get_similarity(hist1,hist2)
    #print("simi",similarity)
    # caculate new center
    while (num < 50):
        #print(num)
        x_shift = 0
        y_shift = 0
        sum_w = 0


        #print('ce?',box_cx,box_cy)

        hist2 = color_hist(roi)
        #bin = gray_16bin()

        similarity = get_similarity(hist1, hist2)
        s_mean=np.mean(similarity)
        sim.append(s_mean)
        print("simi", s_mean)
        num = num+1
        countt = 0
        #print('roi_error',roi.shape)
        for col in range(0, box_h):
            for row in range(0,box_w):

                #color meanshift
                qr = img[col][row][0] / 16
                qg = img[col][row][1] / 16
                qb = img[col][row][2] / 16
                q_temp = qr * 239 + qg * 16 + qb
                q_temp = np.around(q_temp).astype(int)

                #gray meanshift
                #color = roi[col][row]

                #color_bin = bin[color]
                sum_w = sum_w + similarity[q_temp]
                #print(sum_w)
                #print('change',box_cx,box_cy)
                #print(row,col)
                # version 2

                #x_shift = row*similarity[color_bin]+x_shift
                #print("loop of x_shift",x_shift)
                #y_shift = col*similarity[color_bin]+y_shift

                #version 1
                #gray_shift
                # y_shift = y_shift + similarity[color_bin]*(col-box_h/2)
                # x_shift = x_shift + similarity[color_bin]*(row-box_w/2)
                y_shift = y_shift + similarity[q_temp] * (col - box_h / 2)
                x_shift = x_shift + similarity[q_temp] * (row - box_w / 2)

                #print("step shift",x_shift,y_shift)

        #print('before nol',x_shift,y_shift)
        #shift distance

        #shift version 1
        #print(sum_w)
        if sum_w == 0:
            sum_w = 1
        y_shift = y_shift/sum_w
        x_shift = x_shift/sum_w

        #print("firstshift", x_shift, y_shift)

        #print('beforeshift',box_cx,box_cy)
        #new center version 1

        #
        box_cx = box_cx + x_shift
        box_cy = box_cy + y_shift

        # box_cx = x_shift/len
        # box_cy = y_shift/len

        #print('aftershift',box_cx,box_cy)
        # left top
        # box_cx = box_cx-box_w/2
        # box_cy = box_cy+box_h/2
        # print('left top',box_cx,box_cy)
        box_cx = np.around(box_cx)
        box_cx = box_cx.astype(int)
        box_cy = np.around(box_cy)
        box_cy = box_cy.astype(int)
        #print('centerx',box_cx,box_cy)

        #test change x and y
        roi = img[box_cy:box_cy + box_h, box_cx:box_cx + box_w]

        #show

        # l=np.around(box_cx + box_w / 2 - 3)
        # t=np.around(box_cy - box_h / 2 + 3)
        # r=np.around(box_cx + box_w / 2 +3)
        # b=np.around(box_cy - box_h / 2 - 3)
        # #tracking way
        # print((l,t))
        # im = cv2.rectangle(img, (box_cx, box_cy), (box_cx + box_w, box_cy + box_h), 255, 2)
        # #im = cv2.rectangle(img, (l,t), (r,b), 255)
        #
        # cv2.imshow('Video', im)
        # cv2.waitKey(0)

        #roi = img[box_cx:box_cx + box_h, box_cy:box_cy + box_w]
        #print(roi.shape)

        #print(num)


   # print("final",box_cx,box_cy)

    return box_cx,box_cy