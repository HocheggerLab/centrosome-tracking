#!/opt/local/bin/python
import numpy as np
import cv2


def adjust_brightness(img, val):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # convert it to hsv

    h, s, v = cv2.split(hsv)
    v += val
    final_hsv = cv2.merge((h, s, v))

    out = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return out

def nothing(x):
    pass



dir = "/Users/Fabio/Documents/Trabajo/universidad/PHD Tesis/Computer Vision/12-2-16"
file_pre = "Capture 5.Project Maximum Z_XY1455292007_Z0_T"
file_post = "_C0.tiff"


def substract_ex():
    # create window
    cv2.namedWindow('frame')
    cv2.namedWindow('output')
    cv2.namedWindow('sub')
    cv2.moveWindow('frame', 0, 0)
    cv2.moveWindow('sub', 500, 80)
    cv2.moveWindow('output', 0, 600)


    # create trackbars for color change
    cv2.createTrackbar('contrast','frame',60,100,nothing)
    cv2.createTrackbar('brightness','frame',10,100,nothing)
    cv2.createTrackbar('threshold','output',0,255,nothing)


    img_prev = cv2.imread('%s/Capture 5.Project Maximum Z_XY1455292007_Z0_T%03d_C0.tiff'%(dir,0), cv2.IMREAD_UNCHANGED)
    exit_flag = False
    while not exit_flag:
        for i in range(148):
            # Load frame-by-frame images
            filename = '%s/Capture 5.Project Maximum Z_XY1455292007_Z0_T%03d_C0.tiff'%(dir,i)
            # print 'reading %s'%filename
            img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)

            # adjust brightness and contrast
            contrast = cv2.getTrackbarPos('contrast','frame')/10.0
            brightness = cv2.getTrackbarPos('brightness','frame')/10.0
            thresh = cv2.getTrackbarPos('threshold','output')

            #substract two images
            # sub = cv2.subtract(img, img_prev)
            sub = cv2.addWeighted(img, contrast, img_prev, -contrast, brightness)
            sub_8 = np.uint8(sub)
            out = cv2.threshold(sub_8, thresh, 255, cv2.THRESH_TOZERO)

            # img_prev = img

            # Display the resulting frame
            cv2.imshow('frame',img)
            cv2.imshow('sub',sub)
            # cv2.imshow('output',out[1])
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                exit_flag = True
                break
            elif key & 0xFF == ord('r'):
                break

            # save current frame to disk
            filename_out = '%s/out/out_frame Z_XY1455292007_Z0_T%03d_C0.tiff'%(dir,i)
            cv2.imwrite(filename_out, sub)

    # When everything done, release the capture
    cv2.destroyAllWindows()


def substract_img():
    # create window
    cv2.namedWindow('frame')
    cv2.namedWindow('filter mask')
    cv2.namedWindow('background')
    cv2.namedWindow('sub')
    cv2.moveWindow('frame', 0, 0)
    cv2.moveWindow('background', 0, 600)
    cv2.moveWindow('filter mask', 500, 0)
    cv2.moveWindow('sub', 500, 600)


    # create trackbars for color change
    cv2.createTrackbar('contrast','frame',60,100,nothing)
    cv2.createTrackbar('brightness','frame',10,100,nothing)
    # create trackbars for filter
    cv2.createTrackbar('learning rate','filter mask',50,100,nothing)
    cv2.createTrackbar('history','filter mask',50,100,nothing)
    # create trackbars for background
    cv2.createTrackbar('pixel size','background',1,10,nothing)

    conv16to8 = float(np.iinfo(np.int8).max)/float(np.iinfo(np.int16).max)

    exit_flag = False
    histMOG = 100
    while not exit_flag:
        fgbg = cv2.createBackgroundSubtractorMOG2(history=histMOG, varThreshold=1, detectShadows=False)
        for i in range(1, 148):
            # Load frame-by-frame images
            filename = '%s/Capture 5.Project Maximum Z_XY1455292007_Z0_T%03d_C0.tiff'%(dir,i)
            # print 'reading %s'%filename
            img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)

            # adjust brightness and contrast
            contrast = cv2.getTrackbarPos('contrast','frame')/10.0
            brightness = cv2.getTrackbarPos('brightness','frame')

            lrate = cv2.getTrackbarPos('learning rate','filter mask')/100.0
            histMOG = cv2.getTrackbarPos('history','filter mask') * 100

            pixelSize = cv2.getTrackbarPos('pixel size', 'background')
            pixelSize = pixelSize if pixelSize>0 else 1

            #substract two images
            img = cv2.scaleAdd(img, contrast, img*0+brightness)
            mask = fgbg.apply(img, learningRate = lrate)
            back = fgbg.getBackgroundImage()
            # apply opening morphological op to background to remove small pixels (noise)
            kernel = np.ones((pixelSize, pixelSize), np.uint8)
            back = cv2.morphologyEx(back, cv2.MORPH_OPEN, kernel)
            img_8 = img * conv16to8
            img_8 = np.uint8(img_8)
            sub = cv2.subtract(img_8, back)
            # sub = cv2.scaleAdd(sub, contrast, sub*0+brightness)


            # Display the resulting frame
            cv2.imshow('frame', img_8)
            cv2.imshow('filter mask', mask)
            cv2.imshow('sub', sub)
            cv2.imshow('background', back)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                exit_flag = True
                break
            elif key & 0xFF == ord('r'):
                break

    # When everything done, release the capture
    cv2.destroyAllWindows()

# substract_img()
substract_ex()