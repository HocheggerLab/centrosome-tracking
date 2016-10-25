#!/opt/local/bin/python
import numpy as np
import cv2
import tifffile as tf
import javabridge
import bioformats
import xml.etree.ElementTree


def nothing(x):
    pass


class image_preprocessing():

    def __init__(self):
        self.dir = "/Users/Fabio/Documents/Trabajo/universidad/PHD Tesis/Computer Vision"
        self.file_pre = "Capture 9_XY1418831432_Z0_T"
        self.fileout_pre = "out_frame 9_XY1418831432_Z0_T"
        self.file_post = "_C2.tiff"

        self.filename = '%s/%s%02d%s' % (self.dir, self.file_pre, 0, self.file_post)

    def _save_images(self):
        if not hasattr(self, 'outImg'):
            return

        self._load_omexml()

        # save current frame to disk - a bit hacky
        filename_out = '%s/%s%02d%s' % (self.dir, self.fileout_pre, 0, self.file_post)

        with tf.TiffWriter(filename_out, software="Fabio Echegaray's python script", imagej=True) as tif:
            tifftgs = []
            for t in self.tags:
                t = self.tags[t]
                dtype = t.dtype
                if dtype=='1H': dtype='H'
                if dtype=='1I': dtype='I'
                if dtype=='1s': dtype='s'
                if dtype=='1f': dtype='f'
                tifftgs.append((t.code, dtype, t.count, t.value, True))

        # for oImg in self.outImg[1:]: # don't save first image (all zeros)
            tif.save(self.outImg[1:],
                     description=self.omexml,
                     resolution=(self.resolutionX, self.resolutionY),
                     contiguous=True,
                     extratags=tifftgs)

    def _load_omexml(self):
        try:
            javabridge.start_vm(class_path=bioformats.JARS)

            self.omexml = bioformats.get_omexml_metadata(self.filename).encode('ascii', 'ignore')
            e = xml.etree.ElementTree.fromstring(self.omexml)
            self.omexml.replace(self.file_pre, self.fileout_pre)

        finally:
            javabridge.kill_vm()

    def substract_ex(self):
        # create windows
        cv2.namedWindow('frame')
        cv2.namedWindow('output')
        cv2.namedWindow('sub')
        cv2.moveWindow('frame', 0, 0)
        cv2.moveWindow('sub', 500, 80)
        cv2.moveWindow('output', 0, 600)

        # create trackbars for color change
        cv2.createTrackbar('contrast','frame',60,100,nothing)
        cv2.createTrackbar('brightness','frame',1,100,nothing)
        cv2.createTrackbar('threshold','output',0,255,nothing)

        with tf.TiffFile(self.filename, fastij=True) as tif:
            imgs = tif.asarray()
            firstPage = tif.series[0].pages[2]
            self.tags = firstPage.tags
            img_prev = imgs[0][2] # FIXME: handle channel properly

            n_imgs = len(tif.pages)
            self.sizeT = tif.series[0].shape[0]
            self.sizeY = tif.series[0].shape[2]
            self.sizeX = tif.series[0].shape[3]
            self.sizeZ = tif.pages[0].image_depth
            channels = 1 #tif.series[0].shape[0]
            self.resolutionX = firstPage.x_resolution[0] / firstPage.x_resolution[1]
            self.resolutionY = firstPage.y_resolution[0] / firstPage.y_resolution[1]
            self.outImg = np.zeros((self.sizeT, channels, self.sizeZ, self.sizeX, self.sizeY), np.uint16)
            exit_flag = False
            while not exit_flag:
                for i in range(n_imgs):
                    # Load frame-by-frame images
                    img = imgs[i][2] # FIXME: handle channel properly

                    # adjust brightness and contrast
                    contrast = cv2.getTrackbarPos('contrast','frame')/10.0
                    brightness = cv2.getTrackbarPos('brightness','frame')/100.0
                    thresh = cv2.getTrackbarPos('threshold','output')

                    #substract two images
                    self.outImg[i][0][0] = cv2.addWeighted(img, contrast, img_prev, -contrast, brightness)
                    # self.outImg[i][0][0] = ((img_prev-img)*contrast+brightness).astype(np.uint16)
                    #adjust input image for viewing
                    img = (img*contrast + brightness).astype(np.uint16)

                    # Display the resulting frame
                    cv2.imshow('frame',img)
                    cv2.imshow('sub',self.outImg[i][0][0])
                    key = cv2.waitKey(1)
                    if key & 0xFF == ord('q'):
                        exit_flag = True
                        break
                    elif key & 0xFF == ord('r'):
                        break

        # When everything done, release the capture
        cv2.destroyAllWindows()
        self._save_images()


    def substract_img(self):
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

    def adjust_brightness(img, val):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # convert it to hsv

        h, s, v = cv2.split(hsv)
        v += val
        final_hsv = cv2.merge((h, s, v))

        out = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        return out

if __name__ == '__main__':
    imgObj = image_preprocessing()
    imgObj.substract_ex()