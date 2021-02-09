from PIL import Image
import numpy as np
import math
import cv2
import timeit
from skimage.measure import compare_ssim

class PSNR:
    _BACKGROUND_VAL = 0 #greater than 0
    _FOREGROUND_VAL = 1
    
    _H_list = [0,  160,  320,  480,  640,  800,  960, 1120, 1280]
    _W_list = [0,  160,  320,  480,  640,  800,  960, 1120, 1280, 1440, 1600,
                   1760, 1920, 2080, 2240, 2400]
    _D = 160
    _topic_dict = {'1': 'skiing.mp4', '0': 'conan1.mp4', '3': 'conan2.mp4', '2': 'alien.mp4', '4': 'surfing.mp4', '7': 'football.mp4', '6': 'cooking.mp4', '8': 'rhinos.mp4'}
    _time_dict = {'1': (9, 33), '0': (9, 43), '3': (17, 43), '2': (72, 128), '4': (37, 73), '7': (78, 114), '6': (17, 43), '8': (37, 78)}
    
    _img0_template = './PSNR_frame_tiles/{}/{}_{}.jpg'
    _img1_template = './PSNR_frame_tiles/{}/{}_{}_{}_{}.jpg'
    _img_dat = {}
    
    def __init__(self):
        for topic in ['0', '1', '2', '3', '4', '6', '7', '8']:
            tb, te = self.get_range(topic)
            self._img_dat[topic] = {}
            for ts in range(tb, te):
                im0, im1 = self.get_imagepair(topic, ts)
                self._img_dat[topic][ts] = [im0, im1]
        return
    
    def get_vname(self, topic):
        return self._topic_dict[topic].replace('.mp4', '')
    
    def get_range(self, topic):
        return self._time_dict[topic]
    
    def load_image(self, _infilename ):
        img = Image.open(_infilename )
        img.load()
        data = np.asarray( img, dtype="int" ).astype(np.uint8)
        return data

    def stitch_image(self, _img0, _topic, _ts):
        #'./PSNR_frame_tiles/skiing/{}_{}_{}_{}.jpg'#fill in topic, topic, wi, hi, pt
        vname = self.get_vname(_topic)
        H, W, C = _img0.shape
        
        result = np.zeros(shape=(H, W, C), dtype=np.uint8)
        for wi in self._W_list:
            for hi in self._H_list:
                img_path = self._img1_template.format(vname, vname, wi, hi, _ts)
                img = self.load_image(img_path)
                result[hi:hi+self._D, wi:wi+self._D] = img
        return result

    
    def psnr(self, img1, img2):
        mse = np.mean((img1 - img2) ** 2)
        if mse == 0:
            return 100
        PIXEL_MAX = 255.0
        return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
    
    def ssim(self, img1, img2):
        return compare_ssim(img1, img2, img2.max() - img2.min(), multichannel=True)

    
    def get_imagepair(self, topic, ts):
        vname = self.get_vname(topic)
        ts = int(ts)
        img0_path = self._img0_template.format(vname, vname, ts)
        img0 = self.load_image(img0_path)
        img1 = self.stitch_image(img0, topic, ts)
        
        return img0, img1
    
    
    
    def create_mask(self, img0, tile_mask):
        H, W, C = img0.shape
        img_mask = np.zeros(img0.shape, dtype=np.uint8)
        for idx_w, wi in enumerate(self._W_list):
            for idx_h, hi in enumerate(self._H_list):
                if tile_mask[idx_h, idx_w] == 1:
                    img_mask[hi:hi+self._D, wi:wi+self._D] = np.zeros(shape=(self._D, self._D, C), dtype=np.uint8) + self._FOREGROUND_VAL
        return img_mask
    
    
    def extract_vpimage(self, in_img, vp_mask):
        H, W, C = in_img.shape
        vp_size = np.sqrt(vp_mask.sum()*1.0/C/self._FOREGROUND_VAL)
        if vp_size % 1 > 0:
            print 'ERROR: VIEWPORT_MASK is not a square'
            raise Exception
        vp_size = int(vp_size)
        viewport = in_img[vp_mask>0].reshape(vp_size, vp_size, 3)
        
        return viewport
    
    
    def viewport_perceived(self, topic, ts, vp_tilemap, pd_tilemap):
        import timeit
        
        #im0, im1 = self.get_imagepair(topic, ts)#return original & stitched image
        im0, im1 = self._img_dat[topic][int(ts)]

        prediction_mask = self.create_mask(im1, pd_tilemap)
        im2 = np.copy(im1)
        im2[prediction_mask==0] = self._BACKGROUND_VAL

        viewport_mask = self.create_mask(im1, vp_tilemap)
        viewport_full = self.extract_vpimage(im1, viewport_mask)
        viewport_pred = self.extract_vpimage(im2, viewport_mask)
        viewport0 = self.extract_vpimage(im0, viewport_mask)

        pr_pred = self.psnr(viewport_pred, viewport0)
        pr_full = self.psnr(viewport_full, viewport0)
        
        #try:
        sm_pred = compare_ssim(viewport0, viewport_pred,  multichannel=True)
        sm_full = compare_ssim(viewport0, viewport_full,  multichannel=True)
        #except:
        #    sm_pred = 0.0
        #    sm_full = 0.0

        
        #sm_pred = pr_pred
        #sm_full = pr_full

        return pr_pred, sm_pred, pr_full, sm_full, im0, im1, viewport_pred, viewport_full, viewport0
