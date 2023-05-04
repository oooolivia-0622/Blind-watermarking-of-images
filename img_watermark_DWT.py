# Import the packages we use
import numpy as np
import cv2
import pywt
import random
import math
import cmath


# arnold permutation, k is the number of permutation
def arnold (img, key):
    r = img.shape[0]
    c = img.shape[1]
    p = np.zeros ( (r, c), np.uint8 )

    a = 1
    b = 1
    for k in range ( key ):
        for i in range ( r ):
            for j in range ( c ):
                x = (i + b * j) % r
                y = (a * i + (a * b + 1) * j) % c
                p[x, y] = img[i, j]
    return p


# Inverse Arnold replacement, k is the number of permutation
def deArnold (img, key):
    r = img.shape[0]
    c = img.shape[1]
    p = np.zeros ( (r, c), np.uint8 )

    a = 1
    b = 1
    for k in range ( key ):
        for i in range ( r ):
            for j in range ( c ):
                x = ((a * b + 1) * i - b * j) % r
                y = (-a * i + j) % c
                p[x, y] = img[i, j]
    return p


# Watermark embedding
def setwaterMark (waterTmg, Img, key):
    print ( 'Watermark embedding...' )
    Img = cv2.resize ( Img, (400, 400) )
    waterTmg = cv2.resize ( waterTmg, (201, 201) )
    # Grayscale processing of carrier image
    Img1 = cv2.cvtColor ( Img, cv2.COLOR_RGB2GRAY )
    waterTmg1 = cv2.cvtColor ( waterTmg, cv2.COLOR_RGB2GRAY )
    cv2.imshow ( 'original image', Img1 )
    cv2.imshow ( 'watermark image', waterTmg1 )
    cv2.waitKey ( 0 )
    # Arnold permutation is implemented on the watermark image
    waterTmg1 = arnold ( waterTmg1, key )

    # carrier image three-order wavelet transform
    c = pywt.wavedec2 ( Img1, 'db2', level = 3 )
    [cl, (cH3, cV3, cD3), (cH2, cV2, cD2), (cH1, cV1, cD1)] = c
    # watermark image first order wavelet transform
    waterTmg1 = cv2.resize ( waterTmg1, (101, 101) )
    d = pywt.wavedec2 ( waterTmg1, 'db2', level = 1 )
    [ca1, (ch1, cv1, cd1)] = d

    # Custom embedding coefficients
    a1 = 0.1
    a2 = 0.2
    a3 = 0.1
    a4 = 0.1
    # embedding
    cl = cl + ca1 * a1
    cH3 = cH3 + ch1 * a2
    cV3 = cV3 + cv1 * a3
    cD3 = cD3 + cd1 * a4

    # Image reconstruction
    newImg = pywt.waverec2 ( [cl, (cH3, cV3, cD3), (cH2, cV2, cD2), (cH1, cV1, cD1)], 'db2' )
    newImg = np.array ( newImg, np.uint8 )

    print ( 'Watermark embedding complete!' )

    cv2.imshow ( "carrier image", newImg )
    cv2.imwrite ( './after.bmp', newImg )
    cv2.waitKey ( 0 )


def getwaterMark (originalImage, Img, key):
    print ( 'Watermark Extraction...' )
    # Original image gray processing
    originalImage = cv2.resize ( originalImage, (400, 400) )

    Img1 = cv2.cvtColor ( originalImage, cv2.COLOR_RGB2GRAY )
    Img = cv2.cvtColor ( Img, cv2.COLOR_RGB2GRAY )

    #     cv2.imshow('original image',Img1)
    cv2.waitKey ( 0 )

    # carrier image three-order wavelet transform
    c = pywt.wavedec2 ( Img, 'db2', level = 3 )
    [cl, (cH3, cV3, cD3), (cH2, cV2, cD2), (cH1, cV1, cD1)] = c

    # Original image three-order wavelet transform
    d = pywt.wavedec2 ( Img1, 'db2', level = 3 )
    [dl, (dH3, dV3, dD3), (dH2, dV2, dD2), (dH1, dV1, dD1)] = d

    # The embedding algorithm (reverse)
    # Custom embedding coefficients
    a1 = 0.1
    a2 = 0.2
    a3 = 0.1
    a4 = 0.1

    ca1 = (cl - dl) * 10
    ch1 = (cH3 - dH3) * 5
    cv1 = (cV3 - dV3) * 10
    cd1 = (cD3 - dD3) * 10

    # Reconstruction of watermark image
    waterImg = pywt.waverec2 ( [ca1, (ch1, cv1, cd1)], 'db2' )
    waterImg = np.array ( waterImg, np.uint8 )

    cv2.imshow ( "Reconstruction of watermark image", waterImg )
    cv2.waitKey ( 0 )

    # Inverse Arnold permutation is performed on the extracted watermark image
    waterImg = deArnold ( waterImg, key )
    cv2.imshow ( "Reconstruction of watermark image", waterImg )
    cv2.waitKey ( 0 )
    print ( 'Watermark Extraction Complete！' )

    return waterImg

def rotate_about_center(src, angle, scale=1.):
    w = src.shape[1]
    h = src.shape[0]
    rangle = np.deg2rad(angle)  # angle in radians
    nw = (abs(np.sin(rangle) * h) + abs(np.cos(rangle) * w)) * scale
    nh = (abs(np.cos(rangle) * h) + abs(np.sin(rangle) * w)) * scale
    rot_mat = cv2.getRotationMatrix2D((nw * 0.5, nh * 0.5), angle, scale)
    rot_move = np.dot(rot_mat, np.array([(nw - w) * 0.5, (nh - h) * 0.5, 0]))
    rot_mat[0, 2] += rot_move[0]
    rot_mat[1, 2] += rot_move[1]
    return cv2.warpAffine(src, rot_mat, (int(math.ceil(nw)), int(math.ceil(nh))), flags=cv2.INTER_LANCZOS4)

def attack (fname, type):
    img = cv2.imread(fname)

    # Get the image before we attack the image
    if type == "ori":
        return img

    # Blur attack
    if type == "blur":
        kernel = np.ones ( (3, 3), np.float32 ) / 9
        return cv2.filter2D ( img, -1, kernel )

    # Rotating attack
    if type == "rotate90":
        return rotate_about_center ( img, 90 )

    # Noise attack
    if type == "saltnoise":
        for k in range ( 1000 ):
            i = int ( np.random.random () * img.shape[1] )
            j = int ( np.random.random () * img.shape[0] )
            if img.ndim == 2:
                img[j, i] = 255
            elif img.ndim == 3:
                img[j, i, 0] = 255
                img[j, i, 1] = 255
                img[j, i, 2] = 255
        return img

    # Graffiti attack
    if type == "randline":
        cv2.rectangle ( img, (384, 0), (510, 128), (0, 255, 0), 3 )
        cv2.rectangle ( img, (0, 0), (300, 128), (255, 0, 0), 3 )
        cv2.line ( img, (0, 0), (511, 511), (255, 0, 0), 5 )
        cv2.line ( img, (0, 511), (511, 0), (255, 0, 255), 5 )

        return img

    # Cropping attack
    if type == "cropping":
        cv2.circle ( img, (256, 256), 63, (255, 255, 255), -1 )
        font = cv2.FONT_HERSHEY_SIMPLEX
        return img
    return img

    # Brightness attack
    if type == "brighter10":
        w, h = img.shape[:2]
        for xi in range ( 0, w ):
            for xj in range ( 0, h ):
                img[xi, xj, 0] = int ( img[xi, xj, 0] * 10 )
                img[xi, xj, 1] = int ( img[xi, xj, 1] * 10 )
                img[xi, xj, 2] = int ( img[xi, xj, 2] * 10 )
        return img

attack_list = {}
attack_list['ori'] = 'Original image'
attack_list['blur'] = 'Blur attack'
attack_list['rotate90'] = 'Rotating attack'
attack_list['saltnoise'] = 'Noise attack'
attack_list['randline'] = 'Graffiti attack'
attack_list['cropping'] = 'Cropping attack'
attack_list['brighter10'] = 'Brightness attack'

if __name__ == '__main__':

    # read original image and watermark image
    waterImg = cv2.imread ( 'school_badge.jpg' )
    Img = cv2.imread ( 'Jisoo.jpg' )
    # watermark embedding
    setwaterMark ( waterImg, Img, 10 )

    #     # read original image and carrier image
    originalImage = cv2.imread ( 'Jisoo.jpg' )
    Img = cv2.imread ( './after.bmp' )
    # watermark extraction

    getwaterMark ( originalImage, Img, 10 )
    # image attack
    for k, v in attack_list.items ():
        wmd = attack ( 'after.bmp', k )
        cv2.imshow ( 'new', wmd )
        filename = "./output/{}.jpg".format ( k )
        cv2.imwrite ( filename, wmd )

        watermark = getwaterMark ( originalImage, wmd, 10 )
        wm_filename = "./output/watermark/{}.jpg".format ( k )
        cv2.imshow ( "extraction watermark image", watermark )
        cv2.imwrite ( wm_filename, watermark )

        cv2.waitKey ( 0 )
