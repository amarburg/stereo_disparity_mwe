#!/usr/bin/env python3

import argparse
from pathlib import Path
import logging
import time

import pprint as pp
import numpy as np
import cv2

import yaml

import elas

parser = argparse.ArgumentParser()

parser.add_argument('--output-dir,-o', dest='output', type=Path,
                    default=Path(__file__).absolute().parent / "output_elas",
                    help='Output directory')

parser.add_argument('--log', nargs='?',
                    default='INFO',
                    help='Logging level')

parser.add_argument("--left", type=Path,
                    default=Path(__file__).absolute().parent / "camera_info" / "18482016_water.yaml")

parser.add_argument("--right", type=Path,
                    default=Path(__file__).absolute().parent / "camera_info" / "18505502_water.yaml")

parser.add_argument("stereofiles", nargs="*")


args = parser.parse_args()
logging.basicConfig( level=args.log.upper() )

def mat_from_dict( d ):
    return np.reshape( d['data'], (d['rows'], d['cols'] ) )

def cam_from_yaml( y ):
    return { 'size': (y['image_width'], y['image_height']),
             'K': mat_from_dict( y['camera_matrix'] ),
             'dist': mat_from_dict( y['distortion_coefficients']),
             'projection': mat_from_dict( y['projection_matrix']),
             'rectification': mat_from_dict( y['rectification_matrix'])}

with open(args.left) as f:
    y = yaml.load(f, Loader=yaml.SafeLoader)
    leftCam = cam_from_yaml( y )

with open(args.right) as f:
    y = yaml.load(f, Loader=yaml.SafeLoader)
    rightCam = cam_from_yaml(y)

print("Left camera:")
pp.pprint(leftCam)
print()
print("Right camera:")
pp.pprint(rightCam)


def stereo_save( left, right, filename ):
    out = np.concatenate( (left, right), axis=1 )
    outColor = cv2.cvtColor( out, cv2.COLOR_GRAY2RGB )
    cv2.imwrite( str(filename), outColor )

maxdepth = 100

for infile in args.stereofiles:
    basename = Path(infile).name

    logging.info("Processing %s" % basename)

    imgStereo = cv2.imread(infile)

    left = imgStereo[:, 0:int(imgStereo.shape[1]/2) ]
    right = imgStereo[:, int(imgStereo.shape[1]/2):int(imgStereo.shape[1]) ]

    left = cv2.cvtColor(left, cv2.COLOR_RGB2GRAY)
    right = right[:,:,1]

    stereo_save( left, right, args.output / ("input_"+basename) )

    def remap( img, cam ):
        ## Extract newK from projection
        newK = cam['projection'][:,0:2]

        map1,map2 = cv2.initUndistortRectifyMap( cam['K'], cam['dist'], cam['rectification'], cam['projection'], cam['size'], cv2.CV_32FC1)
        return cv2.remap(img, map1, map2, cv2.INTER_LINEAR);

    imgLrect = remap( left, leftCam )
    imgRrect = remap( right, rightCam )

    ## Flip and swap
    if True:
        foo = cv2.rotate( imgRrect, cv2.ROTATE_180 )
        imgRrect = cv2.rotate( imgLrect, cv2.ROTATE_180 )
        imgLrect = foo

    stereo_save( imgLrect, imgRrect, args.output / ("rect_"+basename) )

    scale = 0.25
    newSize = (int(imgLrect.shape[1]*scale), int(imgLrect.shape[0]*scale) )

    imgLresized = cv2.resize(imgLrect, newSize )
    imgRresized = cv2.resize(imgRrect, newSize )

    imgLstereo = imgLresized
    imgRstereo = imgRresized

    d1 = np.empty_like(imgLstereo, dtype=np.float32)
    d2 = np.empty_like(imgRstereo, dtype=np.float32)

    params = elas.Elas_parameters()
    params.max_disparity = 64
    params.min_disparity = -10
    params.support_threshold = 0.85
    params.postprocess_only_left = True
    params.ipol_gap_width = 3
    params.add_corners = False
    params.match_texture = 1
    params.filter_median = False
    params.filter_adaptive_mean = True

    stereo = elas.Elas(params)

    start = time.time()
    stereo.process_stereo(imgLstereo, imgRstereo, d1, d2)
    end = time.time()
    print("Processing took %.3f sec" % (end - start) )

    d1 = np.fmax( d1, np.zeros_like(d1, dtype=np.float32) )

    # print(np.amax(d1))
    # print(np.amin(d1))

    maxdepth = max( maxdepth, np.amax(d1) )

    disparityL = (d1 / maxdepth * 255.0).astype(np.uint8)
    #disparityR = (d2 / np.amax(d2) * 255.0).astype(np.uint8)

    #stereo_save( disparityL, disparityR, args.output / ("disparity_"+basename) )

    cv2.imwrite( str(args.output / ("left_disparity_"+basename)), disparityL )

    ## Attempt background removal

#     right_matcher = cv2.ximgproc.createRightMatcher(stereo)
# #     right_matcher = cv2.StereoBM_create(numDisparities=stereo.getNumDisparities(), blockSize=stereo.getBlockSize())
#
#     ## These are set appropriately by createRightMatcher
#
#     #right_matcher.setTextureThreshold(stereo.getTextureThreshold())
#     right_matcher.setSpeckleWindowSize(stereo.getSpeckleWindowSize())
#     right_matcher.setSpeckleRange(stereo.getSpeckleRange())
#     dispR = right_matcher.compute(imgRstereo, imgLstereo)
#
#     ## Display right-to-left disparity
#     cv2.imwrite( str(args.output / ("right_disparity_"+basename)), dispR )
#
    # FILTER Parameters
    # lmbda = 10000
    # sigma = 1.0
    # #visual_multiplier = 1.0
    #
    # wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=stereo)
    # wls_filter.setLambda(lmbda)
    # wls_filter.setSigmaColor(sigma)
    # # wls_filter.setDepthDiscontinuityRadius(1)
    # #wls_filter.setDepthDiscontinuityRadius(4)
    #
    # filteredImg = wls_filter.filter(disparityL, imgLstereo, None, disparityR, None, imgRstereo )
    #
    # filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
    # filteredImg = np.uint8(filteredImg)
    #
    # cv2.imwrite( str(args.output / ("smoothed_"+basename)), filteredImg )
