import cv2
import numpy as np

# DDPI functions
def ddpi_templatematch(imraw, im, P1ROI, P4ROI, sz,
    P1THRESH, GBLUR, ds, P1WIN, P4WIN, P4TEMP, MASK):

    # grayscale
    imraw = cv2.cvtColor(imraw, cv2.COLOR_BGR2GRAY)

    # mask out edges
    im[MASK[1]:MASK[3],MASK[0]:MASK[2]] = imraw[MASK[1]:MASK[3],MASK[0]:MASK[2]]

    # gaussian blur
    blurred = cv2.GaussianBlur(im, (GBLUR, GBLUR), 0)

    # threshold to remove noise (maybe not necessary)
    ret, blurred2 = cv2.threshold(blurred, 50, 255,cv2.THRESH_TOZERO)

    # DOWNSAMPLE
    dsize = [sz[1]//ds, sz[0]//ds]
    imsm = cv2.resize(blurred2, dsize)

    # TEMPLATE MATCHING FOR P4
    gaussian_kernel = cv2.getGaussianKernel(P4TEMP, P4TEMP/2) 
    kernel_2D = gaussian_kernel @ gaussian_kernel.transpose()
    kernel_Box = np.ones((P4TEMP, P4TEMP))/P4TEMP**2
    imgauss = cv2.filter2D(imsm, -1, kernel_2D)
    imbox = cv2.filter2D(imsm, -1, kernel_Box)

    result = cv2.matchTemplate(imsm, kernel_2D, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    I = imgauss/(imbox**2 + 1e-3)
    ctr = np.where(np.max(I)==I)

    p4y = ctr[0][0]*ds
    p4x = ctr[1][0]*ds
    P4ROI[:,:] = blurred[p4y-P4WIN:p4y+P4WIN, p4x-P4WIN:p4x+P4WIN]

    # THRESHOLD FOR P1
    ret, thresh = cv2.threshold(imsm, P1THRESH, 255, cv2.THRESH_BINARY)

    M = cv2.moments(thresh)
    tol = 1e-8
    p1x = int(M["m10"] / (M["m00"] + tol))*ds
    p1y = int(M["m01"] / (M["m00"] + tol))*ds
    P1ROI[:,:] = blurred[p1y-P1WIN:p1y+P1WIN, p1x-P1WIN:p1x+P1WIN]

    M = cv2.moments(P1ROI)
    P1X = M["m10"] / (M["m00"] + tol)
    P1Y = M["m01"] / (M["m00"] + tol)

    P4ROI = P4ROI**3
    M = cv2.moments(P4ROI)
    P4X = M["m10"] / (M["m00"] + tol)
    P4Y = M["m01"] / (M["m00"] + tol)

    return P1X+p1x-P1WIN, P1Y+p1y-P1WIN, P4X+p4x-P4WIN, P4Y+p4y-P4WIN

def ddpi_template(im, P1ROI, P4ROI, sz,
    P1THRESH, ds, P1WIN, P4WIN, P4TEMP):
    # DOWNSAMPLE
    dsize = [sz[1]//ds, sz[0]//ds]
    imsm = cv2.resize(im, dsize)

    # TEMPLATE MATCHING FOR P4
    gaussian_kernel = cv2.getGaussianKernel(P4TEMP, P4TEMP/2) 
    kernel_2D = gaussian_kernel @ gaussian_kernel.transpose()
    kernel_Box = np.ones((P4TEMP, P4TEMP))/P4TEMP**2
    imgauss = cv2.filter2D(imsm, -1, kernel_2D)
    imbox = cv2.filter2D(imsm, -1, kernel_Box)

    I = imgauss/(imbox + 1e-3)
    ctr = np.where(np.max(I)==I)

    p4y = ctr[0][0]*ds
    p4x = ctr[1][0]*ds
    P4ROI[:,:] = im[p4y-P4WIN:p4y+P4WIN, p4x-P4WIN:p4x+P4WIN]

    # THRESHOLD FOR P1
    ret, thresh = cv2.threshold(imsm, P1THRESH, 255, cv2.THRESH_BINARY)

    M = cv2.moments(thresh)
    tol = 1e-8
    p1x = int(M["m10"] / (M["m00"] + tol))*ds
    p1y = int(M["m01"] / (M["m00"] + tol))*ds
    P1ROI[:,:] = im[p1y-P1WIN:p1y+P1WIN, p1x-P1WIN:p1x+P1WIN]

    M = cv2.moments(P1ROI)
    P1X = M["m10"] / (M["m00"] + tol)
    P1Y = M["m01"] / (M["m00"] + tol)

    P4ROI = P4ROI**3
    M = cv2.moments(P4ROI)
    P4X = M["m10"] / (M["m00"] + tol)
    P4Y = M["m01"] / (M["m00"] + tol)

    return P1X+p1x-P1WIN, P1Y+p1y-P1WIN, P4X+p4x-P4WIN, P4Y+p4y-P4WIN

def ddpi_blob(im, P1ROI, P4ROI, sz,
    PTHRESH, ds, P1WIN, P4WIN):
    # use blob analysis to find P1 and P4 (this is not parallelizable and will not be fast enough)

    # DOWNSAMPLE
    dsize = [sz[1]//ds, sz[0]//ds]
    imsm = cv2.resize(im, dsize)
    _, thresh = cv2.threshold(imsm, PTHRESH, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, 1, 2)

    areas = [cv2.contourArea(c) for c in contours]
    inds = np.argsort(areas)

    M = cv2.moments(contours[inds[-1]])
    tol = 1e-8
    p1x = int(M["m10"] / (M["m00"] + tol))*ds
    p1y = int(M["m01"] / (M["m00"] + tol))*ds
    P1ROI[:,:] = im[p1y-P1WIN:p1y+P1WIN, p1x-P1WIN:p1x+P1WIN]

    M = cv2.moments(contours[inds[-2]])
    p4x = int(M["m10"] / (M["m00"] + tol))*ds
    p4y = int(M["m01"] / (M["m00"] + tol))*ds
    P4ROI[:,:] = im[p4y-P4WIN:p4y+P4WIN, p4x-P4WIN:p4x+P4WIN]

    M = cv2.moments(P1ROI)
    P1X = M["m10"] / (M["m00"] + tol)
    P1Y = M["m01"] / (M["m00"] + tol)

    P4ROI = P4ROI**3
    M = cv2.moments(P4ROI)
    P4X = M["m10"] / (M["m00"] + tol)
    P4Y = M["m01"] / (M["m00"] + tol)

    # P4X, P4Y = radialcenter(P4ROI)

    return P1X+p1x-P1WIN, P1Y+p1y-P1WIN, P4X+p4x-P4WIN, P4Y+p4y-P4WIN


def mask_pupil(blurred, PTHRESH):

    # not parallelized 

    # Find pupil
    _, thresh = cv2.threshold(blurred, PTHRESH, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(thresh, 1,2)

    # Find the index of the largest contour
    areas = [cv2.contourArea(c) for c in contours]
    max_index = np.argmax(areas)
    cnt=contours[max_index]

    ellipse = cv2.fitEllipse(cnt)
    mask = np.zeros(blurred.shape, np.uint8)
    mask = cv2.ellipse(mask,ellipse,(255,255,255),-1)
    blurred = cv2.bitwise_and(blurred, blurred, mask=mask)
    return blurred, ellipse

def ddpi_pupil(imraw, P1ROI, P4ROI, sz,
    PTHRESH, GBLUR, ds, P1WIN, P4WIN):

    # grayscale
    imgray = cv2.cvtColor(imraw, cv2.COLOR_BGR2GRAY)

    # gaussian blur
    blurred = cv2.GaussianBlur(imgray, (GBLUR, GBLUR), 0)

    blurred, pellipse = mask_pupil(blurred, 16)

    P1X, P1Y, P4X, P4Y = ddpi_blob(blurred, P1ROI, P4ROI, sz,
    PTHRESH, ds, P1WIN, P4WIN)

    return P1X, P1Y, P4X, P4Y

def ddpi_thresh(im, P1ROI, P4ROI, sz,
    P1THRESH, ds, P1WIN, P4WIN):

    # use thresholding to find P1 (brightest spot) and P4 (second brightest spot) after masking the pupil or masking a region
    # this is the fastest algorithm and is parallelizable, but it is not robust -- means the experimental apparatus has to be dialed in
    _, im = cv2.threshold(im, P1THRESH//2, 255, cv2.THRESH_TOZERO)

    # DOWNSAMPLE
    dsize = [sz[1]//ds, sz[0]//ds]
    imsm = cv2.resize(im, dsize, interpolation=cv2.INTER_AREA)

    # THRESHOLD FOR P1 (P1 is saturated, so find the peak lockation)
    ret, thresh = cv2.threshold(imsm, P1THRESH, 255, cv2.THRESH_BINARY)

    M = cv2.moments(thresh) # use moments to find the center of mass of P1
    tol = 1e-8
    p1x_ = (M["m10"] / (M["m00"] + tol))
    p1xr = p1x_ % 1
    p1x_ = int(p1x_)
    p1x = p1x_*ds

    p1y_ = (M["m01"] / (M["m00"] + tol))
    p1yr = p1x_ % 1
    p1y_ = int(p1y_)
    p1y = p1y_*ds
    P1ROI[:,:] = im[p1y-P1WIN:p1y+P1WIN, p1x-P1WIN:p1x+P1WIN]
    M = cv2.moments(P1ROI)
    P1X = M["m10"] / (M["m00"] + tol)
    P1Y = M["m01"] / (M["m00"] + tol)

    p1x = int(P1X+p1x-P1WIN)
    p1y = int(P1Y+p1y-P1WIN)

    P1ROI[:,:] = im[p1y-P1WIN:p1y+P1WIN, p1x-P1WIN:p1x+P1WIN]
    M = cv2.moments(P1ROI)
    P1X = M["m10"] / (M["m00"] + tol)
    P1Y = M["m01"] / (M["m00"] + tol)

    p1x = P1X+p1x-P1WIN+p1xr
    p1y = P1Y+p1y-P1WIN+p1yr

    # Get P4 ROI (second brightest spot)

    # block out P1 in downsampled image
    P1WIN_ = P1WIN//ds
    imsm[p1y_-P1WIN_:p1y_+P1WIN_, p1x_-P1WIN_:p1x_+P1WIN_] = 0

    # find max location == P4 ROI center
    _,_,_,maxloc = cv2.minMaxLoc(imsm)

    p4x = maxloc[0]*ds
    p4y = maxloc[1]*ds

    P4ROI[:,:] = im[p4y-P4WIN:p4y+P4WIN, p4x-P4WIN:p4x+P4WIN]

    # P4ROI = P4ROI**2 # raise to power to get a softmax function
    M = cv2.moments(P4ROI)
    P4X = M["m10"] / (M["m00"] + tol)
    P4Y = M["m01"] / (M["m00"] + tol)

    p4x = int(P4X+p4x-P4WIN)
    p4y = int(P4Y+p4y-P4WIN)

    P4ROI[:,:] = im[p4y-P4WIN:p4y+P4WIN, p4x-P4WIN:p4x+P4WIN]
    
    # P4ROI = P4ROI**3 # raise to power to get a softmax function
    M = cv2.moments(P4ROI)
    P4X = M["m10"] / (M["m00"] + tol)
    P4Y = M["m01"] / (M["m00"] + tol)
    
    p4x = P4X+p4x-P4WIN
    p4y = P4Y+p4y-P4WIN
    # P4X, P4Y = radialcenter(P4ROI)

    return p1x, p1y, p4x, p4y

def ddpi_pupil_thresh(imraw, P1ROI, P4ROI, sz,
    PTHRESH, GBLUR, ds, P1WIN, P4WIN):

    # grayscale
    imgray = cv2.cvtColor(imraw, cv2.COLOR_BGR2GRAY)

    # gaussian blur
    blurred = cv2.GaussianBlur(imgray, (GBLUR, GBLUR), 0)

    blurred, pellipse = mask_pupil(blurred, 16)

    P1X, P1Y, P4X, P4Y = ddpi_thresh(blurred, P1ROI, P4ROI, sz,
        PTHRESH, ds, P1WIN, P4WIN)

    return P1X, P1Y, P4X, P4Y

def radialcenter(I):
    # finds the radial symmetric center of a blob (alternate to center of mass... can handle skewed blobs)
    # RSC just smooths,calculates the gradients and finds where they intersect

    Ny,Nx = I.shape

    # create index into x - this is optimized for speed in matlab, it's unlikely it worked out in the translation
    xm_onecol = np.arange(-(Nx-1)/2.0+0.5,(Nx-1)/2.0+0.5)  # Note that y increases "downward"
    xm = np.outer(np.ones( (1,Ny-1)),xm_onecol.T)
    
    # do the same for y
    ym_onerow = np.arange(-(Ny-1)/2.0+0.5,(Ny-1)/2.0+0.5)
    ym = np.outer(ym_onerow, np.ones((Nx-1, 1)))

    # Calculate derivatives along 45-degree shifted coordinates (u and v)
    # Note that y increases "downward" (increasing row number) -- we'll deal
    # with this when calculating "m" below.
    dIdu = I[0:Ny-1, 0:Nx-1] - I[1:Ny, 1:Nx]
    dIdv = I[0:Ny-1, 1:Nx] - I[1:Ny, 0:Nx-1]

    GBLUR = 5
    fdu = cv2.GaussianBlur(dIdu, (GBLUR, GBLUR), 0)
    fdv = cv2.GaussianBlur(dIdv, (GBLUR, GBLUR), 0)
    
    # h = np.ones((3,3))/9 # simple 3x3 averaging
    # fdu = cv2.filter2D(dIdu, -1, h)
    # fdv = cv2.filter2D(dIdv, -1, h)

    dImag2 = fdu*fdu + fdv*fdv # gradient magnitude, squared

    # Slope of the gradient .  Note that we need a 45 degree rotation of
    # the u,v components to express the slope in the x-y coordinate system.
    # The negative sign "flips" the array to account for y increasing
    # "downward"
    m = -(fdv + fdu) / (fdu-fdv)

    # handle *rare* edge cases 
    # *Very* rarely, m might be NaN if (fdv + fdu) and (fdv - fdu) are both
    # zero.  In this case, replace with the un-smoothed gradient.
    NNanm = np.sum(np.isnan(m.flatten()))
    if NNanm > 0:
        unsmoothm = (dIdv + dIdu) / (dIdu-dIdv)
        m[np.isnan(m)]=unsmoothm[np.isnan(m)]

    # If it's still NaN, replace with zero. (Very unlikely.)
    NNanm = np.sum(np.isnan(m.flatten()))
    if NNanm > 0:
        m[np.isnan(m)]=0
    
    m[np.abs(m)>10]=0
    
    # Shorthand "b", which also happens to be the
    # y intercept of the line of slope m that goes through each grid midpoint
    b = ym - m*xm

    # Weighting: weight by square of gradient magnitude and inverse
    # distance to gradient intensity centroid.
    sdI2 = np.sum(dImag2.flatten())

    # approximate centroid
    xcentroid = np.sum(np.sum(dImag2*xm))/sdI2
    ycentroid = np.sum(np.sum(dImag2*ym))/sdI2

    # weights
    w  = dImag2/np.sqrt((xm-xcentroid)*(xm-xcentroid)+(ym-ycentroid)*(ym-ycentroid))
    
    # least-squares minimization to determine the translated coordinate
    # system origin (xc, yc) such that lines y = mx+b have
    # the minimal total distance^2 to the origin:
    wm2p1 = w/(m*m+1)
    sw = np.sum(np.sum(wm2p1))
    smmw = np.sum(np.sum(m*m*wm2p1))
    smw = np.sum(np.sum(m*wm2p1))
    smbw = np.sum(np.sum(m*b*wm2p1))
    sbw = np.sum(np.sum(b*wm2p1))
    det = smw*smw - smmw*sw
    xc = (smbw*sw - smw*sbw)/det # relative to image center
    yc = (smbw*smw - smmw*sbw)/det # relative to image center

    # Return output relative to upper left coordinate
    xc = xc + (Nx-1)/2.0
    yc = yc + (Ny-1)/2.0

    return xc, yc