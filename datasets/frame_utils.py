import numpy as np
from PIL import Image
from os.path import *
import re
import imageio
import cv2

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

TAG_CHAR = np.array([202021.25], np.float32)


def readFlow(fn):
    """Read .flo file in Middlebury format"""
    # Code adapted from:
    # http://stackoverflow.com/questions/28013200/reading-middlebury-flow-files-with-python-bytes-array-numpy

    # WARNING: this will work on little-endian architectures (eg Intel x86) only!
    # print 'fn = %s'%(fn)
    with open(fn, "rb") as f:
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            print("Magic number incorrect. Invalid .flo file")
            return None
        else:
            w = np.fromfile(f, np.int32, count=1)
            h = np.fromfile(f, np.int32, count=1)
            # print 'Reading %d x %d flo file\n' % (w, h)
            data = np.fromfile(f, np.float32, count=2 * int(w) * int(h))
            # Reshape data into 3D array (columns, rows, bands)
            # The reshape here is for visualization, the original code is (w,h,2)
            return np.resize(data, (int(h), int(w), 2))


def readPFM(file):
    file = open(file, "rb")

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    if header == b"PF":
        color = True
    elif header == b"Pf":
        color = False
    else:
        raise Exception("Not a PFM file.")

    dim_match = re.match(rb"^(\d+)\s(\d+)\s$", file.readline())
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception("Malformed PFM header.")

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = "<"
        scale = -scale
    else:
        endian = ">"  # big-endian

    data = np.fromfile(file, endian + "f")
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data


def readDispSintelStereo(file_name):
    """Return disparity read from filename."""
    f_in = np.array(Image.open(file_name))
    d_r = f_in[:, :, 0].astype("float64")
    d_g = f_in[:, :, 1].astype("float64")
    d_b = f_in[:, :, 2].astype("float64")

    disp = d_r * 4 + d_g / (2 ** 6) + d_b / (2 ** 14)
    mask = np.array(Image.open(file_name.replace("disparities", "occlusions")))
    valid = (mask == 0) & (disp > 0)
    return disp, valid


def readDispMiddlebury(file_name):
    assert basename(file_name) == "disp0GT.pfm"
    disp = readPFM(file_name).astype(np.float32)
    assert len(disp.shape) == 2
    nocc_pix = file_name.replace("disp0GT.pfm", "mask0nocc.png")
    assert exists(nocc_pix)
    nocc_pix = imageio.imread(nocc_pix) == 255
    assert np.any(nocc_pix)
    return disp, nocc_pix


def read_gen(file_name, pil=False):
    ext = splitext(file_name)[-1]
    if ext == ".png" or ext == ".jpeg" or ext == ".ppm" or ext == ".jpg":
        return Image.open(file_name)
    elif ext == ".bin" or ext == ".raw":
        return np.load(file_name)
    elif ext == ".flo":
        return readFlow(file_name).astype(np.float32)
    elif ext == ".pfm":
        flow = readPFM(file_name).astype(np.float32)
        if len(flow.shape) == 2:
            return flow
        else:
            return flow[:, :, :-1]
    return []
