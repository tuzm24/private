import os
import random
import struct
from CIF.operation.opt import *

SUPPORTED_EXTENSIONS = ('.bin')


def read_dataset(rootdir):
    """Returns a list of all image files in the given directory"""

    matches = []
    for root, dirnames, filenames in os.walk(rootdir):
        for filename in filenames:
            if filename.endswith(SUPPORTED_EXTENSIONS):
                matches.append(os.path.join(root, filename))

    return matches


def getFileList(dir, sort=True):
    if not os.path.isdir(dir):
        assert False
    fList = read_dataset(dir)
    if sort:
        random.shuffle(fList)
    return fList

def RawImageUnpack(imgPath):
    img = open(imgPath, 'rb')
    qp = struct.unpack('B', img.read(1))[0]
    intraMode = struct.unpack('B', img.read(1))[0]
    width = struct.unpack('<h', img.read(2))[0]
    height = struct.unpack('<h', img.read(2))[0]
    #with padding
    Owxh = width*height
    Owxh2 = (int)((width*height)/2)
    strOwxh = '<' + str(Owxh) + 'h'
    UVstrOwxh = '<'+str(Owxh2)+'h'
 #   print(width, height)
    width = width +4
    height = height+4

    Cwidth = (int)(width/2)
    Cheight = (int)(height/2)
    wxh = width*height
    strwxh = '<'+str(wxh)+'h'
    wxh2 = (int)((width*height)/2)
    UVstrwxh = '<'+str(wxh2)+'h'

    # Original is no padding
    orgY = np.array(struct.unpack(strOwxh, img.read(2*Owxh))).reshape((height-4, width-4,1))
    orgUV = np.array(struct.unpack(UVstrOwxh, img.read(Owxh))).reshape((2, Cheight-2, Cwidth-2)).transpose((1,2,0))

    predY = np.array(struct.unpack(strwxh, img.read(2*wxh))).reshape((height, width,1))
    predUV = np.array(struct.unpack(UVstrwxh, img.read(wxh))).reshape((2, Cheight, Cwidth)).transpose((1,2,0))
    reconY = np.array(struct.unpack(strwxh, img.read(2*wxh))).reshape((height, width,1))
    reconUV = np.array(struct.unpack(UVstrwxh, img.read(wxh))).reshape((2, Cheight, Cwidth)).transpose((1,2,0))
    UnfilteredY = np.array(struct.unpack(strwxh, img.read(2*wxh))).reshape((height, width,1))
    UnfilteredUV = np.array(struct.unpack(UVstrwxh, img.read(wxh))).reshape((2, Cheight, Cwidth)).transpose((1,2,0))
    ReconResY = reconY - predY
    ReconResUV = reconUV - predUV
    UnfilteredResY = UnfilteredY - predY
    UnfilteredResUV = UnfilteredUV - predUV
    OrgResY = orgY - predY[2:height-2, 2:width-2, :]
    OrgResUV = orgUV - predUV[1:Cheight-1,1:Cwidth-1, :]
    NoPadReconResY = ReconResY[2:height-2, 2:width-2, :]
    NoPadReconResUV = ReconResUV[1:Cheight-1,1:Cwidth-1, :]

    return OrgResY, OrgResUV, UnfilteredResY, UnfilteredResUV, predY, predUV, width-4, height-4, (int)(width/2-2), (int)(height/2-2), width, height, (int)(width/2), (int)(height/2), NoPadReconResY, NoPadReconResUV

def Parser(imgPath):
    #OrgResY, OrgResUV, ReconResY, ReconResUV, PredY, PredUV, width, height = tf.py_func(RawImageUnpack, [imgPath], [tf.int32,tf.int32,tf.int32,tf.int32,tf.int32,tf.int32,tf.int32,tf.int32])
    OrgResY, OrgResUV, ReconResY, ReconResUV, PredY, PredUV, width1, height1, width2, height2, width3, height3, width4, height4, NoPadReconResY, NoPadReconResUV = tf.py_func(RawImageUnpack, [imgPath],
                                                                                                                                                                                [tf.int32, tf.int32, tf.int32,
                                                                                                                                                                                 tf.int32, tf.int32, tf.int32,
                                                                                                                                                                                 tf.int32, tf.int32, tf.int32,
                                                                                                                                                                                 tf.int32, tf.int32, tf.int32,
                                                                                                                                                                                 tf.int32, tf.int32, tf.int32, tf.int32])
    #OrgResY = LumaImagePack(OrgResY, Cheight, Cwidth)
    #ReconResY = LumaImagePack(ReconResY, Cheight, Cwidth)
    #PredY =  LumaImagePack(PredY, Cheight, Cwidth)
    OrgResY = tf.reshape(OrgResY, [height1, width1, 1],name = "ROrgResY")
    OrgResUV = tf.reshape(OrgResUV, [height2, width2,2], name = "ROrgResUV")
    ReconResY = tf.reshape(ReconResY, [height3, width3, 1], name = "RReconResY")
    ReconResUV = tf.reshape(ReconResUV, [height4, width4,2], name = "RReconResUV")
    PredY = tf.reshape(PredY, [height3, width3, 1], name = "RPredY")
    PredUV = tf.reshape(PredUV, [height4, width4,2], name = "RPredUV")
    NoPadReconResY = tf.reshape(NoPadReconResY, [height1, width1, 1],name = "NoPadReconY")
    NoPadReconResUV = tf.reshape(NoPadReconResUV, [height2, width2, 2],name = "NoPadReconUV")

    Luma_Input = tf.cast(Concatenation((ReconResY, PredY)),tf.float32)/1023.
    Croma_Input = tf.cast(Concatenation((ReconResUV, PredUV)), tf.float32)/1023.
    Luma_Output = tf.cast((OrgResY), tf.float32)/1023.
    Croma_Output = tf.cast((OrgResUV), tf.float32)/1023.
    NoPadReconResY = tf.cast(NoPadReconResY,tf.float32)/1023.
    NoPadReconResUV = tf.cast(NoPadReconResUV, tf.float32)/1023.
    minWH = tf.minimum(width1, height1)
    return Luma_Input, Croma_Input, Luma_Output, Croma_Output, minWH, NoPadReconResY, NoPadReconResUV


def build_dataset(index, dataset_path, batch_size):
    with tf.device('/cpu:0'):
        data_path = getFileList(dataset_path[index])
        data_files = tf.convert_to_tensor(data_path, dtype=tf.string)
        data = tf.data.Dataset.from_tensor_slices(data_files)
        data = data.map(lambda x:Parser(x))
        data = data.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
        data = data.repeat()
        iterator = data.make_initializable_iterator()
        initializer = iterator.initializer
        next_element = iterator.get_next()
        return initializer, next_element