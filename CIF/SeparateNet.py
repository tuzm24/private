import numpy as np
import struct
import math
import tensorflow as tf
from tensorflow.contrib.layers import batch_norm
from tensorflow.contrib.framework import arg_scope
import os
import random
import yaml



save_file = "model/model_105.ckpt"

dataset_kinds = 17

dataset_path = ["Pad_Dataset/Block/4x4/4x4", "Pad_Dataset/Block/4x4/4x8", "Pad_Dataset/Block/4x4/4x16", "Pad_Dataset/Block/4x4/4x32", "Pad_Dataset/Block/4x4/8x4",
                "Pad_Dataset/Block/4x4/16x4", "Pad_Dataset/Block/4x4/32x4", "Pad_Dataset/Block/8x8/8x8", "Pad_Dataset/Block/8x8/16x8", "Pad_Dataset/Block/8x8/32x8",
                "Pad_Dataset/Block/8x8/8x16", "Pad_Dataset/Block/8x8/8x32", "Pad_Dataset/Block/16x16/16x16", "Pad_Dataset/Block/16x16/16x32", "Pad_Dataset/Block/16x16/32x16",
                "Pad_Dataset/Block/32x32/32x32", "Pad_Dataset/Block/64x64/64x64", "Pad_Dataset/CTU"]
dataset_iter = [33, 33, 6, 1, 33,
                6, 1, 70, 20, 2,
                20, 2, 83, 13, 13,
                100, 100, 100]
dataset_size = [4, 4, 4, 4, 4,
                4, 4, 8, 8, 8,
                8, 8, 16, 16, 16,
                32, 64, 128]

dataset_area = [1,2,4,8,2,
                4,8,4,8,16,
                8,16,16,32,32,
                64,256]

dataset_div = [256,128,64,32,128,
               64,32,64,32,16,
               32, 16, 16, 8, 8,
               4, 1]

for i in range(18):
    dataset_iter[i] = dataset_iter[i] * 300

#array 길이 17
dataset_name = ['4x4', '4x8', '4x16', '4x32', '8x4', '16x4', '32x4', '8x8', '16x8', '32x8', '8x16', '8x32', '16x16', '16x32', '32x16','32x32', '64x64']

batch_size = 6
NumberK = 12
nb_between = 2
nb_addition_layer = 3


learning_rate = 1e-4
epsilon = 1e-4 # AdamOptimizer epsilon


SUPPORTED_EXTENSIONS = ('.bin')

class ConfigMember(dict):
    def __getattr__(self, name):
        value = self[name]
        if isinstance(value, dict):
            value = ConfigMember(value)
        return value

class Config(dict):
    def __init__(self, file_path):
        assert os.path.exists(file_path), "ERROR: Config File doesn't exist."
        with open(file_path, 'r') as f:
            self.member = yaml.load(f)
            f.close()
#        self.PRETRAINED_MODEL_PATH = self.MODEL_PATH + self.PRETRAINED_MODEL_PATH
        self.TENSORBOARD_LOG_PATH = self.MODEL_PATH + self.TENSORBOARD_LOG_PATH

        os.makedirs(self.MODEL_PATH,exist_ok=True)
        os.makedirs(self.TENSORBOARD_LOG_PATH,exist_ok=True)

    def __getattr__(self, name):
        value = self.member[name]
        if isinstance(value, dict):
            value = ConfigMember(value)
        return value

def write_yml(config):
    path = config.NET_INFO_PATH
    with open(path, 'w+') as fp:
        for key, value in config.member.items():
            if type(value)==str:
                fp.write("{}: '{}'\n".format(key, value))
            else:
                fp.write("{}: {}\n".format(key, value))
        fp.close()


def psnr(loss):
    if loss==0:
        return 100
    return math.log10(1/loss)*10

def read_dataset(rootdir, pattern='*.JPEG'):
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

def Concatenation(layers) :
    return tf.concat(layers, axis=-1)


def LumaImagePack(img, Cheight, Cwidth):
    packingLuma = np.zeros((Cheight, Cwidth, 4), dtype = 'i')
    for i in range(Cheight):
        for j in range(Cwidth):
            packingLuma[i][j][0] = img[i*2][j*2]
            packingLuma[i][j][1] = img[i * 2][j * 2 + 1]
            packingLuma[i][j][2] = img[i * 2 + 1][j * 2]
            packingLuma[i][j][3] = img[i * 2 + 1][j * 2 + 1]
    return packingLuma;

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
    #PredY = LumaImagePack(PredY, Cheight, Cwidth)
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


def build_dataset(index):
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

def conv2d(x, cnum, ksize, stride=1, rate=1, name='conv',
             padding='SAME', activation=tf.nn.relu, training=True):
    assert padding in ['SYMMETRIC', 'VALID', 'SAME', 'REFELECT']
    if padding == 'SYMMETRIC' or padding == 'REFELECT':
        p = int(rate*(ksize-1)/2)
        x = tf.pad(x, [[0,0], [p, p], [p, p], [0,0]], mode=padding)
        padding = 'VALID'
    x = tf.layers.conv2d(
        x, cnum, ksize, stride, dilation_rate=rate,
        activation=activation, padding=padding, name=name)
    return x

def batch_activ_conv(x, out_features, kernel_size, is_training, activation=None, rate=1, name="layer"):
    with tf.variable_scope(name):
          # no dropout!
          x = tf.contrib.layers.batch_norm(x, scale=True, is_training=is_training, updates_collections=None)
          x = tf.nn.relu(x)
          x = conv2d(x, out_features, kernel_size, activation=activation, rate=rate)
          return x


def Relu(x):
    return tf.nn.relu(x)

def Batch_Normalization(x, training, scope):
    with arg_scope([batch_norm],
                   scope=scope,
                   updates_collections=None,
                   decay=0.9,
                   center=True,
                   scale=True,
                   zero_debias_moving_mean=True) :
        return tf.cond(training,
                       lambda : batch_norm(inputs=x, is_training=training, reuse=None),
                       lambda : batch_norm(inputs=x, is_training=training, reuse=True))


def bottleneck_layer(x, scope, training, filters):
    # print(x)
    with tf.name_scope(scope):
        x = tf.contrib.layers.batch_norm(x, scale=True, is_training=training, updates_collections=None)
        x = Relu(x)
        x = conv2d(x, 4*filters, 1, name = scope+'_conv1')
        x = tf.contrib.layers.batch_norm(x, scale=True, is_training=training, updates_collections=None)
        x = Relu(x)
        x = conv2d(x, filters, 3, name = scope+'_conv2')
        return x

def Dense_block(input_x, nb_layers, training, layer_name):
    with tf.name_scope(layer_name):
        layers_concat = list()
        layers_concat.append(input_x)

        x = bottleneck_layer(input_x, scope=layer_name+'_bottleN_'+str(0), training = training, filters = NumberK)

        for i in range(nb_layers - 1):
            x = Concatenation(layers_concat)
            x = bottleneck_layer(x, scope=layer_name + '_bottleN'+str(i+1), training = training, filters = NumberK)
            layers_concat.append(x)
        x = Concatenation(layers_concat)

        return x



def Main_Network(Input_Luma, Input_Croma, training, reuse = False):
    with tf.variable_scope("Preprocessing", reuse = reuse) as scope:
        if reuse:
            scope.reuse_variable()
        ch_num = [0, 32, 16, 16]
        conv1 = conv2d(Input_Luma, ch_num[1], 3, name = 'conv1', padding='VALID')
        conv2 = conv2d(conv1, ch_num[2], 1, padding='VALID', name = 'conv2')
        conv3 = conv2d(conv2, ch_num[3], 3, padding='VALID', name = 'conv3')
        n1 = np.array([[1,0],[0,0]])
        n1 = np.stack((n1,)* ch_num[3], axis=-1)
        n2 = np.array([[0,1],[0,0]])
        n2 = np.stack((n2,)*ch_num[3], axis=-1)
        n3 = np.array([[0,0],[1,0]])
        n3 = np.stack((n3,)*ch_num[3], axis=-1)
        n4 = np.array([[0,0],[0,1]])
        n4 = np.stack((n4,) * ch_num[3], axis=-1)
        n = np.stack((n1, n2, n3, n4), axis=-1)
        n = n.astype(np.float32)
        w = tf.Variable(initial_value= tf.constant(n), trainable=False)
        luma_result = tf.nn.depthwise_conv2d(conv3, w, [1,2,2,1], padding='VALID', name = 'luma_result')
        croma_result = conv2d(Input_Croma, 16, 3, padding='VALID', name = 'croma_result')
        mainNet1 =  Concatenation((luma_result, croma_result))
        CheckPoint64x64 = bottleneck_layer(mainNet1, 'CheckPoint64', training, 18)
        CheckPoint32x32 = Dense_block(CheckPoint64x64, nb_between, training, layer_name='CheckPoint32')
        CheckPoint16x16 = Dense_block(CheckPoint32x32, nb_between, training, layer_name='CheckPoint16')
        CheckPoint8x8 = Dense_block(CheckPoint16x16, nb_between, training, layer_name='CheckPoint8')
        CheckPoint4x4 = Dense_block(CheckPoint8x8, nb_between, training, layer_name='CheckPoint4')
        return CheckPoint64x64, CheckPoint32x32, CheckPoint16x16, CheckPoint8x8, CheckPoint4x4


def Additional_Network(input_x, number_layers, training, layer_name):
    with tf.name_scope(layer_name):
        layers_concat = list()
        layers_concat.append(input_x)

        x = bottleneck_layer(input_x, scope=layer_name+'_bottleN_'+str(0), training = training, filters = NumberK)

        for i in range(number_layers - 1):
            x = Concatenation(layers_concat)
            x = bottleneck_layer(x, scope=layer_name + '_bottleN'+str(i+1), training = training, filters = NumberK)
            layers_concat.append(x)
        x = Concatenation(layers_concat)
        x = tf.layers.conv2d(x, 6, [3,3], padding='SAME')
        return x


def Separable_Network(Input_Luma, Input_Croma, training, nb_layers = 8, reuse = False, name = 'Separable'):
    with tf.variable_scope(name+"_scope", reuse = reuse) as scope:
        if reuse:
            scope.reuse_variable()
        ch_num = [0, 32, 16, 16]
        conv1 = conv2d(Input_Luma, ch_num[1], 3, padding='VALID', name = name + 'conv1')
        conv3 = conv2d(conv1, ch_num[3], 3, padding='VALID', name = name+'conv3')
        n1 = np.array([[1,0],[0,0]])
        n1 = np.stack((n1,)* ch_num[3], axis=-1)
        n2 = np.array([[0,1],[0,0]])
        n2 = np.stack((n2,)*ch_num[3], axis=-1)
        n3 = np.array([[0,0],[1,0]])
        n3 = np.stack((n3,)*ch_num[3], axis=-1)
        n4 = np.array([[0,0],[0,1]])
        n4 = np.stack((n4,) * ch_num[3], axis=-1)
        n = np.stack((n1, n2, n3, n4), axis=-1)
        n = n.astype(np.float32)
        w = tf.Variable(initial_value= tf.constant(n), trainable=False)
        luma_result = tf.nn.depthwise_conv2d(conv3, w, [1,2,2,1], padding='VALID', name = name+'luma_result')
        croma_result = conv2d(Input_Croma, 16, 3, padding='VALID', name = name+'croma_result')
        mainNet1 =  Concatenation((luma_result, croma_result))

        CheckPoint64x64 = bottleneck_layer(mainNet1, name+'CheckPoint64', training, 18)

        layers_concat = list()
        layers_concat.append(CheckPoint64x64)
        for i in range(nb_layers - 1):
            x = Concatenation(layers_concat)
            x = bottleneck_layer(x, scope=name + '_bottleN'+str(i+1), training = training, filters = NumberK)
            layers_concat.append(x)
        x = Concatenation(layers_concat)
        x = tf.layers.conv2d(x, 6, [3,3], padding='SAME')
        return x



def SeparableNet(reuse = False, name = 'SepaN', training = True, index=0):
    with tf.variable_scope(name+"_scope", reuse = reuse) as scope:
        if reuse:
            scope.reuse_variable()
        tf_initializer, tr_get_data = build_dataset(index)
        Luma_Input, Croma_Input, Luma_Output, Croma_Output, minWH, NoPad_Luma_Input, NoPad_Croma_Input = tr_get_data

        n1 = np.array([[1,0],[0,0]])
        n1 = np.stack((n1,)*1, axis=-1)
        n2 = np.array([[0,1],[0,0]])
        n2 = np.stack((n2,)*1, axis=-1)
        n3 = np.array([[0,0],[1,0]])
        n3 = np.stack((n3,)*1, axis=-1)
        n4 = np.array([[0,0],[0,1]])
        n4 = np.stack((n4,)*1, axis=-1)
        n = np.stack((n1,n2,n3,n4), axis=-1)
        n = n.astype(np.float32)

        w = tf.Variable(initial_value=tf.constant(n), trainable=False)
        c = tf.nn.depthwise_conv2d(Luma_Output, w, [1,2,2,1], padding='VALID')

        Ground_Truth = Concatenation((c, Croma_Output))

        Noc = tf.nn.depthwise_conv2d(NoPad_Luma_Input, w, [1,2,2,1], padding = 'VALID')
        with_LF = Concatenation((Noc, NoPad_Croma_Input))
        SepaResult = Separable_Network(Luma_Input, Croma_Input, training, name=name+"Net", nb_layers=8)
        loss = tf.reduce_mean(tf.losses.absolute_difference(labels = Ground_Truth, predictions= SepaResult))
        L2_loss = tf.losses.mean_squared_error(labels = Ground_Truth, predictions = SepaResult)
        L2_LF_loss = tf.losses.mean_squared_error(labels =Ground_Truth, predictions= with_LF)
        return loss, L2_loss, L2_LF_loss, tf_initializer


class Trainer(object):
    def __init__(self, config, name, index):
        self.loss, self.L2_loss, self.L2_LF_loss, self.tf_initializer = SeparableNet(reuse=False, name=name, training = True, index=index)
        self.optimizer =  tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=epsilon)
        self.opt = self.optimizer.minimize(self.loss)
        self.writer = tf.summary.FileWriter(config.TENSORBOARD_LOG_PATH + config.INDEX[index])
        self.LFwriter = tf.summary.FileWriter(config.TENSORBOARD_LOG_PATH + "LF" + config.INDEX[index])
        self.log = tf.Variable(0.0)
        self.summary = tf.summary.scalar('TU' + config.INDEX[index], self.log)


networks = list()
config = Config("net_info.yml");


for i in range(dataset_kinds):
    new_net = Trainer(config, "Sepa"+config.INDEX[i], i)
    networks.append(new_net)


iter = config.ITER
object_iter = config.OBJECT_ITER
write_op = tf.summary.merge_all()
saver = tf.train.Saver()
save_file = config.MODEL_SAVE
print("Network Build Complete")

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, save_file)
    for i in range(dataset_kinds):
        sess.run(networks[i].tf_initializer)
    print("Get File List Complete")
    flag = True
    while flag:
        for i in range(dataset_kinds):
            for j in range(dataset_kinds):
                if iter[j] <object_iter[j]:
                    continue
                flag = False
            if iter[i] >= object_iter[i]:
                continue
            while True:
                _, _loss, _LFLoss = sess.run([networks[i].opt, networks[i].L2_loss, networks[i].L2_LF_loss])
                _loss = psnr(_loss)
                _LFLoss= psnr(_LFLoss)
                summary = sess.run(write_op, {networks[i].log: _loss})
                networks[i].writer.add_summary(summary, iter[i])
                summary = sess.run(write_op, {networks[i].log: _LFLoss})
                networks[i].LFwriter.add_summary(summary, iter[i])
                iter[i] = iter[i]+1
                if iter[i] % 30000 == 0 or iter[i] >= object_iter[i]:
                    networks[i].writer.flush()
                    networks[i].LFwriter.flush()
                    print(iter[i])
                    print("SepaNet_" + dataset_name[i] + "_PSNR :", _loss)
                    print("Inloop_" + dataset_name[i] + "_PSNR : ", _LFLoss)
                    config.member['GLOBAL_STEP'] = config.GLOBAL_STEP +1
                    config.member['MODEL_SAVE'] = config.MODEL_PATH+'model_'+str(config.GLOBAL_STEP)+".ckpt"
                    write_yml(config)
                    config = Config("net_info.yml");
                    iter = config.ITER
                    save_path = saver.save(sess, config.MODEL_PATH+'model_'+str(config.GLOBAL_STEP)+".ckpt")
                    print("Model saved in file %s" % save_path)
                    if iter[i] >= object_iter[i]:
                        break


