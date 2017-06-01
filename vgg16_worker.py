from multiprocessing import Queue, Process
import cv2
import numpy as np
import os

class Vgg16Worker(Process):
    def __init__(self, gpuid, queue):
        Process.__init__(self, name='ModelProcessor')
        self._gpuid = gpuid
        self._queue = queue

    def run(self):

        #set enviornment
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self._gpuid)

        #load models
        import vgg16
        #download the vgg16 weights from
        #https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5
        xnet = vgg16.Vgg16('vgg16_weights_tf_dim_ordering_tf_kernels.h5')

        print 'vggnet init done', self._gpuid

        while True:
            xfile = self._queue.get()
            if xfile == None:
                self._queue.put(None)
                break
            label = self.predict(xnet, xfile)
            print 'woker', self._gpuid, ' xfile ', xfile, " predicted as label", label

        print 'vggnet done ', self._gpuid

    def predict(self, xnet, imgfile):
        #BGR
        im = cv2.resize(cv2.imread(imgfile), (224, 224)).astype(np.float32)

        im[:, :, 0] -= 103.939
        im[:, :, 1] -= 116.779
        im[:, :, 2] -= 123.68

        im = im.reshape((1, 224, 224, 3))
        out = xnet.predict(im)
        return np.argmax(out)

