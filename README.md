Simple Example to run Keras models in multiple processes
===================


This git repo contains an example to illustrate how to run Keras models prediction in multiple processes with multiple gpus. Each process owns one gpu.  I wanted to run prediction by using multiple gpus, but did not find a clear solution after searching online. So, I created this example to show how to do that. Hope this git repo can help others who met the same problem.

This software works as a producer-consumer model.  The scheduler scans the image path and put all of them into a Queue; while each worker as a separate process process the images in the Queue. If all of images are proceeded, the worker process will terminate.

Pay attention to the implementation Vgg16Worker::Run() 

    def run(self):
        #set environment
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self._gpuid)
        #load models
        import vgg16
        #download the vgg16 weights from
        xnet = vgg16.Vgg16('vgg16_weights_tf_dim_ordering_tf_kernels.h5')

        print 'vggnet init done', self._gpuid

        while True:
            xfile = self._queue.get()
            if xfile == None:
                self._queue.put(None)
                break
            label = self.predict(xnet, xfile)
            print 'woker', self._gpuid, ' xfile ', xfile, " predicted as label", label

----------


How to run the sample
-------------

 **Dependency:**

> - Keras 2.0
> - Tensorflow
> - OpenCV

 **Download VGG16 Keras Model from:**
https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5

 **Command:**
 usage: main.py [-h] [--imgpath IMGPATH] [--gpuids GPUIDS]

optional arguments:
  -h, --help         show this help message and exit
  --imgpath IMGPATH  path to your images to be proceed
  --gpuids GPUIDS    gpu ids to run


  
