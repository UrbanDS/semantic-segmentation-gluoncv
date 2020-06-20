import mxnet as mx
from mxnet import image
from mxnet.gluon.data.vision import transforms
from gluoncv.data.transforms.presets.segmentation import test_transform
from matplotlib import pyplot as plt
import glob
import gluoncv
from skimage import io
import mxnet as mx
from mxnet import image
from gluoncv.utils.viz import get_color_pallete
import matplotlib.image as mpimg
import glob
import os

import mxnet as mx
from mxnet import image
from mxnet.gluon.data.vision import transforms
from gluoncv.data.transforms.presets.segmentation import test_transform
from matplotlib import pyplot as plt
import gluoncv
import numpy
from PIL import Image


model = gluoncv.model_zoo.get_model('psp_resnet101_ade', pretrained=True, ctx=mx.cpu(0))
ctx = mx.cpu(0)
import numpy
from PIL import Image

import glob
all_files = glob.glob(r'photos/*.jpg')
Essex_test_files = glob.glob(r'photos/*.jpg')
print(Essex_test_files)
saved_path = r'op/'
# Essex_test_files = glob.glob(r'I:\t\depthmap\*.jpg')
Essex_test_files = glob.glob(r'photos/*.jpg')
print("len of Essex_test_files:", len(Essex_test_files))

import random

print("before sampling:", len(Essex_test_files))
Essex_test_files = random.sample(Essex_test_files, len(Essex_test_files))
print("after sampling:", len(Essex_test_files))

for idx, filename in enumerate(Essex_test_files):
    try:
        img = image.imread(filename)
        img = image.resize_short(img, 1024)
        #         img = image.resize_short(img, 100)

        print("filename: ", filename)
        #         ctx = mx.gpu(0)

        img = test_transform(img, ctx)
        #         print("img: ", img)

        output = model.predict(img)

        #         print("output: ", output)

        predict = mx.nd.squeeze(mx.nd.argmax(output, 1)).asnumpy()

        #         print("predict: ", predict)

        mask = get_color_pallete(predict, 'ade20k')

        # predict.save('predict.png')
        # mmask = mpimg.imread('output.png')
        predict = predict.astype(numpy.uint8)
        convert_single_array = numpy.array(predict)
        unique_numbers = numpy.unique(convert_single_array)
        print(unique_numbers)
        new_basename = os.path.basename(filename).replace(".jpg", ".png")
        new_name = os.path.join(saved_path, new_basename)
        mask.save(new_name)
        color_img = image.imread(new_name)
        # colors, counts = numpy.unique(color_img.reshape(-1, 3), return_counts=True, axis=0)
        total_pixel = numpy.sum(predict)
        for i in unique_numbers:
            individual_count  = numpy.sum(predict==i)
            print(individual_count)
            csv_data = []
            csv_data.append(filename)
            csv_data.append(total_pixel)
        # csv_data = [filename,predict,colors,counts,total_pixel]
            print(csv_data)
        if idx % 10 == 0:
            print("Processed: ", idx)
    except Exception as e:
        print("Error in PSPNet:", idx+'       '+filename, e)
        continue