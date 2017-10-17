'''
    进阶的卷积网络：数据集CIFAR-100
    下载地址：http://www.cs.toronto.edu/~kriz/cifar.html
    创建文件夹：cifar10，里面放入解压后的cifar10的文件
'''
from numpy import *
import pickle

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import *
import tensorflow as tf

def read_cifar10(file_queue):
    labels = []
    data = []
    name = []
    for file in file_queue:
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
            # 图片名称
            for n in dict[b'filenames']:
                name.append(str(n, 'utf-8'))
            # 图片标签
            labels.extend(dict[b'labels'])
            # 图片数据1024+1024+1024(红绿蓝)32*32
            m = dict[b'data']
            data.extend(m)
    images = []
    for d in data:
        unit8 = reshape(d, [3,32,32])
        image = transpose(unit8, [1,2,0])
        images.append(image)
    # unit8 = reshape(data, [3,32,32]) # [depth, height, width]
    # image = tf.transpose(unit8, [1,2,0]) # [height, width, depth]
    return data, labels, name, images

def rgb2gray(rgb):
    return dot(rgb[...,:3], [0.299, 0.587, 0.114])

def showPicture(image, name):
    # image = rgb2gray(image)
    plt.subplots_adjust(wspace=0.2, hspace = 0.2, left=0.12, bottom = 0.5, right = 0.85, top = 0.6)
    plt.imshow(image)#cmap='Greys_r'
    plt.axis('off')
    plt.title(name)
    plt.show()

def showAll(images, names):
    root = Tk()
    class Picture(object):
        def __init__(self, object):
            self.parent = object;
            self.images = images;
            self.names = names;
            self.num = len(names)
            # print(len(names))
            self.n = 0
            Button(root, text='上一张图片', command=self.prePicture).pack()
            Button(root, text='下一张图片', command=self.nextPicture).pack()
            fig = []
            self.canvas = []
            for i in range(self.num):
                fig.append(plt.figure(i))
                plt.subplots_adjust(wspace=0.2, hspace = 0.2, left=0.12, bottom = 0.5, right = 0.85, top = 0.6)
                plt.imshow(self.images[i])#cmap='Greys_r'
                plt.axis('off')
                plt.title(self.names[i])
                self.canvas.append(FigureCanvasTkAgg(fig[i],master=self.parent))
            self.canvas[0]._tkcanvas.pack()

        def prePicture(self):
            if(self.n == 0):
                print("到头了")
                pass
            else:
                self.canvas[self.n]._tkcanvas.pack_forget()
                self.n -= 1
                print("当前图片：",self.n+1,"总共：",self.num )
                self.canvas[self.n]._tkcanvas.pack()

        def nextPicture(self):
            if(self.n == len(self.names) - 1):
                print("到尾了")
                pass
            else:
                self.canvas[self.n]._tkcanvas.pack_forget()
                self.n += 1
                print("当前图片：",self.n+1,"总共：",self.num )
                self.canvas[self.n]._tkcanvas.pack()
    Picture(root)
    root.mainloop()

def test():
    file_queue = ['cifar10/data_batch_1','cifar10/data_batch_2']
    data, labels, name, images = read_cifar10(file_queue)
    showAll(images[:10], name[:10])

'''
    主要部分
'''
def variable_with_weight_loss(shape, stddev, w1):
    var = tf.Variable(tf.truncated_normal(shape, stddev))
    if wl is not None:
        weigth_loss = tf.multiply(tf.nn.l2_loss(var), wl, name='weight_loss')
        tf.add_to_collection('losses', weigth_loss)
    return var