import torch
from torch import nn
from Utilities import warp_utils
import numpy as np
#from ObstructionRemoval import warp_utils

def create_outgoing_mask(flow):
    """
    Generate a mask based on a input flow
    :param flow: The flow that is going to be used to create the mask
    :return: The mask
    """
    #gets the desired dimensions
    num_batch,height,width,_ = flow.shape

    #create the x grid and y grid to construct the mask on
    gridx = torch.reshape(torch.range(0,width-1),[1,1,width])
    gridx = torch.from_numpy(np.tile(gridx,[num_batch,height,1]))
    gridy = torch.reshape(torch.range(0, height-1), [1, height, 1])
    gridy = torch.from_numpy(np.tile(gridy, [num_batch, 1, width]))

    #create the mask
    flowu,flowv = torch.unbind(flow,3)[0:2]
    posx = gridx.float()+flowu
    posy = gridy.float()+flowv
    #insidex = torch.logical_and(posx <= (width-1).type(torch.FloatTensor),posx >= 0)
    insidex = torch.logical_and(posx <= (width-1),posx >= 0)
    insidey = torch.logical_and(posy <= (height-1),posy >= 0)
    inside = torch.logical_and(insidex,insidey)
    return torch.unsqueeze(inside.type(torch.FloatTensor),3)

class ImageReconstruction:
    def __init__(self,batch_size,CROP_PATCH_H,CROP_PATCH_W,level=4,weighted_fusion=True):
        """
        :param batch_size: size of the batche
        :param CROP_PATCH_H: height of the patch that is being cropped
        :param CROP_PATCH_W: width of the patch that is being cropped
        :param level: the level that we are working at
        :param weighted_fusion: is this network weighted or not
        """
        self.batch_size = batch_size
        self.CROP_PATCH_H = CROP_PATCH_H
        self.CROP_PATCH_W = CROP_PATCH_W
        self.level = level
        self.weighted_fusion = weighted_fusion

    def downsample(self,x,outchannels,filtersize):
        """
        Downsample the input tensors
        :param x: The input tensors
        :param outchannels: The number of output channels
        :param filtersize: The size of the kernel
        :return: The result of the leaky relu applied to two convolutional layers
        """
        x = nn.AvgPool2d(x,2,2)
        m = nn.LeakyReLU(.1)
        nn.Conv2d.weight = x
        x = m(nn.Conv2d(x.shape,outchannels,filtersize,1))
        x = m(nn.Conv2d(x.shape,outchannels,filtersize,1))
        return x

    def upsample(self,x,outchannels,skpCn):
        """
        Upsample the input tensors
        :param x: The input tensors
        :param outchannels: The number of output channels
        :param skpCn: The result of the pretrained network
        :return: The result of the leaky relu applied to two convolutional layers
        """
        x = nn.AvgPool2d(x, 2, 2)
        m = nn.LeakyReLU(.1)
        nn.Conv2d.weight = x
        x = m(nn.Conv2d(x.shape, outchannels, 3, 1))
        x = m(nn.Conv2d(torch.cat((x,skpCn),-1).shape, outchannels, 3, 1))
        return x

    def network(self,image_2B,alpha,image_2,image_0,image_1,image_3,image_4,flow20,flow21,flow23,flow24,level):
        """
        Run a convolutional network on the provied images, alpha map, and flows at a given level to seperate the background
        from the foreground. The network structure is Conv-ReLU-Conv-ReLU-Conv-ReLU-Conv-ReLU-Conv-ReLU and then either
        Conv-Softmax or Conv
        :param image_2B: background of the image we are applying the network to
        :param alpha: Alpha map for reconstructing the background
        :param image_2: image we are applying the network two
        :param image_0: The first image
        :param image_1: The second image
        :param image_3: The fourth image
        :param image_4: The last image
        :param flow20: The flow decomposition for image 0 with respect to image 2
        :param flow21: The flow decomposition for image 1 with respect to image 2
        :param flow23: The flow decomposition for image 3 with respect to image 2
        :param flow24: The flow decomposition for image 4 with respect to image 2
        :param level: The level that we are running the network at
        :return: The background and the alpha map
        """
        #b,h,w,_ = torch.unbind(image_2.shape)
        b,h,w,_ = image_2.shape

        #warp registered background images
        #registered_background_20 = self.warp(image_0,flow20,b,h,w,3)
        #registered_background_21 = self.warp(image_1,flow21,b,h,w,3)
        #registered_background_23 = self.warp(image_3,flow23,b,h,w,3)
        #registered_background_24 = self.warp(image_4,flow24,b,h,w,3)
        registered_background_20 = image_0
        registered_background_21 = image_1
        registered_background_23 = image_3
        registered_background_24 = image_4

        #outgoing mask
        outgoing_mask_20 = create_outgoing_mask(flow20)
        outgoing_mask_21 = create_outgoing_mask(flow21)
        outgoing_mask_23 = create_outgoing_mask(flow23)
        outgoing_mask_24 = create_outgoing_mask(flow24)

        #calculate the difference between the image we are loooking at and each registered background
        diff_20 = torch.abs(image_2B-registered_background_20)
        diff_21 = torch.abs(image_2B-registered_background_21)
        diff_23 = torch.abs(image_2B-registered_background_23)
        diff_24 = torch.abs(image_2B-registered_background_24)

        B_registered = torch.cat([image_2B,alpha,image_2,
                                    registered_background_20,outgoing_mask_20,diff_20,
                                    registered_background_21,outgoing_mask_21,diff_21,
                                    registered_background_23,outgoing_mask_23,diff_23,
                                    registered_background_24,outgoing_mask_24,diff_24],-1)

        #actual network
        x = torch.cat(B_registered,axis=3)
        nn.Conv2d.weight = x
        x = nn.Conv2d(x.shape,128,(3,3))
        m = nn.LeakyReLU(.1)
        x = m(x)
        nn.Conv2d.weight = x
        x = nn.Conv2d(x.shape, 128, (3, 3))
        m = nn.LeakyReLU(.1)
        x = m(x)
        nn.Conv2d.weight = x
        x = nn.Conv2d(x.shape, 96, (3, 3))
        m = nn.LeakyReLU(.1)
        x = m(x)
        nn.Conv2d.weight = x
        x = nn.Conv2d(x.shape, 64, (3, 3))
        m = nn.LeakyReLU(.1)
        x = m(x)
        nn.Conv2d.weight = x
        x = nn.Conv2d(x.shape, 32, (3, 3))
        m = nn.LeakyReLU(.1)
        x = m(x)

        #if weighted network, then add in a softmax layer.
        #else, have one more convolutional layer
        if self.weighted_fusion:
            nn.Conv2d.weight = x
            x = nn.Conv2d(5,3,(1,1))
            weights = torch.nn.Softmax(-1)
            img_diff_0 = registered_background_20 - image_2B
            img_diff_1 = registered_background_21 - image_2B
            img_diff_3 = registered_background_23 - image_2B
            img_diff_4 = registered_background_24 - image_2B
            output_B = image_2B + (weights[...,0:1]*img_diff_0+weights[...,1:2]*img_diff_1+weights[...,2:3]*img_diff_3+weights[...,3:4]*img_diff_4)
            return output_B,alpha+x[...,4:5]
        else:
            nn.Conv2d.weight = x
            x = nn.Conv2d(4,3,(1,1))
            return image_2B+x[...,0:3],alpha+x[...,3:4]

    def warp(self,I,F,b,h,w,c):
        """
        Reshape the image and background labels to be the desired shape
        :param I: image to be warped
        :param F: The background labels
        :param b: The desired number of batches
        :param h: The desired height
        :param w: The desired width
        :param c: The desired number of channels
        :return: The reshaped image
        """
        return torch.reshape(warp_utils.dense_image_warp(I,torch.stack([-F[...,-1], -F[...,0]],-1)),[b,h,w,c])

    def build_model(self,input_images,B0_last,B1_last,B2_last,B3_last,B4_last,A0_last,A1_last,A2_last,A3_last,A4_last,
                    FB01,FB02,FB03,FB04,FB10,FB12,FB13,FB14,FB20,FB21,FB23,FB24,FB30,FB31,FB32,FB34,FB40,FB41,FB42,FB43):
        """
        Create the model that the network is run on.
        :param input_images: The images we want to run the network on
        :param B0_last: The last background of image 0
        :param B1_last: The last background of image 1
        :param B2_last: The last background of image 2
        :param B3_last: The last background of image 3
        :param B4_last: The last background of image 4
        :param A0_last: The last alpha map of image 0
        :param A1_last: The last alpha map of image 1
        :param A2_last: The last alpha map of image 2
        :param A3_last: The last alpha map of image 3
        :param A4_last: The last alpha map of image 4
        :param FB01: The background labels of images 0 and 1
        :param FB02: The background labels of images 0 and 2
        :param FB03: The background labels of images 0 and 3
        :param FB04: The background labels of images 0 and 4
        :param FB10: The background labels of images 1 and 0
        :param FB12: The background labels of images 1 and 2
        :param FB13: The background labels of images 1 and 3
        :param FB14: The background labels of images 1 and 4
        :param FB20: The background labels of images 2 and 0
        :param FB21: The background labels of images 2 and 1
        :param FB23: The background labels of images 2 and 3
        :param FB24: The background labels of images 2 and 4
        :param FB30: The background labels of images 3 and 0
        :param FB31: The background labels of images 3 and 1
        :param FB32: The background labels of images 3 and 2
        :param FB34: The background labels of images 3 and 4
        :param FB40: The background labels of images 4 and 0
        :param FB41: The background labels of images 4 and 1
        :param FB42: The background labels of images 4 and 2
        :param FB43: The background labels of images 4 and 3
        :return:The predicted background and alpha map for each image
        """
        b = self.batch_size
        h = self.CROP_PATCH_H // (2**self.level)
        w = self.CROP_PATCH_W // (2**self.level)

        #upsample each image to make it a given size
        I0 = nn.UpsamplingBilinear2d((h,w))
        I0 = I0(input_images[...,0:3])
        I1 = nn.UpsamplingBilinear2d((h, w))
        I1 = I0(input_images[..., 3:6])
        I2 = nn.UpsamplingBilinear2d((h, w))
        I2 = I0(input_images[..., 6:9])
        I3 = nn.UpsamplingBilinear2d((h, w))
        I3 = I0(input_images[..., 9:12])
        I4 = nn.UpsamplingBilinear2d((h, w))
        I4 = I0(input_images[..., 12:15])

        #if the level is 4, create the final background images
        if self.level == 4:
            B0_last = (I0+self.warp(I1,FB01,b,h,w,3)+self.warp(I2,FB02,b,h,w,3)+self.warp(I3,FB03,b,h,w,3)+self.warp(I4,FB04,b,h,w,3))/5.0
            B1_last = (I1+self.warp(I0,FB10,b,h,w,3)+self.warp(I2,FB12,b,h,w,3)+self.warp(I3,FB13,b,h,w,3)+self.warp(I4,FB14,b,h,w,3))/5.0
            B2_last = (I2+self.warp(I0,FB20,b,h,w,3)+self.warp(I1,FB21,b,h,w,3)+self.warp(I3,FB23,b,h,w,3)+self.warp(I4,FB24,b,h,w,3))/5.0
            B3_last = (I3+self.warp(I0,FB30,b,h,w,3)+self.warp(I1,FB31,b,h,w,3)+self.warp(I2,FB32,b,h,w,3)+self.warp(I4,FB34,b,h,w,3))/5.0
            B4_last = (I4+self.warp(I0,FB40,b,h,w,3)+self.warp(I1,FB41,b,h,w,3)+self.warp(I2,FB42,b,h,w,3)+self.warp(I3,FB43,b,h,w,3))/5.0

        #create the alpha maps
        A0_last = torch.zeros_like(B0_last[...,0:1])
        A1_last = torch.zeros_like(B1_last[...,0:1])
        A2_last = torch.zeros_like(B2_last[...,0:1])
        A3_last = torch.zeros_like(B3_last[...,0:1])
        A4_last = torch.zeros_like(B4_last[...,0:1])

        #run the network on each image
        B0_pred,A0_pred = self.network(B0_last,A0_last,I0,I1,I2,I3,I4,FB01,FB02,FB03,FB04,self.level)
        B1_pred,A1_pred = self.network(B1_last,A1_last,I1,I0,I2,I3,I4,FB10,FB12,FB13,FB14,self.level)
        B2_pred,A2_pred = self.network(B2_last,A2_last,I2,I0,I1,I3,I4,FB20,FB21,FB23,FB24,self.level)
        B3_pred,A3_pred = self.network(B3_last,A3_last,I3,I0,I1,I2,I4,FB30,FB31,FB32,FB34,self.level)
        B4_pred,A4_pred = self.network(B4_last,A4_last,I4,I0,I1,I2,I3,FB40,FB41,FB42,FB43,self.level)

        return B0_pred,B1_pred,B2_pred,B3_pred,B4_pred,A0_pred,A1_pred,A2_pred,A3_pred,A4_pred