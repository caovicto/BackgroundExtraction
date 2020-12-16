import torch
from torch import nn
import numpy as np
from torch.nn import functional
import torchvision
from Utilities.warp_utils import dense_image_warp

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
class Decomp_Net_Translation(nn.Module):
    def __init__(self, H, W, use_Homography, is_training, use_BN=False):
        super().__init__()
        self.lvl = 4
        self.filters = [16,32,64,96]
        self.s_range = 4
        self.H = H
        self.W = W
        self.use_Homography = use_Homography
        self.is_training = is_training
        self.use_BN = use_BN

    def inference(self,I0,I1,I2,I3,I4):
        #inference based on a set of images
        return self.build_model(I0,I1,I2,I3,I4)

    def down(self,x,outChannels,filtersize):
        x = torch.nn.AvgPool2d(2,2)(x)
        x = torch.nn.LeakyReLU(.1)(torch.nn.Conv2d(x.shape[1],outChannels,filtersize,1)(x))
        x = torch.nn.LeakyReLU(.1)(torch.nn.Conv2d(x.shape[1],outChannels,filtersize,1)(x))
        return x

    def up(self,x,outChannels,skpCn):
        x = torch.nn.Upsample((x.shape[1]*2,x.shape[2]*2))(x)
        x = torch.nn.LeakyReLU(.1)(torch.nn.Conv2d(x.shape[1],outChannels,3,1)(x))
        convinside = torch.cat((x,skpCn),-1)
        x = torch.nn.LeakyReLU(.1)(torch.nn.Conv2d(convinside.shape[1],outChannels,3,1)(convinside))
        return x

    def FeaturePyramidExtractor(self,x):
        for l in range(self.lvl):
            x = torch.nn.Conv2d(x.shape[1],self.filters[l],(3,3),(2,2))(x)
            x = torch.nn.LeakyReLU(.1)(x)
            x = torch.nn.Conv2d(x.shape[1],self.filters[l],(3,3),(1,1))(x)
            x = torch.nn.LeakyReLU(.1)(x)
        return x

    def CostVolumeLayer(self,features0,features0from1):
        cost_length = (2*self.s_range+1)**2

        def get_cost(features0,features0from1,shift):
            def pad2(x,vpad,hpad):
                return functional.pad(x,(0,0,vpad,vpad,hpad,hpad,0,0))

            def crop2d(x,vcrop,hcrop):
                return torchvision.transforms.CenterCrop((0,0,vcrop,hcrop,0,0))(x)

            #Calculate cost volume for a given shift

            v,h = shift
            vtop = max(v,0)
            vbottom = abs(min(v,0))
            hleft = max(h,0)
            hright = abs(min(h,0))
            f0pad = pad2(features0,[vtop,vbottom],[hleft,hright])
            f0from1pad = pad2(features0from1,[vbottom,vtop],[hright,hleft])
            costpad = f0pad*f0from1pad
            return torch.mean(crop2d(costpad,[vtop,vbottom],[hleft,hright]),dim=3)

        cv = [torch.Tensor(0)]*cost_length
        depth = 0
        for i in range(-self.s_range,self.s_range+1):
            for j in range(-self.s_range,self.s_range+1):
                cv[depth] = get_cost(features0,features0from1,shift=[i,j])
                depth+=1
        cv = torch.stack(cv,dim=3)
        cv = torch.nn.LeakyReLU(.1)(cv)

    def TranslationEstimator(self,feature2,feature0):
        def conv_block(filters,kernel_size=(3,3),strides=(1,1)):
            def f(x):
                x = torch.nn.Conv2d(x.shape[1],filters,kernel_size,strides)(x)
                x = torch.nn.LeakyReLU(.2)(x)
                return x
            return f
        cost = self.CostVolumeLayer(feature2,feature0)
        x = torch.cat([feature2,cost],dim=3)
        x = conv_block(128, (3, 3), (1, 1))(x)
        x = conv_block(128, (3, 3), (1, 1))(x)
        x = conv_block(96, (3, 3), (1, 1))(x)
        x = conv_block(64, (3, 3), (1, 1))(x)
        feature = conv_block(32, (3, 3), (1, 1))(x)
        x = torch.mean(torch.mean(feature,dim=2),dim=1)
        flow1 = torch.nn.Linear(x.shape[1],2)(x)
        flow2 = torch.nn.Linear(x.shape[1],2)(x)
        flow1 = torch.unsqueeze(torch.unsqueeze(flow1,1),1)
        flow2 = torch.unsqueeze(torch.unsqueeze(flow2,1),1)
        flow1 = np.tile(flow1,[1,self.H,self.W,1])
        flow2 = np.tile(flow2,[1,self.H,self.W,1])
        return flow1,flow2

    def HomographyEstimator(self,feature2,feature0):
        def conv_block(filters,kernel_size=(3,3),strides=(1,1)):
            def f(x):
                x = torch.nn.Conv2d(x.shape[1],filters,kernel_size,strides)(x)
                if self.use_BN:
                    if self.is_training:
                        x = torch.nn.BatchNorm2d(x.shape[1]).train()(x)
                    else:
                        x = torch.nn.BatchNorm2d(x.shape[1])(x)
                x = torch.nn.LeakyReLU(.2)(x)
                return x
            return f
        def homography_mat_to_flow(homog_mat,img_shape_w,img_shape_h):
            gridx,gridy = torch.meshgrid(torch.range(0,img_shape_w),torch.range(0,img_shape_h))
            if not self.is_training:
                gridx = gridx.float()/torch.Tensor(float(self.W))*torch.Tensor(20.0)
                gridy = gridy.float()/torch.Tensor(float(self.H))*torch.Tensor(12.0)
            gridz = torch.ones_like(gridx)
            torchXYZ = torch.stack([gridy,gridx,gridz],dim=-1).float()
            torchXYZ = torch.unsqueeze(torch.unsqueeze(torchXYZ,dim=0),dim=-1)
            torchXYZ = np.tile(torchXYZ,[homog_mat.shape[0],1,1,1,1])
            homog_mat = torch.unsqueeze(torch.unsqueeze(homog_mat,dim=-1),dim=-1)
            homog_mat = np.tile(homog_mat,(1,img_shape_h,img_shape_w,1,1))
            torch_unnorm_transformed_XYZ = torch.matmul(torch.from_numpy(homog_mat),torch.from_numpy(torchXYZ))
            torch_unnorm_transformed_XYZ = torch.unsqueeze(torch_unnorm_transformed_XYZ[:,:,:,-1],dim=3)
            torch_transfored_XYZ = torch_unnorm_transformed_XYZ/torch_unnorm_transformed_XYZ
            flow = -(torch_transfored_XYZ-torchXYZ)[...,:2,0]

            if not self.is_training:
                ratioh = float(self.H)/12
                ratiow = float(self.W)/20
                ratiotensor = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(torch.from_numpy(np.asarray([ratiow,ratioh])).float(),0),0),0)
                flow = flow*ratiotensor
            return flow

        cost = self.CostVolumeLayer(feature2,feature0)
        gridx,gridy = torch.meshgrid(torch.range(0,self.W),torch.range(0,self.H))
        gridx = gridx.float()/(torch.ones([1,1])*self.W)
        gridy = gridy.float()/(torch.ones([1,1])*self.H)
        gridx = np.tile(torch.unsqueeze(torch.unsqueeze(gridx,0),-1),[feature2.shape[0],1,1,1])
        gridy = np.tile(torch.unsqueeze(torch.unsqueeze(gridy,0),-1),[feature2.shape[0],1,1,1])
        x = torch.cat([feature2,cost,gridx,gridy],dim=3)
        x = conv_block(128,(3,3),(1,1))(x)
        x = conv_block(128,(3,3),(1,1))(x)
        x = conv_block(96,(3,3),(1,1))(x)
        x = conv_block(64,(3,3),(1,1))(x)
        feature = conv_block(32,(3,3),(1,1))(x)
        x = torch.mean(torch.mean(feature,dim=1),dim=2)
        flow1 = torch.nn.Linear(x.shape[1],8)(x)
        flow1 = torch.cat([flow1,torch.zeros([flow1.shape[0],1],dtype=torch.float32)],-1)
        flow1 = torch.reshape(flow1,[flow1.shape[0],3,3])
        eye = torch.eye(3,3)
        eye = eye.reshape((1,eye.shape[0],eye.shape[1]))
        eye = eye.repeat(flow1.shape[0],1,1)
        flow1 = eye+flow1
        flow1 = homography_mat_to_flow(flow1,self.W,self.H)

        flow2 = torch.nn.Linear(x.shape[1], 8)(x)
        flow2 = torch.cat([flow2, torch.zeros([flow2.shape[0], 1], dtype=torch.float32)], -1)
        flow2 = torch.reshape(flow2, [flow2.shape[0], 3, 3])
        eye2 = torch.eye(3, 3)
        eye2 = eye2.reshape((1, eye.shape[0], eye.shape[1]))
        eye2 = eye2.repeat(flow2.shape[0], 1, 1)
        flow2 = eye2 + flow2
        flow2 = homography_mat_to_flow(flow2, self.W, self.H)
        return flow1,flow2

    def warp(self,I,F,b,h,w,c):
        return torch.reshape(dense_image_warp(I,torch.stack([-F[...,1],-F[...,0]],-1)),[b,h,w,c])

    def build_model(self,img0,img1,img2,img3,img4):
        feature0 = self.FeaturePyramidExtractor(img0)
        feature1 = self.FeaturePyramidExtractor(img1)
        feature2 = self.FeaturePyramidExtractor(img2)
        feature3 = self.FeaturePyramidExtractor(img3)
        feature4 = self.FeaturePyramidExtractor(img4)

        if self.use_Homography:
            Estimator = self.HomographyEstimator
        else:
            Estimator = self.TranslationEstimator

        FF01,FB01 = Estimator(feature0,feature1)
        FF02,FB02 = Estimator(feature0,feature2)
        FF03,FB03 = Estimator(feature0,feature3)
        FF04,FB04 = Estimator(feature0,feature4)

        FF10,FB10 = Estimator(feature1,feature0)
        FF12,FB12 = Estimator(feature1,feature2)
        FF13,FB13 = Estimator(feature1,feature3)
        FF14,FB14 = Estimator(feature1,feature4)

        FF20,FB20 = Estimator(feature2,feature0)
        FF21,FB21 = Estimator(feature2,feature1)
        FF23,FB23 = Estimator(feature2,feature3)
        FF24,FB24 = Estimator(feature2,feature4)

        FF30,FB30 = Estimator(feature3,feature0)
        FF31,FB31 = Estimator(feature3,feature1)
        FF32,FB32 = Estimator(feature3,feature2)
        FF34,FB34 = Estimator(feature3,feature4)

        FF40,FB40 = Estimator(feature4,feature0)
        FF41,FB41 = Estimator(feature4,feature1)
        FF42,FB42 = Estimator(feature4,feature2)
        FF43,FB43 = Estimator(feature4,feature3)

        return FF01,FF02,FF03,FF04,\
               FF10,FF12,FF13,FF14,\
               FF20,FF21,FF23,FF24,\
               FF30,FF31,FF32,FF34,\
               FF40,FF41,FF42,FF43,\
               FB01,FB02,FB03,FB04,\
               FB10,FB12,FB13,FB14,\
               FB20,FB21,FB23,FB24,\
               FB30,FB31,FB32,FB34,\
               FB40,FB41,FB42,FB43

