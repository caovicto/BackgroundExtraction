import numpy as np
import os
import torch
from torch.nn import functional
from Utilities.warp_utils import dense_image_warp
import cv2
from Models.BackgroundReconstruction import ImageReconstruction
from Models.Decomposition_Net import Decomp_Net_Translation
from Models import FlowDecomposer
from matplotlib import pyplot as plt

def test_train(train=False):
    batch_size = 1
    #read in images
    ### needs to be finished ###
    I0 = cv2.imread("../Files/SyntheticDataset/Places365_val_00000969/Places365_val_00000969_f0.png").astype(np.float32)[..., :-1] / 255.0
    I1 = cv2.imread("../Files/SyntheticDataset/Places365_val_00000969/Places365_val_00000969_f1.png").astype(np.float32)[..., :-1] / 255.0
    I2 = cv2.imread("../Files/SyntheticDataset/Places365_val_00000969/Places365_val_00000969_f2.png").astype(np.float32)[..., :-1] / 255.0
    I3 = cv2.imread("../Files/SyntheticDataset/Places365_val_00000969/Places365_val_00000969_f3.png").astype(np.float32)[..., :-1] / 255.0
    I4 = cv2.imread("../Files/SyntheticDataset/Places365_val_00000969/Places365_val_00000969_f4.png").astype(np.float32)[..., :-1] / 255.0
    Original_H = I0.shape[0]
    Original_W = I0.shape[1]
    Resized_H = int(np.ceil(float(Original_H)*1/16.0))*16
    Resized_W = int(np.ciel(float(Original_W)*1/16.0))*16

    I0 = np.expand_dims(cv2.resize(I0,dsize=(Resized_W,Resized_H),interpolation=cv2.INTER_CUBIC),0)
    I1 = np.expand_dims(cv2.resize(I1,dsize=(Resized_W,Resized_H),interpolation=cv2.INTER_CUBIC),0)
    I2 = np.expand_dims(cv2.resize(I2,dsize=(Resized_W,Resized_H),interpolation=cv2.INTER_CUBIC),0)
    I3 = np.expand_dims(cv2.resize(I3,dsize=(Resized_W,Resized_H),interpolation=cv2.INTER_CUBIC),0)
    I4 = np.expand_dims(cv2.resize(I4,dsize=(Resized_W,Resized_H),interpolation=cv2.INTER_CUBIC),0)

    Crop_Patch_H = Resized_H
    Crop_Patch_W = Resized_W

    def flow_to_image(flow):
        flowmag = torch.sqrt(1e-6+flow[...,0]**2+flow[...,1]**2)
        flowang = torch.atan2(flow[...,0],flow[...,1])
        hsv0 = ((flowang/np.pi)+1)/2
        hsv1 = (flowmag-torch.min(torch.min(flowmag,dim=1,keepdim=True),dim=2,keepdim=True))     \
               /(1e-6+torch.max(torch.min(flowmag,dim=1,keepdim=True),dim=2,keepdim=True)-torch.min(torch.max(flowmag,dim=1,keepdim=True),dim=2,keepdim=True))
        hsv2 = torch.ones(hsv1.shape)
        hsv = torch.stack([hsv0,hsv1,hsv2],-1)
        #based on hsv2rgb function found at https://memotut.com/convert-rgb-and-hsv-in-a-differentiable-form-with-pytorch-20819/
        h = hsv[:,0]
        s = hsv[:,1]
        v = hsv[:,2]
        h = (h-torch.floor(h/360)*360)/60
        c = s*v
        x = c*(1-torch.abs(torch.fmod(h,2)-1))
        zero = torch.zeros_like(c)
        mat = torch.stack((torch.stack((c,x,zero),dim=1),
                           torch.stack((x,c,zero),dim=1),
                           torch.stack((zero,c,x),dim=1),
                           torch.stack((zero,x,c),dim=1),
                           torch.stack((x,zero,c),dim=1),
                           torch.stack((c,zero,x),dim=1)),dim=0)
        indx = torch.repeat_interleave(torch.floor(h).unsqueeze(1),3,dim=1).unsqueeze(0).to(torch.long)
        rgb = (mat.gather(dim=0,index=indx)+(v-c)).squeeze(0)
        return rgb

    def warp(img,F):
        return torch.reshape(dense_image_warp(img,torch.stack([-F[...,1],-F[...,0]],-1)),[batch_size,Crop_Patch_H,Crop_Patch_W,3])

    fused_frame0 = torch.tensor(I0,torch.float32)
    fused_frame1 = torch.tensor(I1,torch.float32)
    fused_frame2 = torch.tensor(I2,torch.float32)
    fused_frame3 = torch.tensor(I3,torch.float32)
    fused_frame4 = torch.tensor(I4,torch.float32)

    fuse = torch.nn.Upsample([192,320],mode='bilinear')
    fused_frame0_small = fuse(fused_frame0)
    fused_frame1_small = fuse(fused_frame1)
    fused_frame2_small = fuse(fused_frame2)
    fused_frame3_small = fuse(fused_frame3)
    fused_frame4_small = fuse(fused_frame4)

    ### add support for pretrained network ###
    def pretrained_net(F0,F1,F2,F3,F4,B0,B1,B2,B3,B4,lvlh,lvlw,pretrainedh,pretrainedw):
        ratioh = float(lvlh)/float(pretrainedh)
        ratiow = float(lvlw)/float(pretrainedw)
        nn = FlowDecomposer.Network() ### NEEDS TO BE CHANGED TO TAKE IN THE FLOW NETWORK ###
        pretrained_scale = torch.nn.Upsample((pretrained_h,pretrained_w),mode='bilinear')
        F0 = pretrained_scale(F0)
        F1 = pretrained_scale(F1)
        F2 = pretrained_scale(F2)
        F3 = pretrained_scale(F3)
        F4 = pretrained_scale(F4)
        B0 = pretrained_scale(B0)
        B1 = pretrained_scale(B1)
        B2 = pretrained_scale(B2)
        B3 = pretrained_scale(B3)
        B4 = pretrained_scale(B4)
        temp = []
        temp.append(torch.stack([F0,F1],1))
        temp.append(torch.stack([F0,F2],1))
        temp.append(torch.stack([F0,F3],1))
        temp.append(torch.stack([F0,F4],1))
        temp.append(torch.stack([F1,F0],1))
        temp.append(torch.stack([F1,F2],1))
        temp.append(torch.stack([F1,F3],1))
        temp.append(torch.stack([F1,F4],1))
        temp.append(torch.stack([F2,F0],1))
        temp.append(torch.stack([F2,F1],1))
        temp.append(torch.stack([F2,F3],1))
        temp.append(torch.stack([F2,F4],1))
        temp.append(torch.stack([F3,F0],1))
        temp.append(torch.stack([F3,F1],1))
        temp.append(torch.stack([F3,F2],1))
        temp.append(torch.stack([F3,F4],1))
        temp.append(torch.stack([F4,F0],1))
        temp.append(torch.stack([F4,F1],1))
        temp.append(torch.stack([F4,F2],1))
        temp.append(torch.stack([F4,F3],1))
        temp.append(torch.stack([B0,B1],1))
        temp.append(torch.stack([B0,B2],1))
        temp.append(torch.stack([B0,B3],1))
        temp.append(torch.stack([B0,B4],1))
        temp.append(torch.stack([B1,B0],1))
        temp.append(torch.stack([B1,B2],1))
        temp.append(torch.stack([B1,B3],1))
        temp.append(torch.stack([B1,B4],1))
        temp.append(torch.stack([B2,B0],1))
        temp.append(torch.stack([B2,B1],1))
        temp.append(torch.stack([B2,B3],1))
        temp.append(torch.stack([B2,B4],1))
        temp.append(torch.stack([B3,B0],1))
        temp.append(torch.stack([B3,B1],1))
        temp.append(torch.stack([B3,B2],1))
        temp.append(torch.stack([B3,B4],1))
        temp.append(torch.stack([B4,B0],1))
        temp.append(torch.stack([B4,B1],1))
        temp.append(torch.stack([B4,B2],1))
        temp.append(torch.stack([B4,B3],1))

        pretrainedin = torch.cat(temp,0)
        pretrainedin = torch.reshape(pretrainedin,[batch_size*40,2,pretrainedh,pretrainedw,3])
        pred_labels,_ = nn.nn(pretrainedin,reuse=torch.AUTO_REUSE) ### need to change to match the pretrained net ###

        labelsup = torch.nn.Upsample((lvlh,lvlw),pred_labels)

        ratio_tensor = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(torch.from_numpy(np.asarray([ratiow,ratioh])).float(),0),0),0)
        FF01 = pred_labels[batch_size*0:batch_size*1]*ratio_tensor
        FF02 = pred_labels[batch_size*1:batch_size*2]*ratio_tensor
        FF03 = pred_labels[batch_size*2:batch_size*3]*ratio_tensor
        FF04 = pred_labels[batch_size*3:batch_size*4]*ratio_tensor
        FF10 = pred_labels[batch_size*4:batch_size*5]*ratio_tensor
        FF12 = pred_labels[batch_size*5:batch_size*6]*ratio_tensor
        FF13 = pred_labels[batch_size*6:batch_size*7]*ratio_tensor
        FF14 = pred_labels[batch_size*7:batch_size*8]*ratio_tensor
        FF20 = pred_labels[batch_size*8:batch_size*9]*ratio_tensor
        FF21 = pred_labels[batch_size*9:batch_size*10]*ratio_tensor
        FF23 = pred_labels[batch_size*10:batch_size*11]*ratio_tensor
        FF24 = pred_labels[batch_size*11:batch_size*12]*ratio_tensor
        FF30 = pred_labels[batch_size*12:batch_size*13]*ratio_tensor
        FF31 = pred_labels[batch_size*13:batch_size*14]*ratio_tensor
        FF32 = pred_labels[batch_size*14:batch_size*15]*ratio_tensor
        FF34 = pred_labels[batch_size*15:batch_size*16]*ratio_tensor
        FF40 = pred_labels[batch_size*16:batch_size*17]*ratio_tensor
        FF41 = pred_labels[batch_size*17:batch_size*18]*ratio_tensor
        FF42 = pred_labels[batch_size*18:batch_size*19]*ratio_tensor
        FF43 = pred_labels[batch_size*19:batch_size*20]*ratio_tensor
        FB01 = pred_labels[batch_size*20:batch_size*21]*ratio_tensor
        FB02 = pred_labels[batch_size*21:batch_size*22]*ratio_tensor
        FB03 = pred_labels[batch_size*22:batch_size*23]*ratio_tensor
        FB04 = pred_labels[batch_size*23:batch_size*24]*ratio_tensor
        FB10 = pred_labels[batch_size*24:batch_size*25]*ratio_tensor
        FB12 = pred_labels[batch_size*25:batch_size*26]*ratio_tensor
        FB13 = pred_labels[batch_size*26:batch_size*27]*ratio_tensor
        FB14 = pred_labels[batch_size*27:batch_size*28]*ratio_tensor
        FB20 = pred_labels[batch_size*28:batch_size*29]*ratio_tensor
        FB21 = pred_labels[batch_size*29:batch_size*30]*ratio_tensor
        FB23 = pred_labels[batch_size*30:batch_size*31]*ratio_tensor
        FB24 = pred_labels[batch_size*31:batch_size*32]*ratio_tensor
        FB30 = pred_labels[batch_size*32:batch_size*33]*ratio_tensor
        FB31 = pred_labels[batch_size*33:batch_size*34]*ratio_tensor
        FB32 = pred_labels[batch_size*34:batch_size*35]*ratio_tensor
        FB34 = pred_labels[batch_size*35:batch_size*36]*ratio_tensor
        FB40 = pred_labels[batch_size*36:batch_size*37]*ratio_tensor
        FB41 = pred_labels[batch_size*37:batch_size*38]*ratio_tensor
        FB42 = pred_labels[batch_size*38:batch_size*39]*ratio_tensor
        FB43 = pred_labels[batch_size*39:batch_size*40]*ratio_tensor
        if train:
            FF01.requires_grad = False
            FF02.requires_grad = False
            FF03.requires_grad = False
            FF04.requires_grad = False
            FF10.requires_grad = False
            FF12.requires_grad = False
            FF13.requires_grad = False
            FF14.requires_grad = False
            FF20.requires_grad = False
            FF21.requires_grad = False
            FF23.requires_grad = False
            FF24.requires_grad = False
            FF30.requires_grad = False
            FF31.requires_grad = False
            FF32.requires_grad = False
            FF34.requires_grad = False
            FF40.requires_grad = False
            FF41.requires_grad = False
            FF42.requires_grad = False
            FF43.requires_grad = False
            FB01.requires_grad = False
            FB02.requires_grad = False
            FB03.requires_grad = False
            FB04.requires_grad = False
            FB10.requires_grad = False
            FB12.requires_grad = False
            FB13.requires_grad = False
            FB14.requires_grad = False
            FB20.requires_grad = False
            FB21.requires_grad = False
            FB23.requires_grad = False
            FB24.requires_grad = False
            FB30.requires_grad = False
            FB31.requires_grad = False
            FB32.requires_grad = False
            FB34.requires_grad = False
            FB40.requires_grad = False
            FB41.requires_grad = False
            FB42.requires_grad = False
            FB43.requires_grad = False


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

    model = Decomp_Net_Translation(Crop_Patch_W//16,Crop_Patch_W//16,False,False,False)
    FF01_4,FF02_4,FF03_4,FF04_4,\
    FF10_4,FF12_4,FF13_4,FF14_4,\
    FF20_4,FF21_4,FF23_4,FF24_4,\
    FF30_4,FF31_4,FF32_4,FF34_4,\
    FF40_4,FF41_4,FF42_4,FF43_4,\
    FB01_4,FB02_4,FB03_4,FB04_4,\
    FB10_4,FB12_4,FB13_4,FB14_4,\
    FB20_4,FB21_4,FB23_4,FB24_4,\
    FB30_4,FB31_4,FB32_4,FB34_4,\
    FB40_4,FB41_4,FB42_4,FB43_4 = model.inference(fused_frame0,fused_frame1,fused_frame2,fused_frame3,fused_frame4)

    flows = []
    flows.append((FB01_4,FB02_4,FB03_4,FB04_4,
              FB10_4,FB12_4,FB13_4,FB14_4,
              FB20_4,FB21_4,FB23_4,FB24_4,
              FB30_4,FB31_4,FB32_4,FB34_4,
              FB40_4,FB41_4,FB42_4,FB43_4))

    #image
    model4 = ImageReconstruction(batch_size,Crop_Patch_H,Crop_Patch_W,level=4,weighted_fusion=False)
    B0_pred4,B1_pred4,B2_pred4,B3_pred4,B4_pred4,\
    A0_pred4,A1_pred4,A2_pred4,A3_pred4,A4_pred4 = model4.build_model(torch.cat([fused_frame0,fused_frame1,fused_frame2,fused_frame3,fused_frame4],3),
                                                                  None,None,None,None,None,
                                                                  None,None,None,None,None,
                                                                  FB01_4, FB02_4, FB03_4, FB04_4,
                                                                  FB10_4, FB12_4, FB13_4, FB14_4,
                                                                  FB20_4, FB21_4, FB23_4, FB24_4,
                                                                  FB30_4, FB31_4, FB32_4, FB34_4,
                                                                  FB40_4, FB41_4, FB42_4, FB43_4)
    #upsample since model doesn't resize
    up4 = torch.nn.Upsample([Crop_Patch_H//(2**3),Crop_Patch_W//(2**3)],mode='bilinear')
    B0_pred4_up = up4(B0_pred4)
    B1_pred4_up = up4(B1_pred4)
    B2_pred4_up = up4(B2_pred4)
    B3_pred4_up = up4(B3_pred4)
    B4_pred4_up = up4(B4_pred4)
    A0_pred4_up = up4(A0_pred4)
    A1_pred4_up = up4(A1_pred4)
    A2_pred4_up = up4(A2_pred4)
    A3_pred4_up = up4(A3_pred4)
    A4_pred4_up = up4(A4_pred4)
    _,_,_,_,\
    _,_,_,_,\
    _,_,_,_,\
    _,_,_,_,\
    _,_,_,_,\
    FB01_3,FB02_3,FB03_3,FB04_3,\
    FB10_3,FB12_3,FB13_3,FB14_3,\
    FB20_3,FB21_3,FB23_3,FB24_3,\
    FB30_3,FB31_3,FB32_3,FB34_3,\
    FB40_3,FB41_3,FB42_3,FB43_3 = pretrained_net(B0_pred4_up,B1_pred4_up,B2_pred4_up,B3_pred4_up,B4_pred4_up,
                                     B0_pred4_up,B1_pred4_up,B2_pred4_up,B3_pred4_up,B4_pred4_up,
                                     Crop_Patch_H//(2**3),Crop_Patch_W//(2**3),
                                     int(np.ceil(float(Crop_Patch_H//(2**3))/64.0))*64,
                                     int(np.ceil(float(Crop_Patch_W//(2**3))/64.0))*64)

    flows.append((FB01_3,FB02_3,FB03_3,FB04_3,
              FB10_3,FB12_3,FB13_3,FB14_3,
              FB20_3,FB21_3,FB23_3,FB24_3,
              FB30_3,FB31_3,FB32_3,FB34_3,
              FB40_3,FB41_3,FB42_3,FB43_3))

#call implmentation of call to pretrained net

    #level 3
    model3 = ImageReconstruction(batch_size,Crop_Patch_H,Crop_Patch_W,level=3,weighted_fusion=False)
    B0_pred3,B1_pred3,B2_pred3,B3_pred3,B4_pred3,\
    A0_pred3,A1_pred3,A2_pred3,A3_pred3,A4_pred3 = model3.build_model(torch.cat([fused_frame0,fused_frame1,fused_frame2,fused_frame3,fused_frame4],3),
                                                                  B0_pred4_up,B1_pred4_up,B2_pred4_up,B3_pred4_up,B4_pred4_up,
                                                                  A0_pred4_up,A1_pred4_up,A2_pred4_up,A3_pred4_up,A4_pred4_up,
                                                                  FB01_3, FB02_3, FB03_3, FB04_3,
                                                                  FB10_3, FB12_3, FB13_3, FB14_3,
                                                                  FB20_3, FB21_3, FB23_3, FB24_3,
                                                                  FB30_3, FB31_3, FB32_3, FB34_3,
                                                                  FB40_3, FB41_3, FB42_3, FB43_3)
    #upsample since model doesn't resize
    up3 = torch.nn.Upsample([Crop_Patch_H//(2**2),Crop_Patch_W//(2**2)],mode='bilinear')
    B0_pred3_up = up3(B0_pred3)
    B1_pred3_up = up3(B1_pred3)
    B2_pred3_up = up3(B2_pred3)
    B3_pred3_up = up3(B3_pred3)
    B4_pred3_up = up3(B4_pred3)
    A0_pred3_up = up3(A0_pred3)
    A1_pred3_up = up3(A1_pred3)
    A2_pred3_up = up3(A2_pred3)
    A3_pred3_up = up3(A3_pred3)
    A4_pred3_up = up3(A4_pred3)
    _, _, _, _, \
    _, _, _, _, \
    _, _, _, _, \
    _, _, _, _, \
    _, _, _, _,\
    FB01_2,FB02_2,FB03_2,FB04_2,\
    FB10_2,FB12_2,FB13_2,FB14_2,\
    FB20_2,FB21_2,FB23_2,FB24_2,\
    FB30_2,FB31_2,FB32_2,FB34_2,\
    FB40_2,FB41_2,FB42_2,FB43_2 = pretrained_net(B0_pred3_up,B1_pred3_up,B2_pred3_up,B3_pred3_up,B4_pred3_up,
                                     B0_pred3_up,B1_pred3_up,B2_pred3_up,B3_pred3_up,B4_pred3_up,
                                     Crop_Patch_H//(2**3),Crop_Patch_W//(2**3),
                                     int(np.ceil(float(Crop_Patch_H//(2**3))/64.0))*64,
                                     int(np.ceil(float(Crop_Patch_W//(2**3))/64.0))*64)
    flows.append((FB01_2,FB02_2,FB03_2,FB04_2,
              FB10_2,FB12_2,FB13_2,FB14_2,
              FB20_2,FB21_2,FB23_2,FB24_2,
              FB30_2,FB31_2,FB32_2,FB34_2,
              FB40_2,FB41_2,FB42_2,FB43_2))

#call implmentation of call to pretrained net

    #level 2
    model2 = ImageReconstruction(batch_size,Crop_Patch_H,Crop_Patch_W,level=2,weighted_fusion=False)
    B0_pred2,B1_pred2,B2_pred2,B3_pred2,B4_pred2,\
    A0_pred2,A1_pred2,A2_pred2,A3_pred2,A4_pred2 = model3.build_model(torch.cat([fused_frame0,fused_frame1,fused_frame2,fused_frame3,fused_frame4],3),
                                                                  B0_pred3_up,B1_pred3_up,B2_pred3_up,B3_pred3_up,B4_pred3_up,
                                                                  A0_pred3_up,A1_pred3_up,A2_pred3_up,A3_pred3_up,A4_pred3_up,
                                                                  FB01_2, FB02_2, FB03_2, FB04_2,
                                                                  FB10_2, FB12_2, FB13_2, FB14_2,
                                                                  FB20_2, FB21_2, FB23_2, FB24_2,
                                                                  FB30_2, FB31_2, FB32_2, FB34_2,
                                                                  FB40_2, FB41_2, FB42_2, FB43_2)
    #upsample since model doesn't resize
    up2 = torch.nn.Upsample([Crop_Patch_H//(2**1),Crop_Patch_W//(2**1)],mode='bilinear')
    B0_pred2_up = up2(B0_pred2)
    B1_pred2_up = up2(B1_pred2)
    B2_pred2_up = up2(B2_pred2)
    B3_pred2_up = up2(B3_pred2)
    B4_pred2_up = up2(B4_pred2)
    A0_pred2_up = up2(A0_pred2)
    A1_pred2_up = up2(A1_pred2)
    A2_pred2_up = up2(A2_pred2)
    A3_pred2_up = up2(A3_pred2)
    A4_pred2_up = up2(A4_pred2)
    _, _, _, _, \
    _, _, _, _, \
    _, _, _, _, \
    _, _, _, _, \
    _, _, _, _, \
    FB01_1,FB02_1,FB03_1,FB04_1,\
    FB10_1,FB12_1,FB13_1,FB14_1,\
    FB20_1,FB21_1,FB23_1,FB24_1,\
    FB30_1,FB31_1,FB32_1,FB34_1,\
    FB40_1,FB41_1,FB42_1,FB43_1 = pretrained_net(B0_pred2_up,B1_pred2_up,B2_pred2_up,B3_pred2_up,B4_pred2_up,
                                     B0_pred2_up,B1_pred2_up,B2_pred2_up,B3_pred2_up,B4_pred2_up,
                                     Crop_Patch_H//(2**3),Crop_Patch_W//(2**3),
                                     int(np.ceil(float(Crop_Patch_H//(2**3))/64.0))*64,
                                     int(np.ceil(float(Crop_Patch_W//(2**3))/64.0))*64)
    flows.append((FB01_1,FB02_1,FB03_1,FB04_1,
              FB10_1,FB12_1,FB13_1,FB14_1,
              FB20_1,FB21_1,FB23_1,FB24_1,
              FB30_1,FB31_1,FB32_1,FB34_1,
              FB40_1,FB41_1,FB42_1,FB43_1))

#call implmentation of call to pretrained net

    #level 1
    model1 = ImageReconstruction(batch_size,Crop_Patch_H,Crop_Patch_W,level=1,weighted_fusion=False)
    B0_pred1,B1_pred1,B2_pred1,B3_pred1,B4_pred1,\
    A0_pred1,A1_pred1,A2_pred1,A3_pred1,A4_pred1 = model1.build_model(torch.cat([fused_frame0,fused_frame1,fused_frame2,fused_frame3,fused_frame4],3),
                                                                  B0_pred2_up,B1_pred2_up,B2_pred2_up,B3_pred2_up,B4_pred2_up,
                                                                  A0_pred2_up,A1_pred2_up,A2_pred2_up,A3_pred2_up,A4_pred2_up,
                                                                  FB01_1, FB02_1, FB03_1, FB04_1,
                                                                  FB10_1, FB12_1, FB13_1, FB14_1,
                                                                  FB20_1, FB21_1, FB23_1, FB24_1,
                                                                  FB30_1, FB31_1, FB32_1, FB34_1,
                                                                  FB40_1, FB41_1, FB42_1, FB43_1)
    #upsample since model doesn't resize
    up1 = torch.nn.Upsample([Crop_Patch_H//(2**0),Crop_Patch_W//(2**0)],mode='bilinear')
    B0_pred1_up = up1(B0_pred1)
    B1_pred1_up = up1(B1_pred1)
    B2_pred1_up = up1(B2_pred1)
    B3_pred1_up = up1(B3_pred1)
    B4_pred1_up = up1(B4_pred1)
    A0_pred1_up = up1(A0_pred1)
    A1_pred1_up = up1(A1_pred1)
    A2_pred1_up = up1(A2_pred1)
    A3_pred1_up = up1(A3_pred1)
    A4_pred1_up = up1(A4_pred1)
    _, _, _, _, \
    _, _, _, _, \
    _, _, _, _, \
    _, _, _, _, \
    _, _, _, _,\
    FB01_0,FB02_0,FB03_0,FB04_0,\
    FB10_0,FB12_0,FB13_0,FB14_0,\
    FB20_0,FB21_0,FB23_0,FB24_0,\
    FB30_0,FB31_0,FB32_0,FB34_0,\
    FB40_0,FB41_0,FB42_0,FB43_0 = pretrained_net(B0_pred1_up,B1_pred1_up,B2_pred1_up,B3_pred1_up,B4_pred1_up,
                                     B0_pred1_up,B1_pred1_up,B2_pred1_up,B3_pred1_up,B4_pred1_up,
                                     Crop_Patch_H//(2**3),Crop_Patch_W//(2**3),
                                     int(np.ceil(float(Crop_Patch_H//(2**3))/64.0))*64,
                                     int(np.ceil(float(Crop_Patch_W//(2**3))/64.0))*64)
    flows.append((FB01_0,FB02_0,FB03_0,FB04_0,
              FB10_0,FB12_0,FB13_0,FB14_0,
              FB20_0,FB21_0,FB23_0,FB24_0,
              FB30_0,FB31_0,FB32_0,FB34_0,
              FB40_0,FB41_0,FB42_0,FB43_0))

#call implmentation of call to pretrained net

    #level 0
    model0 = ImageReconstruction(batch_size,Crop_Patch_H,Crop_Patch_W,level=0,weighted_fusion=False)
    B0_pred0,B1_pred0,B2_pred0,B3_pred0,B4_pred0,\
    A0_pred0,A1_pred0,A2_pred0,A3_pred0,A4_pred0 = model0.build_model(torch.cat([fused_frame0,fused_frame1,fused_frame2,fused_frame3,fused_frame4],3),
                                                                  B0_pred1_up,B1_pred1_up,B2_pred1_up,B3_pred1_up,B4_pred1_up,
                                                                  A0_pred1_up,A1_pred1_up,A2_pred1_up,A3_pred1_up,A4_pred1_up,
                                                                  FB01_0, FB02_0, FB03_0, FB04_0,
                                                                  FB10_0, FB12_0, FB13_0, FB14_0,
                                                                  FB20_0, FB21_0, FB23_0, FB24_0,
                                                                  FB30_0, FB31_0, FB32_0, FB34_0,
                                                                  FB40_0, FB41_0, FB42_0, FB43_0)
    B_pred = []
    B_pred.append(torch.cat([B0_pred0,B1_pred0,B2_pred0,B3_pred0,B4_pred0],-1))
    B_pred.append(torch.cat([B0_pred1,B1_pred1,B2_pred1,B3_pred1,B4_pred1],-1))
    B_pred.append(torch.cat([B0_pred1,B1_pred1,B2_pred1,B3_pred1,B4_pred1],-1))
    B_pred.append(torch.cat([B0_pred1,B1_pred1,B2_pred1,B3_pred1,B4_pred1],-1))
    B_pred.append(torch.cat([B0_pred1,B1_pred1,B2_pred1,B3_pred1,B4_pred1],-1))
    A_pred = []
    A_pred.append(torch.cat([A0_pred0,A1_pred0,A2_pred0,A3_pred0,A4_pred0],-1))
    A_pred.append(torch.cat([A0_pred1,A1_pred1,A2_pred1,A3_pred1,A4_pred1],-1))
    A_pred.append(torch.cat([A0_pred1,A1_pred1,A2_pred1,A3_pred1,A4_pred1],-1))
    A_pred.append(torch.cat([A0_pred1,A1_pred1,A2_pred1,A3_pred1,A4_pred1],-1))
    A_pred.append(torch.cat([A0_pred1,A1_pred1,A2_pred1,A3_pred1,A4_pred1],-1))

    #post processing weighted fusion
    def outgoing_mask(flow):
        #creates a mask that is zero at all positions where the flow would carry a pixel over the image boundary
        # gets the desired dimensions
        num_batch, height, width, _ = flow.shape

        # create the x grid and y grid to construct the mask on
        gridx = torch.reshape(torch.range(0, width - 1), [1, 1, width])
        gridx = torch.from_numpy(np.tile(gridx, [num_batch, height, 1]))
        gridy = torch.reshape(torch.range(0, height - 1), [1, height, 1])
        gridy = torch.from_numpy(np.tile(gridy, [num_batch, 1, width]))

        # create the mask
        flowu, flowv = torch.unbind(flow, 3)[0:2]
        posx = gridx.float() + flowu
        posy = gridy.float() + flowv
        insidex = torch.logical_and(posx <= (width - 1), posx >= 0)
        insidey = torch.logical_and(posy <= (height - 1), posy >= 0)
        inside = torch.logical_and(insidex, insidey)
        return torch.unsqueeze(inside.type(torch.FloatTensor), 3)

    def warp_with_large_size(img,F,c):
        return torch.reshape(dense_image_warp(img,torch.stack([-F[...,1],-F[...,0]],-1)),[batch_size,Resized_H,Resized_W,c])

    def generate_gaussian_kernel(size):
        kernel = cv2.getGaussianKernel(size,0)
        kernel = kernel@kernel.T
        return kernel[:,:,np.newaxis,np.newaxis].type(torch.float32)
    kernel = generate_gaussian_kernel(21)
    def apply_gaussian_blur(x):
        x = functional.pad(x,(0,0,40,40,40,40,0,0),'reflect')
        ### need to figure out # in channels and # out channels ###
        x0 = torch.nn.Conv2d(in_channels=x.shape[1],out_channels=x.shape[1],kernel_size=kernel.size)(x[...,0:1])
        x1 = torch.nn.Conv2d(in_channels=x.shape[1],out_channels=x.shape[1],kernel_size=kernel.size)(x[...,1:2])
        x2 = torch.nn.Conv2d(in_channels=x.shape[1],out_channels=x.shape[1],kernel_size=kernel.size)(x[...,2:3])
        out = torch.cat([x0,x1,x2],-1)
        return out

    downlevel = 0
    pretrained_h = int(np.ceil(float(Crop_Patch_H//(2**downlevel))/64))*64
    pretrained_w = int(np.ceil(float(Crop_Patch_W//(2**downlevel))/64))*64
    down = torch.nn.Upsample((pretrained_h,pretrained_w),align_corners=True)
    B0_pred0_down = down(apply_gaussian_blur(B0_pred0))
    B1_pred0_down = down(apply_gaussian_blur(B1_pred0))
    B2_pred0_down = down(apply_gaussian_blur(B2_pred0))
    B3_pred0_down = down(apply_gaussian_blur(B3_pred0))
    B4_pred0_down = down(apply_gaussian_blur(B4_pred0))

    ### right place to move into loss calc? ###
    if train:
        _, _, _, _, \
        _, _, _, _, \
        _, _, _, _, \
        _, _, _, _, \
        _, _, _, _, \
        FB01_, FB02_, FB03_, FB04_, \
        FB10_, FB12_, FB13_, FB14_, \
        FB20_, FB21_, FB23_, FB24_, \
        FB30_, FB31_, FB32_, FB34_, \
        FB40_, FB41_, FB42_, FB43_ = pretrained_net(B0_pred0_down, B1_pred0_down, B2_pred0_down, B3_pred0_down, B4_pred0_down,
                                                        B0_pred0_down, B1_pred0_down, B2_pred0_down, B3_pred0_down, B4_pred0_down,
                                                        Crop_Patch_H // (2 ** 0), Crop_Patch_W // (2 ** 0),
                                                        int(np.ceil(float(Crop_Patch_H // (2 ** 0)) / 64.0)) * 64,
                                                        int(np.ceil(float(Crop_Patch_W // (2 ** 0)) / 64.0)) * 64)
        loss = 0
        loss_weight = [1.0,1.0,1.0,1.0,1.0]
        for i in range(len(B_pred)):
            _,h,w,_ = B_pred[i]
            print('level: '+str(i))
            print(h)
            print(w)
            lvlimg = torch.nn.Upsample((h,w))
            I0lvl = lvlimg(fused_frame0)
            I1lvl = lvlimg(fused_frame1)
            I2lvl = lvlimg(fused_frame2)
            I3lvl = lvlimg(fused_frame3)
            I4lvl = lvlimg(fused_frame4)

            def compute_loss(FB02,FB12,FB32,FB42,B2,A2,I2lvl,I0lvl,I1lvl,I3lvl,I4lvl):
                subloss = 0
                cvtflow2lvl = torch.nn.Upsample([h,w])
                FB02 = cvtflow2lvl(FB02/(2**i))
                FB12 = cvtflow2lvl(FB12/(2**i))
                FB32 = cvtflow2lvl(FB32/(2**i))
                FB42 = cvtflow2lvl(FB42/(2**i))

                #warping loss
                subloss += (loss_weight[i]*torch.mean(torch.abs(I0lvl-warp(B2,FB02))))
                subloss += (loss_weight[i]*torch.mean(torch.abs(I1lvl-warp(B2,FB12))))
                subloss += (loss_weight[i]*torch.mean(torch.abs(I2lvl-B2-A2)))
                subloss += (loss_weight[i]*torch.mean(torch.abs(I3lvl-warp(B2,FB32))))
                subloss += (loss_weight[i]*torch.mean(torch.abs(I4lvl-warp(B2,FB42))))

                #TV loss
                subloss += (loss_weight[i]*(.1*torch.mean(torch.abs(B2[:,1:]-B2[:,:-1]))))
                subloss += (loss_weight[i]*(.1*torch.mean(torch.abs(B2[:,:,1:]-B2[:,:,:,-1]))))
                subloss += (loss_weight[i]*(.1*torch.mean(torch.abs(A2[:,1:]-A2[:,:-1]))))
                subloss += (loss_weight[i]*(.1*torch.mean(torch.abs(A2[:,:,1:]-A2[:,:,:-1]))))

                return subloss
            loss += compute_loss(FB02_,FB12_,FB32_,FB42_,
                                 torch.clamp(B_pred[i][...,6:9],0.0,1.0),torch.clamp(A_pred[i][...,6:9],0.0,1.0),
                                 I2lvl,I0lvl,I1lvl,I3lvl,I4lvl)
            loss += compute_loss(FB10_, FB20_, FB30_, FB40_,
                                 torch.clamp(B_pred[i][..., 0:3], 0.0, 1.0), torch.clamp(A_pred[i][..., 0:3], 0.0, 1.0),
                                 I0lvl, I1lvl, I2lvl, I3lvl, I4lvl)
            loss += compute_loss(FB01_, FB21_, FB31_, FB41_,
                                 torch.clamp(B_pred[i][..., 3:6], 0.0, 1.0), torch.clamp(A_pred[i][..., 3:6], 0.0, 1.0),
                                 I1lvl, I0lvl, I2lvl, I3lvl, I4lvl)
            loss += compute_loss(FB03_, FB13_, FB23_, FB43_,
                                 torch.clamp(B_pred[i][..., 9:12], 0.0, 1.0), torch.clamp(A_pred[i][..., 9:12], 0.0, 1.0),
                                 I3lvl, I0lvl, I1lvl, I2lvl, I4lvl)
            loss += compute_loss(FB04_, FB14_, FB24_, FB34_,
                                 torch.clamp(B_pred[i][..., 12:15], 0.0, 1.0), torch.clamp(A_pred[i][..., 12:15], 0.0, 1.0),
                                 I4lvl, I0lvl, I1lvl, I2lvl, I3lvl)
        return loss, model0




    ratioh = float(Crop_Patch_H//(2**downlevel))/float(pretrained_h)
    ratiow = float(Crop_Patch_W//(2**downlevel))/float(pretrained_w)
    #call pretrained net in test mode
    ### needs to be changed to accomadate the pretrained network ###
    nn = ModelPWCNet(mode='test',options=nn_opts)
    nn.print_config()

    temp = []
    temp.append(torch.stack([B2_pred0_down,B0_pred0_down],1))
    temp.append(torch.stack([B2_pred0_down,B1_pred0_down],1))
    temp.append(torch.stack([B2_pred0_down,B3_pred0_down],1))
    temp.append(torch.stack([B2_pred0_down,B4_pred0_down],1))

    pretrained_in = torch.cat(temp,0)
    pretrained_in = torch.reshape(pretrained_in,[batch_size*4,2,pretrained_h,pretrained_w,3])
    pred_labels,_ = nn.nn(pretrained_in,reuse=torch.Auto_REUSE) #need to change to accomadate our pretrained network
    uplabels = torch.nn.Upsample((Resized_H//(2**downlevel),Resized_W//(2**downlevel)),align_corners=True)
    pred_labels = uplabels(pred_labels)
    #0:W 1:H
    ratio_tensor = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(torch.from_numpy(np.asarray([ratiow,ratioh])).float(),0),0),0)
    FB20 = pred_labels[batch_size*0:batch_size*1]*ratio_tensor*(2**downlevel)
    FB21 = pred_labels[batch_size*1:batch_size*2]*ratio_tensor*(2**downlevel)
    FB23 = pred_labels[batch_size*2:batch_size*3]*ratio_tensor*(2**downlevel)
    FB24 = pred_labels[batch_size*3:batch_size*4]*ratio_tensor*(2**downlevel)

    fb = torch.nn.Upsample((Resized_H,Resized_W),align_corners=True)
    FB20 = fb(FB20)
    FB21 = fb(FB21)
    FB23 = fb(FB23)
    FB24 = fb(FB24)

    def dilation(x):
        kernel = torch.ones((5,5,1))
        out = torch.nn.Conv2d(x.shape[1],x.shape[1],kernel.shape(),dilation=(1,1,1,1))(x)
        return x-torch.ones_like(x)

    II0 = warp_with_large_size(fused_frame0,FB20,3)
    II1 = warp_with_large_size(fused_frame1,FB21,3)
    II3 = warp_with_large_size(fused_frame3,FB23,3)
    II4 = warp_with_large_size(fused_frame4,FB24,3)
    AA0 = warp_with_large_size(torch.clamp(dilation(A0_pred0),0.0,1.0),FB20,1)
    AA1 = warp_with_large_size(torch.clamp(dilation(A1_pred0),0.0,1.0),FB21,1)
    AA3 = warp_with_large_size(torch.clamp(dilation(A3_pred0),0.0,1.0),FB23,1)
    AA4 = warp_with_large_size(torch.clamp(dilation(A4_pred0),0.0,1.0),FB24,1)
    FB20_mask = outgoing_mask(FB20)
    FB21_mask = outgoing_mask(FB21)
    FB23_mask = outgoing_mask(FB23)
    FB24_mask = outgoing_mask(FB24)

    w0 = (1.0-AA0)*FB20_mask
    w1 = (1.0-AA1)*FB21_mask
    w2 = (1.0-torch.clamp(dilation(A2_pred0),0.0,1.0))
    w3 = (1.0-AA3)*FB23_mask
    w4 = (1.0-AA4)*FB24_mask

    final_B2 = (II0*w0*II1*w1+fused_frame2*w2+II3*w3+II4*w4)/torch.maximum(w0+w1+w2+w3+w4,torch.Tensor(1e-10))
    zeroweightmask = torch.cat([w0+w1+w2+w3+w4,w0+w1+w2+w3+w4,w0+w1+w2+w3+w4],-1)
    final_B2 = torch.where(torch.less_equal(zeroweightmask,0.0),B2_pred0,final_B2)
    return final_B2

def train():
    learning_rate = 0.0001
    device = "cuda" if torch.cuda.is_available() else "cpu"  # Configure device
    loss,model = test_train(True)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    num_steps = 1001

    for i in range(0, num_steps):
        optimizer.zero_grad()
        loss, model_params = test_train(True)
        optimizer.step()
        print("step",num_steps,"loss = ",loss)

def test():
    B2 = test_train(False)
    plt.imshow(B2)
    plt.show()


