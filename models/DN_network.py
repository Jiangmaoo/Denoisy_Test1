import math

import torch
import torch.nn as nn


class DN_Net(nn.Module):
    def __init__(self, input_channels=3, output_channels=3,bias=True,kernel_size = 3):
        super(DN_Net, self).__init__()
        #对噪声图像编码
        self.noise_encoder = Encoder(input_channels)
        #对加雾的噪声图像进行编码
        self.haze_encoder = HazeEncoder(input_channels)

        self.clean_decoder = CleanDecoder(input_channels)
        self.noise_decoder = NoiseFeatureDecoder(input_channels)
        #噪声移除解码器
        self.noise_move_decoder=NoiseDecoder(input_channels)

        self.dilateconv=DilateConv(input_channels)
        self.conv_tail = nn.Conv2d(2 * input_channels, output_channels, kernel_size=kernel_size, padding=1, bias=bias)
        self.up = Up()

        #定义一个实例变量placeholder并将其初始化为None，可以再类方法里面使用
        self.placeholder = None

    def forward(self, noise_img, haze_img):
        #分别使用噪声图像编码器，和有雾图像编码器进行编码
        noise_features = self.noise_encoder(noise_img)
        haze_features = self.haze_encoder(haze_img)

        #使用clean解码器
        # clean_recon = self.clean_decoder(noise_features,(noise_features+haze_features)/2.0)
        # noise_recon = self.noise_decoder((noise_features-haze_features)/2.0, haze_features)
        clean_recon = self.clean_decoder(noise_features,haze_features)
        noise_recon = self.noise_decoder(noise_features, haze_features)
        gt_recon=self.noise_move_decoder(noise_features, haze_features)
        # Concatenate the feature vectors and calculate the difference and similarity vectors
        # feat = torch.cat((noise_features, haze_features), dim=1)
        # diff_feat = noise_features - haze_features
        # sim_feat = noise_features + haze_features

        # Decode the similarity and difference vectors to obtain the reconstructed clean image and noise image
        # clean_recon = self.clean_decoder(sim_feat)
        # noise_recon = self.noise_decoder(diff_feat)
        # gt_recon=self.noise_move_decoder(sim_feat)

        # noise_feat = self.encoder_noise(x_noise)
        # haze_feat = self.encoder_haze(x_haze)
        # feat_similar = torch.cat((noise_feat, haze_feat), dim=1)
        # feat_diff = noise_feat - haze_feat
        # x_clean = self.decoder_clean(feat_similar)
        # x_noise = self.decoder_noise(feat_diff)

        #计算损失。这是采用的MSE损失，也可以采用L2损失，用于衡量重建图像与gt图像之间的差异
        # clean_loss = F.mse_loss(clean_features, gt)
        # mask = (noise_recon > 0.05).float()
        #重建噪声图像解码器的BCE损失，用于衡量去噪效果。二分值
        # noise_loss = F.binary_cross_entropy_with_logits(noise_recon, mask)
        # 扩张卷积
        delate_x=self.dilateconv(noise_img)
        x=clean_recon+noise_img
        y=delate_x+noise_img

        z=torch.cat([x,y],dim=1)
        z_=self.conv_tail(z)
        # print(z_.shape)
        # print('----------')


        return clean_recon, noise_recon,gt_recon,z_

    def test(self,noise_img, haze_img):
        noise_features = self.noise_encoder(noise_img)
        haze_features = self.haze_encoder(haze_img)
        reconstruct_gt=self.noise_move_decoder(noise_features, haze_features)   #阴影移除联合解码器Js2sf

        clean_recon = self.clean_decoder(noise_features,haze_features)
        noise_recon = self.noise_decoder(noise_features, haze_features)
        gt_recon=self.noise_move_decoder(noise_features, haze_features)

        # 扩张卷积
        delate_x = self.dilateconv(noise_img)
        z = torch.cat([clean_recon, delate_x], dim=1)
        z_ = self.conv_tail(z)


        return clean_recon,noise_recon,delate_x,z_

    def test1(self,noise_img, haze_img):
        noise_features = self.noise_encoder(noise_img)
        haze_features = self.haze_encoder(haze_img)
        reconstruct_gt=self.noise_move_decoder(noise_features, haze_features)   #阴影移除联合解码器Js2sf


        return reconstruct_gt

    def test_pair(self,noise_img,haze_img):
        noise_features = self.noise_encoder(noise_img)
        haze_features = self.haze_encoder(haze_img)

        if self.placeholder is None or self.placeholder.size(0)!=noise_features.size(0):
            self.placeholder={}
            for key in noise_features.keys():
                self.placeholder[key]=torch.zeros(noise_features.shape,requires_grad=False).to(
                    torch.device(noise_features.device)
                )

        rec_noise=self.noise_encoder(noise_features,self.placeholder)
        rec_clean=self.clean_decoder(self.placeholder,haze_features)

        return rec_clean,rec_noise


class Up(nn.Module):
    def __init__(self):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
class DilateConv(nn.Module):
    def __init__(self,input_channels=3,output_channels=3,bias=True,kernel_size = 3,nc = 64):
        super(DilateConv, self).__init__()

        self.m_dilateconv1 = nn.Conv2d(input_channels, nc, kernel_size=kernel_size, padding=1, dilation=1, bias=bias)
        self.m_bn1 = nn.BatchNorm2d(nc)
        self.m_relu1 = nn.ReLU(inplace=True)

        self.m_dilateconv2 = nn.Conv2d(nc, nc, kernel_size=kernel_size, padding=2, dilation=2, bias=bias)
        self.m_bn2 = nn.BatchNorm2d(nc)
        self.m_relu2 = nn.ReLU(inplace=True)

        self.m_dilateconv3 = nn.Conv2d(nc, nc, kernel_size=kernel_size, padding=3, dilation=3, bias=bias)
        self.m_bn3 = nn.BatchNorm2d(nc)
        self.m_relu3 = nn.ReLU(inplace=True)

        self.m_dilateconv4 = nn.Conv2d(nc, nc, kernel_size=kernel_size, padding=4, dilation=4, bias=bias)
        self.m_bn4 = nn.BatchNorm2d(nc)
        self.m_relu4 = nn.ReLU(inplace=True)

        self.m_dilateconv5 = nn.Conv2d(nc, nc, kernel_size=kernel_size, padding=5, dilation=5, bias=bias)
        self.m_bn5 = nn.BatchNorm2d(nc)
        self.m_relu5 = nn.ReLU(inplace=True)

        self.m_dilateconv6 = nn.Conv2d(nc, nc, kernel_size=kernel_size, padding=6, dilation=6, bias=bias)
        self.m_bn6 = nn.BatchNorm2d(nc)
        self.m_relu6 = nn.ReLU(inplace=True)

        self.m_dilateconv7 = nn.Conv2d(nc, nc, kernel_size=kernel_size, padding=7, dilation=7, bias=bias)
        self.m_bn7 = nn.BatchNorm2d(nc)
        self.m_relu7 = nn.ReLU(inplace=True)

        self.m_dilateconv8 = nn.Conv2d(nc, nc, kernel_size=kernel_size, padding=8, dilation=8, bias=bias)
        self.m_bn8 = nn.BatchNorm2d(nc)
        self.m_relu8 = nn.ReLU(inplace=True)

        self.m_dilateconv7_1 = nn.Conv2d(nc, nc, kernel_size=kernel_size, padding=7, dilation=7, bias=bias)
        self.m_bn7_1 = nn.BatchNorm2d(nc)
        self.m_relu7_1 = nn.ReLU(inplace=True)

        self.m_dilateconv6_1 = nn.Conv2d(nc, nc, kernel_size=kernel_size, padding=6, dilation=6, bias=bias)
        self.m_bn6_1 = nn.BatchNorm2d(nc)
        self.m_relu6_1 = nn.ReLU(inplace=True)

        self.m_dilateconv5_1 = nn.Conv2d(nc, nc, kernel_size=kernel_size, padding=5, dilation=5, bias=bias)
        self.m_bn5_1 = nn.BatchNorm2d(nc)
        self.m_relu5_1 = nn.ReLU(inplace=True)

        self.m_dilateconv4_1 = nn.Conv2d(nc, nc, kernel_size=kernel_size, padding=4, dilation=4, bias=bias)
        self.m_bn4_1 = nn.BatchNorm2d(nc)
        self.m_relu4_1 = nn.ReLU(inplace=True)

        self.m_dilateconv3_1 = nn.Conv2d(nc, nc, kernel_size=kernel_size, padding=3, dilation=3, bias=bias)
        self.m_bn3_1 = nn.BatchNorm2d(nc)
        self.m_relu3_1 = nn.ReLU(inplace=True)

        self.m_dilateconv2_1 = nn.Conv2d(nc, nc, kernel_size=kernel_size, padding=2, dilation=2, bias=bias)
        self.m_bn2_1 = nn.BatchNorm2d(nc)
        self.m_relu2_1 = nn.ReLU(inplace=True)

        self.m_dilateconv1_1 = nn.Conv2d(nc, nc, kernel_size=kernel_size, padding=1, dilation=1, bias=bias)
        self.m_bn1_1 = nn.BatchNorm2d(nc)
        self.m_relu1_1 = nn.ReLU(inplace=True)

        self.m_dilateconv = nn.Conv2d(nc, output_channels, kernel_size=kernel_size, padding=1, bias=bias)

        self.conv_tail = nn.Conv2d(2 * output_channels, output_channels, kernel_size=kernel_size, padding=1, bias=bias)


    def forward(self, x):
        y1 = self.m_dilateconv1(x)
        y1_1 = self.m_bn1(y1)
        y1_1 = self.m_relu1(y1_1)

        y2 = self.m_dilateconv2(y1_1)
        y2_1 = self.m_bn2(y2)
        y2_1 = self.m_relu2(y2_1)

        y3 = self.m_dilateconv3(y2_1)
        y3_1 = self.m_bn3(y3)
        y3_1 = self.m_relu3(y3_1)

        y4 = self.m_dilateconv4(y3_1)
        y4_1 = self.m_bn4(y4)
        y4_1 = self.m_relu4(y4_1)

        y5 = self.m_dilateconv5(y4_1)
        y5_1 = self.m_bn5(y5)
        y5_1 = self.m_relu5(y5_1)

        y6 = self.m_dilateconv6(y5_1)
        y6_1 = self.m_bn6(y6)
        y6_1 = self.m_relu6(y6_1)

        y7 = self.m_dilateconv7(y6_1)
        y7_1 = self.m_bn7(y7)
        y7_1 = self.m_relu7(y7_1)

        y8 = self.m_dilateconv8(y7_1)
        y8_1 = self.m_bn8(y8)
        y8_1 = self.m_relu8(y8_1)

        y9 = self.m_dilateconv7_1(y8_1)
        y9 = self.m_bn7_1(y9)
        y9 = self.m_relu7_1(y9)

        y10 = self.m_dilateconv6_1(y9 + y7)
        y10 = self.m_bn6_1(y10)
        y10 = self.m_relu6_1(y10)

        y11 = self.m_dilateconv5_1(y10 + y6)
        y11 = self.m_bn5_1(y11)
        y11 = self.m_relu5_1(y11)

        y12 = self.m_dilateconv4_1(y11 + y5)
        y12 = self.m_bn4_1(y12)
        y12 = self.m_relu4_1(y12)

        y13 = self.m_dilateconv3_1(y12 + y4)
        y13 = self.m_bn3_1(y13)
        y13 = self.m_relu3_1(y13)

        y14 = self.m_dilateconv2_1(y13 + y3)
        y14 = self.m_bn2_1(y14)
        y14 = self.m_relu2_1(y14)

        y15 = self.m_dilateconv1_1(y14 + y2)
        y15 = self.m_bn1_1(y15)
        y15 = self.m_relu1_1(y15)

        y = self.m_dilateconv(y15 + y1)

        return y


class Encoder(nn.Module):
    def __init__(self,input_channels=3):
        super(Encoder, self).__init__()
        self.conv1=Cvi(input_channels,64)
        self.conv2 = Cvi(64, 128, before="LReLU", after="BN")
        self.conv3 = Cvi(128, 256, before="LReLU", after="BN")
        self.conv4 = Cvi(256, 512, before="LReLU")

    def forward(self, x):
        x1=self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4=self.conv4(x3)
        # print(x1.shape)
        # print(x2.shape)
        # print(x3.shape)
        # print(x4.shape)
        # print('-------')

        feature_dic = {
            "x1": x1,
            "x2": x2,
            "x3": x3,
            "x4": x4,

        }

        return feature_dic


class HazeEncoder(nn.Module):
    def __init__(self, input_channels=3):
        super(HazeEncoder, self).__init__()
        self.conv1 = Cvi(input_channels, 64)
        self.conv2 = Cvi(64, 128, before="LReLU", after="BN")
        self.conv3 = Cvi(128, 256, before="LReLU", after="BN")
        self.conv4 = Cvi(256, 512, before="LReLU")

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)

        feature_dic = {
            "x1": x1,
            "x2": x2,
            "x3": x3,
            "x4": x4,

        }

        return feature_dic


class CleanDecoder(nn.Module):
    def __init__(self,output_channels=3):
        super(CleanDecoder, self).__init__()
        self.conv1 = CvTi(1024, 256, before="ReLU", after="BN")
        self.conv2 = CvTi(768, 128, before="ReLU", after="BN")
        self.conv3 = CvTi(384, 64, before="ReLU", after="BN")
        self.conv4 = CvTi(192, output_channels, before="ReLU", after="Tanh")


    def forward(self, noise, haze):

        x4 = torch.cat([noise["x4"], (noise["x4"]+haze["x4"])/2.0], dim=1)
        # print(x4.shape)
        # print(noise["x4"].shape)
        x3 = self.conv1(x4)
        # print(x3.shape)
        # print(noise["x3"].shape)
        # print((noise["x3"]+haze["x3"]).shape)
        # print(haze["x3"].shape)
        cat3 = torch.cat([x3, noise["x3"], (noise["x3"]+haze["x3"])/2.0], dim=1)
        x2 = self.conv2(cat3)
        cat2 = torch.cat([x2, noise["x2"], (noise["x2"]+haze["x2"])/2.0], dim=1)
        x1= self.conv3(cat2)
        cat1 = torch.cat([x1, noise["x1"], (noise["x1"]+haze["x1"])/2.0], dim=1)
        x = self.conv4(cat1)

        return x

# NoiseFeatureDecoder 的输出是一个单通道的噪声图像，这是因为在模型中，我们对噪声特征进行重建，得到的是一张噪声图像。如果要得到掩码，需要进一步处理这张噪声图像。
class NoiseFeatureDecoder(nn.Module):
    def __init__(self, output_channels=3):
        super(NoiseFeatureDecoder, self).__init__()
        self.conv1 = CvTi(1024, 256, before="ReLU", after="BN")
        self.conv2 = CvTi(768, 128, before="ReLU", after="BN")
        self.conv3 = CvTi(384, 64, before="ReLU", after="BN")
        self.conv4 = CvTi(192, output_channels, before="ReLU", after="Tanh")

    def forward(self, noise, haze):
        x4 = torch.cat([noise["x4"], (noise["x4"] - haze["x4"]) / 2.0], dim=1)
        x3 = self.conv1(x4)
        cat3 = torch.cat([x3, noise["x3"], (noise["x3"] -haze["x3"]) / 2.0], dim=1)
        x2 = self.conv2(cat3)
        cat2 = torch.cat([x2, noise["x2"], (noise["x2"] - haze["x2"]) / 2.0], dim=1)
        x1 = self.conv3(cat2)
        cat1 = torch.cat([x1, noise["x1"], (noise["x1"] - haze["x1"]) / 2.0], dim=1)
        x = self.conv4(cat1)

        return x

#噪声去除解码器
class NoiseDecoder(nn.Module):
    def __init__(self, output_channels=3):
        super(NoiseDecoder, self).__init__()
        self.conv1 = CvTi(1024, 256, before="ReLU", after="BN")
        self.conv2 = CvTi(768, 128, before="ReLU", after="BN")
        self.conv3 = CvTi(384, 64, before="ReLU", after="BN")
        self.conv4 = CvTi(192, output_channels, before="ReLU", after="Tanh")

    def forward(self, noise, haze):
        x4 = torch.cat([noise["x4"], (noise["x4"] + haze["x4"]) / 2.0], dim=1)
        x3 = self.conv1(x4)
        cat3 = torch.cat([x3, noise["x3"], (noise["x3"] + haze["x3"]) / 2.0], dim=1)
        x2 = self.conv2(cat3)
        cat2 = torch.cat([x2, noise["x2"], (noise["x2"] + haze["x2"]) / 2.0], dim=1)
        x1 = self.conv3(cat2)
        cat1 = torch.cat([x1, noise["x1"], (noise["x1"] + haze["x1"]) / 2.0], dim=1)
        x = self.conv4(cat1)

        return x

#初始化权重
def weights_init(init_type="gaussian"):
    def init_fun(m):
        classname=m.__class__.__name__
        if (classname.find("Conv")==0 or classname.find("Linear")==0 or hasattr(m,"weight")):
            if init_type=="gaussian":
                nn.init.normal_(m.weight,0.0,0.02)
            elif init_type=="xavier":
                nn.init.xavier_normal_(m.weight,gain=math.sqrt(2))
            elif init_type=="kaiming":
                nn.init.kaiming_normal_(m.weight,a=0,mode="fan_in")
            elif init_type=="orthogonal":
                nn.init.orthogonal_(m.weight,gain=math.sqrt(2))
            elif init_type=="default":
                pass
            else:
                assert 0,"Unsupported initialization:{}".format(init_type)
            if hasattr(m,"bias") and m.bias is not None:
                nn.init.constant_(m.bias,0.0)
    return init_fun
#卷积
class Cvi(nn.Module):
    def __init__(self,in_channels,out_channels,before=None,after=False,kernel_size=4,stride=2,
                 padding=1,dilation=1,groups=1,bias=False):
        super(Cvi,self).__init__()

        #初始化卷积
        self.conv=nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,
                            stride=stride,padding=padding,dilation=dilation,groups=groups,bias=bias)

        #初始化卷积参数
        self.conv.apply(weights_init("gaussian"))

        #卷积后进行的操作
        if after=="BN":
            self.after=nn.BatchNorm2d(out_channels)   #归一化
        elif after=="Tanh":
            self.after=torch.tanh #tanh激活函数（-1到1S型）
        elif after=="sigmoid":
            self.after=torch.sigmoid    #sigmoid激活函数（0到1S型）

        #卷积前进行的操作
        if before=="ReLU":
            self.after=nn.ReLU(inplace=True)  #ReLU激活函数（<0时=0；>0时等于自身)(inplace=True,节省反复申请与释放内存的空间和时间)
        elif before=="LReLU":
            self.before=nn.LeakyReLU(negative_slope=0.2,inplace=False)  #LeakyReLu激活函数（<0时斜率为0.2）

    def forward(self,x):
        if hasattr(self,"before"):
            x=self.before(x)
        x=self.conv(x)
        if hasattr(self,"after"):
            x=self.after(x)
        return x


# 扩张卷积
class DilateCvi(nn.Module):
    def __init__(self,in_channels,out_channels,before=None,after=False,kernel_size=4,stride=2,
                 padding=1,dilation=1,groups=1,bias=False):
        super(Cvi,self).__init__()

        #初始化卷积
        self.conv=nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,
                            stride=stride,padding=padding,dilation=dilation,groups=groups,bias=bias)

        #初始化卷积参数
        self.conv.apply(weights_init("gaussian"))

        #卷积后进行的操作
        if after=="BN":
            self.after=nn.BatchNorm2d(out_channels)   #归一化
        elif after=="Tanh":
            self.after=torch.tanh #tanh激活函数（-1到1S型）
        elif after=="sigmoid":
            self.after=torch.sigmoid    #sigmoid激活函数（0到1S型）

        #卷积前进行的操作
        if before=="ReLU":
            self.after=nn.ReLU(inplace=True)  #ReLU激活函数（<0时=0；>0时等于自身)(inplace=True,节省反复申请与释放内存的空间和时间)
        elif before=="LReLU":
            self.before=nn.LeakyReLU(negative_slope=0.2,inplace=False)  #LeakyReLu激活函数（<0时斜率为0.2）

    def forward(self,x):
        if hasattr(self,"before"):
            x=self.before(x)
        x=self.conv(x)
        if hasattr(self,"after"):
            x=self.after(x)
        return x
#逆卷积
class CvTi(nn.Module):
    def __init__(self,in_channels,out_channels,before=None,after=False,kernel_size=4,stride=2,
                 padding=1,dilation=1,groups=1,bias=False):
        super(CvTi, self).__init__()

        #初始化逆卷积
        self.conv=nn.ConvTranspose2d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,
                                     stride=stride,padding=padding)
        #初始化逆卷积权重
        self.conv.apply(weights_init("gaussian"))

        # 卷积后进行的操作
        if after=="BN":
            self.after=nn.BatchNorm2d(out_channels)
        elif after=="Tanh":
            self.after=torch.tanh
        elif after=="sigmoid":
            self.after=torch.sigmoid

        #卷积前进行的操作
        if before=="ReLU":
            self.before=nn.ReLU(inplace=True)
        elif before=="LReLU":
            self.before=nn.LeakyReLU(negative_slope=0.2,inplace=True)
    def forward(self,x):
        if hasattr(self,"before"):
            x=self.before(x)
        x=self.conv(x)
        if hasattr(self,"after"):
            x=self.after(x)
        return x
#鉴别器
class Discriminator(nn.Module):
    def __init__(self,input_channels=4):
        super(Discriminator, self).__init__()
        self.cv0=Cvi(input_channels,64)
        self.cv1=Cvi(64,128,before="LReLU",after="BN")
        self.cv2 = Cvi(128, 256, before="LReLU", after="BN")
        self.cv3 = Cvi(256, 512, before="LReLU", after="BN")
        self.cv4 = Cvi(512, 1, before="LReLU", after="sigmoid")

    def forward(self,x):
        x0=self.cv0(x)
        x1=self.cv1(x0)
        x2=self.cv2(x1)
        x3=self.cv3(x2)
        out=self.cv4(x3)

        return out

if __name__=='__main__':
    #BCHW
    size=(3,3,256,256)
    input1=torch.ones(size)
    input2=torch.ones(size)
    # l2=nn.L2Loss()
    model = DN_Net()
    noise_img = torch.randn(1, 3, 256, 256)
    haze_img = torch.randn(1, 3, 256, 256)
    clean_recon, noise_recon, gt_recon,z_n = model(noise_img, haze_img)
    print(clean_recon.shape, noise_recon.shape, gt_recon.shape)

    # size(3,3,256,256)
    input=torch.ones(size)


# class NoiseEncoder(nn.Module):
#     def __init__(self):
#         super(NoiseEncoder, self).__init__()
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
#         self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
#         self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
#         self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
#         self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
#
#     def forward(self, x):
#         out = nn.functional.relu(self.conv1(x))
#         out = nn.functional.relu(self.conv2(out))
#         out = nn.functional.max_pool2d(out, 2)
#         out = nn.functional.relu(self.conv3(out))
#         out = nn.functional.relu(self.conv4(out))
#         out = nn.functional.max_pool2d(out, 2)
#         out = nn.functional.relu(self.conv5(out))
#         out = nn.functional.relu(self.conv6(out))
#         out = nn.functional.max_pool2d(out, 2)
#         out = out.view(out.size(0), -1)
#         return out
#
# class HazeEncoder(nn.Module):
#     def __init__(self):
#         super(HazeEncoder, self).__init__()
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
#         self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
#         self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
#         self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
#         self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
#
#     def forward(self, x):
#         out = nn.functional.relu(self.conv1(x))
#         out = nn.functional.relu(self.conv2(out))
#         out = nn.functional.max_pool2d(out, 2)
#         out = nn.functional.relu(self.conv3(out))
#         out = nn.functional.relu(self.conv4(out))
#         out = nn.functional.max_pool2d(out, 2)
#         out = nn.functional.relu(self.conv5(out))
#         out = nn.functional.relu(self.conv6(out))
#         out = nn.functional.max_pool2d(out, 2)
#         out = out.view(out.size(0), -1)
#         return out


