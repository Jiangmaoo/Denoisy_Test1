import math

import torch
import torch.nn as nn

class DN_Net(nn.Module):
    def __init__(self, input_channels=3, output_channels=3):
        super(DN_Net, self).__init__()
        #对噪声图像编码
        self.noise_encoder = NoiseEncoder(input_channels)
        #对加雾的噪声图像进行编码
        self.haze_encoder = HazeEncoder(input_channels)

        self.clean_decoder = CleanDecoder(input_channels)
        self.noise_decoder = NoiseFeatureDecoder(input_channels)
        #噪声移除解码器
        self.noise_move_decoder=NoiseDecoder(input_channels)


        #定义一个实例变量placeholder并将其初始化为None，可以再类方法里面使用
        self.placeholder = None

    def forward(self, noise_img, haze_img):
        #分别使用噪声图像编码器，和有雾图像编码器进行编码
        noise_features = self.noise_encoder(noise_img)
        haze_features = self.haze_encoder(haze_img)


        #使用clean解码器
        clean_recon = self.clean_decoder(noise_features,noise_features+haze_features)
        noise_recon = self.noise_decoder(noise_features-haze_features, haze_features)
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

        return clean_recon, noise_recon,gt_recon

    def test(self,noise_img, haze_img):
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

class NoiseEncoder(nn.Module):
    def __init__(self,input_channels=3):
        super(NoiseEncoder, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        return x


class HazeEncoder(nn.Module):
    def __init__(self,input_channels=3):
        super(HazeEncoder, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        return x


class CleanDecoder(nn.Module):
    def __init__(self,output_channels=3):
        super(CleanDecoder, self).__init__()
        self.conv1 = nn.Conv2d(1024, 256, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, output_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()

    def forward(self, noise, haze):
        x = torch.cat((noise, haze), dim=1)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        return x

# NoiseFeatureDecoder 的输出是一个单通道的噪声图像，这是因为在模型中，我们对噪声特征进行重建，得到的是一张噪声图像。如果要得到掩码，需要进一步处理这张噪声图像。
class NoiseFeatureDecoder(nn.Module):
    def __init__(self,output_channels=1):
        super(NoiseFeatureDecoder, self).__init__()
        self.conv1 = nn.Conv2d(1024, 256, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, output_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()

    def forward(self, noise, haze):
        x = torch.cat((noise, haze), dim=1)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        return x

#噪声去除解码器
class NoiseDecoder(nn.Module):
    def __init__(self,output_channels=3):
        super(NoiseDecoder, self).__init__()
        self.conv1 = nn.Conv2d(1024, 256, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, output_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()

    def forward(self, noise, haze):
        x = torch.cat((noise, haze), dim=1)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
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
            x=self.conv(x)
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


