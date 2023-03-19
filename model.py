import torch.nn as nn
import torch
import torch.nn.init as init

def init_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal_(m.weight, 0, 0.01)
            if m.bias is not None:
                init.constant_(m.bias, 0)
#深度可分离卷积
class DP_Conv(nn.Module):
    def __init__(self,indim:int, outdim:int, stride:int, ksize:int):
        super(DP_Conv, self).__init__()
        if ksize==1:
            self.depthwise_conv = nn.Conv2d(in_channels=indim,out_channels=indim,kernel_size=1,
                                            stride=stride,padding=0,groups=indim)
        else:
            self.depthwise_conv = nn.Conv2d(in_channels=indim,out_channels=indim,kernel_size=ksize,
                                            stride=stride,padding=1,groups=indim)
        self.point_conv = nn.Conv2d(in_channels=indim,out_channels=outdim,kernel_size=1)
        self.BN_1 = nn.BatchNorm2d(indim)
        self.relu=nn.ReLU()
        self.BN_2 = nn.BatchNorm2d(outdim)
        init_weights(self)
    def forward(self,x:torch.Tensor)->torch.Tensor:
        x = self.depthwise_conv(x)
        x = self.BN_1(x)
        x = self.relu(x)
        x = self.point_conv(x)
        print(x.shape)
        return self.relu(x)

#卷积层
class Convolutional(nn.Module):
    def __init__(self,indim:int,outdim:int,stride:int,ksize:int):
        super(Convolutional, self).__init__()
        self.conv = DP_Conv(indim,outdim,stride,ksize)
        self.bn = nn.BatchNorm2d(outdim)
        self.lr = nn.LeakyReLU()
        init_weights(self)
    def forward(self,x:torch.Tensor)->torch.Tensor:
        return self.lr(self.bn(self.conv(x)))


class SelfAttention(nn.Module):
    def __init__(self, channels):
        super(SelfAttention, self).__init__()

        self.query_conv = nn.Conv2d(channels, channels // 8, 1)
        self.key_conv = nn.Conv2d(channels, channels // 8, 1)
        self.value_conv = nn.Conv2d(channels, channels, 1)

        self.gamma = nn.Parameter(torch.zeros(1))
        init_weights(self)

    def forward(self, x):
        m_batchsize, C, height, width = x.size()

        query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        key = self.key_conv(x).view(m_batchsize, -1, width * height)

        energy = torch.bmm(query, key)
        attention = torch.softmax(energy, dim=-1)

        value = self.value_conv(x).view(m_batchsize, -1, width * height)

        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out


class Res_conv(nn.Module):
    def __init__(self,indim:int):
        super(Res_conv, self).__init__()
        self.conv1 = Convolutional(indim,int(indim/2),stride=1,ksize=1)
        self.attention = SelfAttention(int(indim/2))
        self.conv2 = Convolutional(int(indim/2),indim,stride=1,ksize=3)
        init_weights(self)

    def forward(self,x:torch.Tensor) ->torch.Tensor:
        x = x+self.conv2(self.attention(self.conv1(x)))
        return x


#主干网络 原型:Darknet53
class Darknet53(nn.Module):
    def __init__(self):
        super(Darknet53, self).__init__()
        self.l1 = nn.Sequential(Convolutional(3,32,1,3), Convolutional(32,64,2,3)) # /2
        self.l2 = nn.Sequential(Res_conv(64), SelfAttention(64)) # Add attention layer
        self.l3 = Convolutional(64,128,2,3)  # /4
        self.l4 = nn.Sequential(Res_conv(128),Res_conv(128))
        self.l5 = Convolutional(128,256,2,3)  # /8
        self.l6 = nn.Sequential()
        for i in range(8):
            self.l6.append(Res_conv(256))
        self.l7 = Convolutional(256,512,2,3)  # /16
        self.l8 = nn.Sequential()
        for i in range(8):
            self.l8.append(Res_conv(512))
        self.l9 = Convolutional(512, 1024, 2, 3)  # /32
        self.l10 = nn.Sequential()
        for i in range(8):
            self.l10.append(Res_conv(1024))
        init_weights(self)

    def forward(self,x:torch.Tensor) ->(torch.Tensor,torch.Tensor,torch.Tensor):
        x1 = self.l6(self.l5(self.l4(self.l3(self.l2(self.l1(x))))))
        x1_1 = self.l7(x1)
        x2 = self.l8(x1_1)
        x3 = self.l10(self.l9(x2))
        return x1,x2,x3

#SPP
class SPP(nn.Module):
    def __init__(self):
        super(SPP, self).__init__()
        self.p1 = nn.MaxPool2d(5,1,2)
        self.p2 = nn.MaxPool2d(9,1,4)
        self.p3 = nn.MaxPool2d(13,1,6)
        init_weights(self)
    def forward(self,x:torch.Tensor) -> torch.Tensor:
        x1 = torch.cat([x,self.p1(x)],axis=1)
        x2 = torch.cat([x1,self.p2(x)],axis=1)
        x3 = torch.cat([x2,self.p3(x)],axis=1)
        return x3

# Convolutional_set
class Convolutional_set(nn.Module):
    def __init__(self,set:int):
        super(Convolutional_set, self).__init__()
        if set ==1:
            outdim=256
            self.l1 = Convolutional(768,outdim,1,1)
        else:
            outdim = 128
            self.l1 = Convolutional(384, outdim, 1, 1)
        self.l2 = Convolutional(outdim, outdim * 2, 1, 3)
        self.l3 = Convolutional(outdim * 2, outdim, 1, 1)
        self.l4 = Convolutional(outdim, outdim * 2, 1, 3)
        self.l5 = Convolutional(outdim * 2, outdim, 1, 1)
        init_weights(self)
    def forward(self,x:torch.Tensor)->torch.Tensor:
        return self.l5(self.l4(self.l3(self.l2(self.l1(x)))))

# Model
class YOLO_3_SPP_Model(nn.Module):
    def __init__(self,outdim:int):
        super(YOLO_3_SPP_Model, self).__init__()
        self.darknet = Darknet53()
        self.l = nn.Sequential(Convolutional(1024,512,1,1),
                               Convolutional(512,1024,1,3),
                               Convolutional(1024,512,1,1))
        self.spp = SPP()
        self.l1 = nn.Sequential(Convolutional(2048,512,1,1),
                                Convolutional(512,1024,1,3),
                                Convolutional(1024,512,1,1))
        self.l1_output = nn.Sequential(Convolutional(512,1024,1,3),
                                       nn.Conv2d(1024,outdim,1,1))
        self.l2 = nn.Sequential(Convolutional(512,256,1,1),
                                nn.Upsample(scale_factor=2))
        self.set1 = Convolutional_set(1)
        self.l2_output = nn.Sequential(Convolutional(256,512,1,3),
                                       nn.Conv2d(512,outdim,1,1))
        self.l3 = nn.Sequential(Convolutional(256,128,1,1),
                                nn.Upsample(scale_factor=2))
        self.set2 = Convolutional_set(2)
        self.l3_output = nn.Sequential(Convolutional(128,256,1,3),
                                       nn.Conv2d(256,outdim,1,1))
        init_weights(self)
    def forward(self,x:torch.Tensor)->(torch.Tensor,torch.Tensor,torch.Tensor):
        x1,x2,x3 = self.darknet(x)
        x1_1 = self.l1(self.spp(self.l(x3)))
        output_1 = self.l1_output(x1_1)
        x2_2 = torch.cat([self.l2(x1_1),x2],axis=1)
        x2_2 = self.set1(x2_2)
        output_2 = self.l2_output(x2_2)
        x3_3 = self.l3(x2_2)
        x3_3 = torch.cat([x3_3,x1],axis=1)
        x3_3 = self.set2(x3_3)
        output_3 = self.l3_output(x3_3)
        return output_1,output_2,output_3

#模型可视化
if __name__ == "__main__":
    import time
    model =YOLO_3_SPP_Model(6)
    x = torch.rand(1,3,224,224)
    a1=time.time()
    c = model(x)
    a2=time.time()-a1
    model = YOLO_3_SPP_Model(100)
    with torch.no_grad():
        torch.onnx.export(
            model,
            x,
            "model.onnx",
            opset_version=11,
            input_names=['input'],
            output_names=['output']
        )
    import netron
    modelPath = "model.onnx"
    netron.start(modelPath)
