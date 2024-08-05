"""
*****************************************

Distill-DBDGAN:
Teacher Model
*****************************************
*****************************************

DEVELOPER: SANKARAGANESH JONNA, MOUSHUMI MEDHI

*****************************************
"""


import torch
from torch import nn
import pretrainedmodels
from torch.nn import functional as F
from torch.nn import BatchNorm2d as bn



class ConvBnop(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1),
                                  nn.BatchNorm2d(out_channels),
                                  nn.ReLU(inplace=True)
                                  )

    def forward(self, x):
        return self.conv(x)
  
class _DenseAsppDBD(nn.Sequential):
    """ ConvNet block for building DenseASPP. """

    def __init__(self, input_num, num1, num2, dilation_rate, drop_out, bn_start=True):
        super(_DenseAsppDBD, self).__init__()
        if bn_start:
            self.add_module('norm1', bn(input_num, momentum=0.0003)),

        self.add_module('relu1', nn.ReLU(inplace=False)),
        self.add_module('conv1', nn.Conv2d(in_channels=input_num, out_channels=num1, kernel_size=1)),

        self.add_module('norm2', bn(num1, momentum=0.0003)),
        self.add_module('relu2', nn.ReLU(inplace=False)),
        self.add_module('conv2', nn.Conv2d(in_channels=num1, out_channels=num2, kernel_size=3,
                                            dilation=dilation_rate, padding=dilation_rate)),

        self.drop_rate = drop_out

    def forward(self, _input):
        feature = super(_DenseAsppDBD, self).forward(_input)

        if self.drop_rate > 0:
            feature = F.dropout2d(feature, p=self.drop_rate, training=self.training)

        return feature
    
class DenseASPP(nn.Module):

    def __init__(self, model_cfg, n_class=19, output_stride=8):
        super(DenseASPP, self).__init__()
        num_init_features = 2048 #512 for VGG19 backbone
       
        dropout0 = 0.1
        d_feature0 = 1024
        d_feature1 = 128

        
        # Each denseblock
        num_features = num_init_features
        # block1*****************************************************************************************************
        

        self.ASPP_3 = _DenseAsppDBD(input_num=num_features, num1=d_feature0, num2=d_feature1,
                                      dilation_rate=2, drop_out=dropout0, bn_start=False)

        self.ASPP_6 = _DenseAsppDBD(input_num=num_features + d_feature1 * 1, num1=d_feature0, num2=d_feature1,
                                      dilation_rate=4, drop_out=dropout0, bn_start=True)

        self.ASPP_12 = _DenseAsppDBD(input_num=num_features + d_feature1 * 2, num1=d_feature0, num2=d_feature1,
                                       dilation_rate=6, drop_out=dropout0, bn_start=True)

        self.ASPP_18 = _DenseAsppDBD(input_num=num_features + d_feature1 * 3, num1=d_feature0, num2=d_feature1,
                                       dilation_rate=8, drop_out=dropout0, bn_start=True)

        num_features = num_features + 4 * d_feature1



        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform(m.weight.data)

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, _input):
        
        aspp1 = self.ASPP_3(_input)
        feature = torch.cat((aspp1, _input), dim=1)

        aspp2 = self.ASPP_6(feature)
        feature = torch.cat((aspp2, feature), dim=1)

        aspp3 = self.ASPP_12(feature)
        feature = torch.cat((aspp3, feature), dim=1)

        aspp4 = self.ASPP_18(feature)
        feature = torch.cat((aspp4, feature), dim=1)
  
        return feature

class SA2(nn.Module):
   
    def __init__(self,in_dim,activation,with_attn=True):
        super(SA2,self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        self.with_attn = with_attn
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1)  
    def forward(self,sa_E,sa_D):

        m_batchsize,C,width ,height = sa_E.size() 
        proj_query  = self.query_conv(sa_D).view(m_batchsize,-1,width*height).permute(0,2,1)  
        proj_key =  self.key_conv(sa_D).view(m_batchsize,-1,width*height)  
        energy =  torch.bmm(proj_query,proj_key)  
        attention = self.softmax(energy)  
        proj_value = self.value_conv(sa_E).view(m_batchsize,-1,width*height) 

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)

        out = self.gamma*out + sa_E
        return out , attention
   
   

class DecoderBlDBD(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, is_deconv=True):
        super(DecoderBlDBD, self).__init__()
        self.is_deconv = is_deconv

        self.deconv = nn.Sequential(
            ConvBnop(in_channels, middle_channels),
            nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.upsample = nn.Sequential(
            ConvBnop(in_channels, out_channels),
            nn.Upsample(scale_factor=2, mode='bilinear'),

        )

    def forward(self, x):
        if self.is_deconv:
            x = self.deconv(x)
        else:
            x = self.upsample(x)
        return x




class Teacher(nn.Module):
	"""ResNext backbone.
	ResNext: https://arxiv.org/abs/1611.05431
	Args:
	encoder_depth (int): Depth of a ResNet encoder.
	num_classes (int): Number of output classes.
	num_filters (int, optional): Number of filters in the last layer of decoder. Defaults to 32.
	dropout_2d (float, optional): Probability factor of dropout layer before output layer. Defaults to 0.2.
	is_deconv (bool, optional):
		False: bilinear interpolation is used in decoder.
		True: deconvolution is used in decoder.
		Defaults to False.
	"""

	def __init__(self, encoder_depth, num_classes, num_filters=32, dropout_2d=0.2, is_deconv=False):
		super().__init__()
		self.num_classes = num_classes
		self.dropout_2d = dropout_2d

		if encoder_depth == 50:
			self.encoder = pretrainedmodels.__dict__['se_resnext50_32x4d'](num_classes=1000, pretrained='imagenet')
			bottom_channel_nr = 2048
		elif encoder_depth == 101:
			self.encoder = pretrainedmodels.__dict__['se_resnext101_32x4d'](num_classes=1000, pretrained='imagenet')
			bottom_channel_nr = 2048 
			bottom_channel_nr_w = 4608 
		else:
			raise NotImplementedError('only 34, 101, 152 version of Resnet are implemented')

		self.pool = nn.MaxPool2d(2, 2)

		self.relu = nn.ReLU(inplace=True)

		self.input_adjust = nn.Sequential(self.encoder.layer0.conv1,
										  self.encoder.layer0.bn1,
										  self.encoder.layer0.relu1)

		self.conv1 = self.encoder.layer1
		self.conv2 = self.encoder.layer2
		self.conv3 = self.encoder.layer3
		self.conv4 = self.encoder.layer4
		
		self.SA2_Ex5=SA2(2048, 'relu', with_attn=False)
		self.SA2_Ex4=SA2(1280, 'relu', with_attn=False)
		self.SA2_Ex3=SA2(768, 'relu', with_attn=False)
		self.SA2_Ex2=SA2(320, 'relu', with_attn=False)
		self.SA2_Ex1=SA2(64, 'relu', with_attn=False)    
		
		self.denasppmodel = DenseASPP(2)


		self.dec4 = DecoderBlDBD(bottom_channel_nr_w, num_filters * 8 * 2, num_filters * 8, is_deconv)
		self.dec3 = DecoderBlDBD(bottom_channel_nr // 2 + num_filters * 8, num_filters * 8 * 2, num_filters * 8,
								   is_deconv)
		self.dec2 = DecoderBlDBD(bottom_channel_nr // 4 + num_filters * 8, num_filters * 4 * 2, num_filters * 2,
								   is_deconv)
		self.dec1 = DecoderBlDBD(bottom_channel_nr // 8 + num_filters * 2, num_filters * 2 * 2, num_filters * 2 * 2,
								   is_deconv)
		self.OUTMASK = nn.Conv2d(num_filters * 2 * 2, num_classes, kernel_size=1)

	def forward(self, x):
		input_adjust = self.input_adjust(x)
		conv1 = self.conv1(input_adjust)
		conv2 = self.conv2(conv1)
		conv3 = self.conv3(conv2)
		center = self.conv4(conv3)
		
		#Bottlenect
		de_aspp=self.denasppmodel(center)
		conv5_attnout, E_attn =self.SA2_Ex5(center,center)
		conv5_out=torch.cat([de_aspp, conv5_attnout], 1)		
	  
		#Decoder		
		dec4 = self.dec4(conv5_out)
		dec3 = self.dec3(torch.cat([dec4, conv3], 1))
		dec2 = self.dec2(torch.cat([dec3, conv2], 1))
		dec1 = F.dropout2d(self.dec1(torch.cat([dec2, conv1], 1)), p=self.dropout_2d)
		
		return F.sigmoid((self.OUTMASK(dec1)))
