from models.modules import common
import torch.nn as nn
import torch.nn.functional as F
import torch


def get_all_conv(net, conv_list = []):

    for name, layer in net._modules.items():
        if not isinstance(layer, nn.Conv2d):
            get_all_conv(layer, conv_list)
        elif isinstance(layer, nn.Conv2d):
           # it's a Conv layer. Register a hook
            conv_list.append(layer)

    for name, layer in net._modules.items():
        if not isinstance(layer, nn.ConvTranspose2d):
            get_all_conv(layer, conv_list)
        elif isinstance(layer, nn.ConvTranspose2d):
           # it's a Conv layer. Register a hook
            conv_list.append(layer)

    return conv_list


def relu(x, lambd):
    lambd = nn.functional.relu(lambd)
    return nn.functional.relu(x - lambd.to(x.device))


class adjoint_conv_op(nn.Module):
    # The adjoint of a conv module.
    def __init__(self, conv_op):
        super().__init__()
        in_channels = conv_op.out_channels
        out_channels = conv_op.in_channels
        kernel_size = conv_op.kernel_size
        padding = kernel_size[0] // 2

        # transpose convolution 
        self.transpose_conv = nn.ConvTranspose2d(in_channels, out_channels,  kernel_size=kernel_size, padding= padding, bias= False)
        
        # tie the weights of transpose convolution with convolution 
        self.transpose_conv.weight = conv_op.weight

    def forward(self, x):
        return self.transpose_conv(x)
    
class up_block(nn.Module):
    """
    A module that contains:
    (1) an up-sampling operation (implemented by bilinear interpolation or upsampling)
    (2) convolution operations
    """
        
    def __init__(self, kernel_size, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # the up-sampling operation
        self.up = nn.ConvTranspose2d(in_channels , in_channels-64, kernel_size=2, stride=2, bias= False)
        
        # the 2d convolution operation
        self.conv = nn.Conv2d((in_channels-64)*2, out_channels, kernel_size=kernel_size, padding= kernel_size // 2, bias= False)
 
    def forward(self, x1, x2):
        # print(x1.shape)
        x1 = self.up(x1)
        # print(x1.shape)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # input is CHW
        x = torch.cat([x2, x1], dim=1)
        # print(x.shape)
        return self.conv(x)
    

class adjoint_up_block(nn.Module):
    # adjoint of up_block module
    
    def __init__(self, up_block_model):
        super().__init__()
        
        # to construct the adjoint model, one should exclude additive biases and use transposed conv for upsampling.
        
        in_channels = up_block_model.out_channels
        out_channels = up_block_model.in_channels
        
        self.adjoint_conv_op = adjoint_conv_op(up_block_model.conv)
        self.adjoint_up =  nn.Conv2d(in_channels , in_channels // 2, kernel_size=2, stride=2, bias= False)
        self.adjoint_up.weight = up_block_model.up.weight
        
        
    def forward(self, x):
        x = self.adjoint_conv_op(x)
        # input is CHW
        x2 = x[:, :int(x.shape[1]/2), :, :]
        x1 = x[:, int(x.shape[1]/2):, :, :]
        x1 = self.adjoint_up(x1)
        return (x1, x2)


class out_conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(out_conv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias= False)
    def forward(self, x):
        return self.conv(x)    
    

class adjoint_out_conv(nn.Module):
    def __init__(self, out_conv_model):
        super().__init__()
        in_channels = out_conv_model.out_channels
        out_channels = out_conv_model.in_channels

        self.adjoint_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=1, bias= False)
        self.adjoint_conv.weight = out_conv_model.conv.weight

    def forward(self, x):
        return self.adjoint_conv(x)
    
    
class dictionary_model(nn.Module):
    def __init__(self,  kernel_size, hidden_layer_width_list, n_classes):
        super(dictionary_model, self).__init__()
        
        self.hidden_layer_width_list = hidden_layer_width_list
        
        in_out_list = [ [hidden_layer_width_list[i], hidden_layer_width_list[i+1]] for i in  range(len(hidden_layer_width_list) -1) ]

        self.num_hidden_layers = len(in_out_list)
        
        self.n_classes = n_classes

        # the initial convolution on the bottleneck layer
        self.bottleneck_conv = nn.Conv2d(hidden_layer_width_list[0], hidden_layer_width_list[0], kernel_size=kernel_size, padding= kernel_size // 2, bias= False)

        self.syn_up_list = []

        for layer_idx in range(self.num_hidden_layers):
            new_up_block = up_block(kernel_size, *in_out_list[layer_idx])
            self.syn_up_list.append(new_up_block)           
        
        self.syn_up_list = nn.Sequential( *self.syn_up_list )
        
        self.syn_outc = out_conv(hidden_layer_width_list[-1], n_classes)

    def forward(self, x_list):

        # x_list is ordered from wide-channel to thin-channel.
        num_res_levels = len(x_list)
                
#         x_prev = x_list[0]
        x_prev = self.bottleneck_conv(x_list[0])
    
        for i in range(1, num_res_levels):
            x = x_list[i] 
            syn_up = self.syn_up_list[i-1]
            x_prev = syn_up(x_prev, x)
            
        syn_output = self.syn_outc(x_prev)
        return syn_output

class adjoint_dictionary_model(nn.Module):
    def __init__(self, dictionary_model):
        super().__init__()
        
        
        self.adjoint_syn_outc = adjoint_out_conv(dictionary_model.syn_outc)
        self.adjoint_syn_bottleneck_conv = adjoint_conv_op(dictionary_model.bottleneck_conv)        
        
        self.adjoint_syn_up_list = []
        
        self.num_hidden_layers = dictionary_model.num_hidden_layers
        
        for layer_idx in range(dictionary_model.num_hidden_layers): 
            self.adjoint_syn_up_list.append(adjoint_up_block(dictionary_model.syn_up_list[layer_idx] ) )
            

    def forward(self, y):
        y = self.adjoint_syn_outc(y)
        x_list = []
        
        for layer_idx in range(self.num_hidden_layers-1, -1, -1):  
            adjoint_syn_up = self.adjoint_syn_up_list[layer_idx]   # 下采样
            y, x = adjoint_syn_up(y)
            x_list.append(x)
        y = self.adjoint_syn_bottleneck_conv(y)            
        x_list.append(y)
        x_list.reverse()
        return x_list 


## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size, reduction,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        #res = self.body(x).mul(self.res_scale)
        res += x
        return res

## Residual Group (RG)
class ResidualGroup(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, n_resblocks):
        super(ResidualGroup, self).__init__()
        modules_body = []
        modules_body = [
            RCAB(
                conv, n_feat, kernel_size, reduction=16, bias=True, bn=False, act=nn.ReLU(True), res_scale=1) \
            for _ in range(n_resblocks)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        # res += x
        return res

class DCN_Align(nn.Module):
    def __init__(self, nf=32, groups=4, kernel=3):
        super(DCN_Align, self).__init__()

        self.offset_conv1_1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True) 
        self.offset_conv2_1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        # down1    
        self.offset_conv3_1 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.offset_conv4_1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        # down2
        self.offset_conv6_1 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.offset_conv7_1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True) 
        # up2
        self.offset_conv1_2 = nn.Conv2d(nf*2, nf, 3, 1, 1, bias=True)
        self.offset_conv2_2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        # up1
        self.offset_conv3_2 = nn.Conv2d(nf*2, nf, 3, 1, 1, bias=True)
        self.offset_conv4_2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        self.dcnpack = DCN_sep(nf, nf, kernel, stride=1, padding=kernel//2, dilation=1,
                            deformable_groups=groups)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, fea1, fea2):
        '''align other neighboring frames to the reference frame in the feature level
        estimate offset bidirectionally
        '''
        offset = torch.cat([fea1, fea2], dim=1)
        offset = self.lrelu(self.offset_conv1_1(offset)) 
        offset1 = self.lrelu(self.offset_conv2_1(offset)) 
        # down1
        offset2 = self.lrelu(self.offset_conv3_1(offset1))
        offset2 = self.lrelu(self.offset_conv4_1(offset2))
        # down2   
        offset3 = self.lrelu(self.offset_conv6_1(offset2))
        offset3 = self.lrelu(self.offset_conv7_1(offset3))
        # up1
        offset = F.interpolate(offset3, scale_factor=2, mode='bilinear', align_corners=False)
        offset = self.lrelu(self.offset_conv1_2(torch.cat((offset, offset2), 1))) 
        offset = self.lrelu(self.offset_conv2_2(offset)) 
        # up2
        offset = F.interpolate(offset, scale_factor=2, mode='bilinear', align_corners=False)
        offset = self.lrelu(self.offset_conv3_2(torch.cat((offset, offset1), 1)))
        base_offset = self.offset_conv4_2(offset)
 
        aligned_fea = self.dcnpack(fea2, base_offset)

        return aligned_fea

class Offset_net(nn.Module):
    def __init__(self, nf=32):
        super(Offset_net, self).__init__()

        self.offset_conv1_1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True) 
        self.offset_conv2_1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        # down1    
        self.offset_conv3_1 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.offset_conv4_1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        # down2
        self.offset_conv6_1 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.offset_conv7_1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True) 
        # up2
        self.offset_conv1_2 = nn.Conv2d(nf*2, nf, 3, 1, 1, bias=True)
        self.offset_conv2_2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        # up1
        self.offset_conv3_2 = nn.Conv2d(nf*2, nf, 3, 1, 1, bias=True)
        self.offset_conv4_2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.offset_last = nn.Conv2d(nf, 2, 3, 1, 1, bias=True)
        self.mask_last = nn.Conv2d(nf, 1, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, fea1, fea2):
        '''align other neighboring frames to the reference frame in the feature level
        estimate offset bidirectionally
        '''
        offset = torch.cat([fea1, fea2], dim=1)
        offset = self.lrelu(self.offset_conv1_1(offset)) 
        offset1 = self.lrelu(self.offset_conv2_1(offset)) 
        # down1
        offset2 = self.lrelu(self.offset_conv3_1(offset1))
        offset2 = self.lrelu(self.offset_conv4_1(offset2))
        # down2   
        offset3 = self.lrelu(self.offset_conv6_1(offset2))
        offset3 = self.lrelu(self.offset_conv7_1(offset3))
        # up1
        offset = F.interpolate(offset3, scale_factor=2, mode='bilinear', align_corners=False)
        offset = self.lrelu(self.offset_conv1_2(torch.cat((offset, offset2), 1))) 
        offset = self.lrelu(self.offset_conv2_2(offset)) 
        # up2
        offset = F.interpolate(offset, scale_factor=2, mode='bilinear', align_corners=False)
        offset = self.lrelu(self.offset_conv3_2(torch.cat((offset, offset1), 1)))
        base_offset = self.lrelu(self.offset_conv4_2(offset))
        offset = self.offset_last(base_offset)
        mask = self.sigmoid(self.mask_last(base_offset))

        return offset, mask


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
backwarp_tenGrid = {}
def warp(tenInput, tenFlow):
    k = (str(tenFlow.device), str(tenFlow.size()))
    if k not in backwarp_tenGrid:
        tenHorizontal = torch.linspace(-1.0, 1.0, tenFlow.shape[3], device=device).view(
            1, 1, 1, tenFlow.shape[3]).expand(tenFlow.shape[0], -1, tenFlow.shape[2], -1)
        tenVertical = torch.linspace(-1.0, 1.0, tenFlow.shape[2], device=device).view(
            1, 1, tenFlow.shape[2], 1).expand(tenFlow.shape[0], -1, -1, tenFlow.shape[3])
        backwarp_tenGrid[k] = torch.cat(
            [tenHorizontal, tenVertical], 1).to(device)

    tenFlow = torch.cat([tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0),
                         tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0)], 1)

    g = (backwarp_tenGrid[k] + tenFlow).permute(0, 2, 3, 1)
    return torch.nn.functional.grid_sample(input=tenInput, grid=g, mode='bilinear', padding_mode='border', align_corners=True)


class ista_unet(nn.Module):
    def __init__(self, kernel_size=3, hidden_layer_width_list=[256,128,64], n_classes=64, ista_num_steps=6, lasso_lambda_scalar=0.01):

        super(ista_unet, self).__init__()

        self.n_classes = n_classes
        self.ista_num_steps = ista_num_steps
        self.lasso_lambda_scalar = lasso_lambda_scalar
        self.hidden_layer_width_list = hidden_layer_width_list
        self.num_layers = len(hidden_layer_width_list)
        
        # list to image parameters ----->  x
        self.encoder_dictionary_x = dictionary_model(kernel_size, hidden_layer_width_list, n_classes)
        [torch.nn.init.kaiming_uniform_(conv.weight, mode = 'fan_in', nonlinearity='linear') for conv in get_all_conv(self.encoder_dictionary_x)];
        # image to list parameters
        self.precond_encoder_dictionary_x = dictionary_model(kernel_size, hidden_layer_width_list, n_classes)
        self.precond_encoder_dictionary_x.load_state_dict(self.encoder_dictionary_x.state_dict())  # initialize with the same atoms
        self.adjoint_encoder_dictionary_x = adjoint_dictionary_model(self.precond_encoder_dictionary_x)    
        # list to image parameters ; for reconstruction
        self.decoder_dictionary_x = dictionary_model(kernel_size, hidden_layer_width_list, n_classes)
        self.decoder_dictionary_x.load_state_dict(self.encoder_dictionary_x.state_dict()) # initialize with the same atoms

        # list to image parameters  ----->  y
        self.encoder_dictionary_y = dictionary_model(kernel_size, hidden_layer_width_list, n_classes)
        [torch.nn.init.kaiming_uniform_(conv.weight, mode = 'fan_in', nonlinearity='linear') for conv in get_all_conv(self.encoder_dictionary_y)];
        # image to list parameters
        self.precond_encoder_dictionary_y = dictionary_model(kernel_size, hidden_layer_width_list, n_classes)
        self.precond_encoder_dictionary_y.load_state_dict(self.encoder_dictionary_y.state_dict())  # initialize with the same atoms
        self.adjoint_encoder_dictionary_y = adjoint_dictionary_model(self.precond_encoder_dictionary_y)  
        # list to image parameters ; for reconstruction 
        self.decoder_dictionary_y = dictionary_model(kernel_size, hidden_layer_width_list, n_classes)
        self.decoder_dictionary_y.load_state_dict(self.encoder_dictionary_y.state_dict()) # initialize with the same atoms


        # list to image parameters  ----->  z
        self.encoder_dictionary_z = dictionary_model(kernel_size, hidden_layer_width_list, n_classes)
        [torch.nn.init.kaiming_uniform_(conv.weight, mode = 'fan_in', nonlinearity='linear') for conv in get_all_conv(self.encoder_dictionary_z)];
        self.encoder_dictionary_zx = dictionary_model(kernel_size, hidden_layer_width_list, n_classes)
        [torch.nn.init.kaiming_uniform_(conv.weight, mode = 'fan_in', nonlinearity='linear') for conv in get_all_conv(self.encoder_dictionary_zx)];
        self.encoder_dictionary_zy = dictionary_model(kernel_size, hidden_layer_width_list, n_classes)
        [torch.nn.init.kaiming_uniform_(conv.weight, mode = 'fan_in', nonlinearity='linear') for conv in get_all_conv(self.encoder_dictionary_zy)];
        # image to list parameters
        self.precond_encoder_dictionary_z = dictionary_model(kernel_size, hidden_layer_width_list, n_classes)
        self.precond_encoder_dictionary_z.load_state_dict(self.encoder_dictionary_z.state_dict())  # initialize with the same atoms
        self.adjoint_encoder_dictionary_z = adjoint_dictionary_model(self.precond_encoder_dictionary_z)    
        # list to image parameters ; for reconstruction 
        self.decoder_dictionary_z = dictionary_model(kernel_size, hidden_layer_width_list, n_classes)
        # self.decoder_dictionary_z.load_state_dict(self.encoder_dictionary_z.state_dict()) # initialize with the same atoms
      

        with torch.no_grad():
            L_x = self.power_iteration_conv_model(self.encoder_dictionary_x, num_simulations = 20)     
        # a list of stepsizes and lambdas, one for each iteration
        self.ista_stepsize_iter_list_x = [nn.Parameter(torch.ones(1)/L_x) for i in range(ista_num_steps)]
        _lasso_lambda_iter_list_x = [[torch.nn.Parameter(lasso_lambda_scalar * torch.ones(1, width, 1, 1) ) for width in hidden_layer_width_list] for i in range(ista_num_steps)]         
        self.lasso_lambda_iter_list_x =  [item for sublist in _lasso_lambda_iter_list_x for item in sublist]

        with torch.no_grad():
            L_y = self.power_iteration_conv_model(self.encoder_dictionary_y, num_simulations = 20)
        self.ista_stepsize_iter_list_y = [nn.Parameter(torch.ones(1)/L_y) for i in range(ista_num_steps)]
        _lasso_lambda_iter_list_y = [[torch.nn.Parameter(lasso_lambda_scalar * torch.ones(1, width, 1, 1) ) for width in hidden_layer_width_list] for i in range(ista_num_steps)]         
        self.lasso_lambda_iter_list_y =  [item for sublist in _lasso_lambda_iter_list_y for item in sublist]

        with torch.no_grad():
            L_z = self.power_iteration_conv_model(self.encoder_dictionary_z, num_simulations = 20)
        self.ista_stepsize_iter_list_z = [nn.Parameter(torch.ones(1)/L_z) for i in range(ista_num_steps)]
        _lasso_lambda_iter_list_z = [[torch.nn.Parameter(lasso_lambda_scalar * torch.ones(1, width, 1, 1) ) for width in hidden_layer_width_list] for i in range(ista_num_steps)]         
        self.lasso_lambda_iter_list_z =  [item for sublist in _lasso_lambda_iter_list_z for item in sublist]

        self.resnet_x = nn.ModuleList([ResidualGroup(conv=common.default_conv, n_feat=self.hidden_layer_width_list[0], kernel_size=3, n_resblocks=5)])

        self.resnet_y = nn.ModuleList([ResidualGroup(conv=common.default_conv, n_feat=self.hidden_layer_width_list[0], kernel_size=3, n_resblocks=5)])

        self.resnet_z = nn.ModuleList([ResidualGroup(conv=common.default_conv, n_feat=self.hidden_layer_width_list[0], kernel_size=3, n_resblocks=5)])
    
        
        self.layer_in_x = nn.Conv2d(1, n_classes, 3, 1, 1)
        self.layer_in_y = nn.Conv2d(1, n_classes, 3, 1, 1)
        self.layer_in_z = nn.Conv2d(2, n_classes, 3, 1, 1)
        self.relu = nn.ReLU()
        self.cpmpress_z = nn.Conv2d(n_classes*2, n_classes, 3, 1, 1)
        self.rec_x_unique = nn.Conv2d(n_classes, 1, 3, 1, 1)
        self.rec_y_unique = nn.Conv2d(n_classes, 1, 3, 1, 1)
        self.rec_x_common = nn.Conv2d(n_classes, 1, 3, 1, 1)
        self.rec_y_common = nn.Conv2d(n_classes, 1, 3, 1, 1)

        self.net_T = Offset_net(nf=64)


    def forward(self, x, y, z):

        offset_yx = []
        # learned hyper-params
        ista_stepsize_iter_list_x = self.ista_stepsize_iter_list_x
        lasso_lambda_iter_list_x =  self.lasso_lambda_iter_list_x 
        ista_stepsize_iter_list_y = self.ista_stepsize_iter_list_y
        lasso_lambda_iter_list_y =  self.lasso_lambda_iter_list_y 
        ista_stepsize_iter_list_z = self.ista_stepsize_iter_list_z
        lasso_lambda_iter_list_z =  self.lasso_lambda_iter_list_z 


        ## initialize MS feature: x_list, y_list, z_list
        b,c,h,w = x.shape
        x_in = x.repeat(1,self.n_classes,1,1)
        y_in = y.repeat(1,self.n_classes,1,1)
        z_in = self.layer_in_z(z)
        adj_err_list_x  = self.adjoint_encoder_dictionary_x(x_in)
        adj_err_list_y  = self.adjoint_encoder_dictionary_y(y_in)
        adj_err_list_z  = self.adjoint_encoder_dictionary_z(z_in)
        x_list = []
        y_list = []
        z_list = []
        ista_stepsize_x = ista_stepsize_iter_list_x[0]
        ista_stepsize_y = ista_stepsize_iter_list_y[0]
        ista_stepsize_z = ista_stepsize_iter_list_z[0] 
        for i in range(self.num_layers):        
            lambd_x = ista_stepsize_x *  lasso_lambda_iter_list_x[i]
            x_list.append(relu(ista_stepsize_x.to(x.device) * adj_err_list_x[i], lambd = lambd_x.to(x.device)))
            lambd_y = ista_stepsize_y *  lasso_lambda_iter_list_y[i]
            y_list.append(relu(ista_stepsize_y.to(y.device) * adj_err_list_y[i], lambd = lambd_y.to(y.device)))
            lambd_z = ista_stepsize_z *  lasso_lambda_iter_list_z[i]
            z_list.append(relu(ista_stepsize_z.to(z.device) * adj_err_list_z[i], lambd = lambd_z.to(z.device)))

        ## calculate x*DT and y*HT
        xD_list = adj_err_list_x
        yH_list = adj_err_list_y

        # starting from the 2nd iteration
        for idx in range(1, self.ista_num_steps):
            err_x = self.encoder_dictionary_x(x_list) + self.encoder_dictionary_zx(z_list)
            ### alignment z->y  ##########
            cur_y_fea = self.encoder_dictionary_y(y_list)
            cur_z_fea = self.encoder_dictionary_zy(z_list)
            img_offset_zy, attention_map = self.net_T(cur_z_fea, cur_y_fea)
            cur_z_fea_aligned = warp(cur_z_fea, img_offset_zy)
            err_y = cur_y_fea + attention_map*cur_z_fea_aligned
            ###### No alignmnet #####
            # err_y = self.encoder_dictionary_y(y_list) + self.encoder_dictionary_zy(z_list)
            adj_err_list_x  = self.adjoint_encoder_dictionary_x(err_x)
            adj_err_list_y  = self.adjoint_encoder_dictionary_y(err_y)
            ista_stepsize_x = ista_stepsize_iter_list_x[idx]
            ista_stepsize_y = ista_stepsize_iter_list_y[idx]
            for i in range(self.num_layers):
                x_list[i] = x_list[i] - ista_stepsize_x.to(x.device) * (adj_err_list_x[i] - xD_list[i])
                y_list[i] = y_list[i] - ista_stepsize_y.to(y.device) * (adj_err_list_y[i] - yH_list[i])
                x_list[i] = self.resnet_x[i](self.relu(x_list[i]))
                y_list[i] = self.resnet_y[i](self.relu(y_list[i]))
            
            x_cur_unique = self.decoder_dictionary_x(x_list)
            x_cur_common = x_in - x_cur_unique
            y_cur_unique = self.decoder_dictionary_y(y_list)
            y_cur_common = y_in - y_cur_unique
            
            ### alignment y->x  ##########
            img_offset_yx, attention_map = self.net_T(y_cur_common, x_cur_common)
            offset_yx.append(img_offset_yx)
            y_cur_common_aligned = warp(y_cur_common, img_offset_yx)
            z = self.cpmpress_z(torch.cat([x_cur_common, y_cur_common_aligned*attention_map], 1))
            ###### No alignmnet #####
            # z = self.cpmpress_z(torch.cat([x_cur_common, y_cur_common], 1))
            
            err_z = self.encoder_dictionary_z(z_list) - z
            adj_err_list_z  = self.adjoint_encoder_dictionary_z(err_z)
            ista_stepsize_z = ista_stepsize_iter_list_z[idx]
            for i in range(self.num_layers):
                z_list[i] = z_list[i] - ista_stepsize_z.to(z.device) * adj_err_list_z[i]
                z_list[i] = self.resnet_z[i](self.relu(z_list[i]))
        
        ## reconstruction, channel = 64 
        common = self.decoder_dictionary_z(z_list)  
        x_cur_common = common 
        y_cur_common = warp(common, img_offset_zy)              
        x_unique = self.rec_x_unique(x_cur_unique)
        y_unique = self.rec_y_unique(y_cur_unique)
        x_common = self.rec_x_common(x_cur_common)
        y_common = self.rec_y_common(y_cur_common)

        x_rec = x_common + x_unique
        y_rec = y_common + y_unique

        # reg
        warped_y = warp(y, offset_yx[-1])

        return x_rec, y_rec, x_unique, x_common, y_unique, y_common, warped_y, offset_yx[-1]


    def initialize_sparse_codes(self, x, rand_bool = False):
        code_list = []

        num_samples =  x.shape[0]    
        input_spatial_dim_1 = x.shape[2]
        input_spatial_dim_2 = x.shape[3]

        if rand_bool:
            initializer = torch.rand
        else:
            initializer = torch.zeros

        for i in range(self.num_layers):
            feature_map_dim_1 = int(input_spatial_dim_1/  (2 ** i) )
            feature_map_dim_2 = int(input_spatial_dim_2/  (2 ** i) )
            code_tensor = initializer(num_samples, self.hidden_layer_width_list[self.num_layers-i-1],  feature_map_dim_1, feature_map_dim_2 )
            code_list.append(code_tensor)

        code_list.reverse() # order the code from low-spatial-dim to high-spatial-dim.
        return code_list

    def power_iteration_conv_model(self, conv_model, num_simulations: int):

        eigen_vec_list = self.initialize_sparse_codes(x = torch.zeros(1, 3, 64, 64), rand_bool = True)

        adjoint_conv_model = adjoint_dictionary_model(conv_model)

        for _ in range(num_simulations):
            # calculate the matrix-by-vector product Ab
            eigen_vec_list = adjoint_conv_model(conv_model(eigen_vec_list))
            # calculate the norm
            flatten_x_norm = torch.norm(torch.cat([x.flatten() for x in eigen_vec_list ]) )
            # re-normalize the vector
            eigen_vec_list = [x/ flatten_x_norm for x in eigen_vec_list] 

        eigen_vecs_flatten = torch.cat([x.flatten() for x in eigen_vec_list])

        linear_trans_eigen_vecs_list = adjoint_conv_model(conv_model(eigen_vec_list ))

        linear_trans_eigen_vecs_list_flatten = torch.cat([x.flatten() for x in linear_trans_eigen_vecs_list] )

        numerator = torch.dot(eigen_vecs_flatten, linear_trans_eigen_vecs_list_flatten)

        denominator = torch.dot(eigen_vecs_flatten, eigen_vecs_flatten)

        eigenvalue = numerator / denominator
        return eigenvalue


class decoder(nn.Module):
    def __init__(self, in_channel, channel_fea):
        super(decoder, self).__init__()
        self.channel = in_channel
        self.kernel_size = 3
        self.padding = self.kernel_size//2
        self.filters = channel_fea
        self.conv_1 = nn.Conv2d(in_channels=self.filters*2, out_channels=self.filters, kernel_size=self.kernel_size,
                                stride=1, padding=self.padding, bias=False)
        nn.init.xavier_uniform_(self.conv_1.weight.data)
        self.conv_2 = nn.Conv2d(in_channels=self.filters, out_channels=self.channel, kernel_size=self.kernel_size,
                                stride=1, padding=self.padding, bias=False)
        nn.init.xavier_uniform_(self.conv_2.weight.data)
        self.relu = nn.ReLU()

    def forward(self, x):
        rec = self.conv_2(self.relu(self.conv_1(x)))
        return rec

class CDic_Align(nn.Module):
    def __init__(self):
        super(CDic_Align, self).__init__()

        self.in_channel = 1
        self.channel_fea = 64

        self.predict_ista = ista_unet(kernel_size=3, hidden_layer_width_list=[64], n_classes=self.channel_fea, ista_num_steps=5)

        self.decoder = decoder(self.in_channel, self.channel_fea)

    def forward(self, x, y):
        x_rec, y_rec, x_cur_common, x_cur_unique, y_cur_common, y_cur_unique, warped_y, offset = self.predict_ista(x, y, torch.cat([x,y], 1))

        return x_rec, y_rec, x_cur_common, x_cur_unique, y_cur_common, y_cur_unique, warped_y, offset
        # return x_rec

   



