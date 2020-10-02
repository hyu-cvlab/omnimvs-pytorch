# network.py
# architecture submitted to PAMI
# Author: Changhee Won (changhee.1.won@gmail.com)
#
#
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from module.basic import *

class FeatureLayers(torch.nn.Module):
    def __init__(self, CH=32, use_rgb=False):
        super(FeatureLayers, self).__init__()
        layers = []
        in_channel = 3 if use_rgb else 1
        layers.append(Conv2D(in_channel,CH,5,2,2)) # conv[1]
        layers += [Conv2D(CH,CH,3,1,1) for _ in range(10)] # conv[2-11]
        for d in range(2,5): # conv[12-17]
            layers += [Conv2D(CH,CH,3,1,d,dilation=d) for _ in range(2)]
        layers.append(Conv2D(CH,CH,3,1,1,bn=False,relu=False)) # conv[18]
        self.layers = torch.nn.ModuleList(layers)
    
    def forward(self, im):
        x = self.layers[0](im)
        for i in range(1,17,2):
            x_ = self.layers[i](x)
            x = self.layers[i+1](x_,residual=x)
        x = self.layers[17](x)
        return x

class SphericalSweep(torch.nn.Module):
    def __init__(self, CH=32):
        super(SphericalSweep, self).__init__()
        self.transfer_conv = \
            Conv2D(CH,CH,3,2,1,bn=False,relu=False)

    def forward(self, feature, grid):
        num_invdepth = grid.shape[0]
        sweep = [F.grid_sample(feature, grid[d,...].unsqueeze(0),
            align_corners=True) for d in range(0, num_invdepth)]
        sweep = torch.cat(sweep, 0) # -> N/2 x CH x H x W
        spherical_feature = self.transfer_conv(sweep)
        return spherical_feature

class CostCompute(torch.nn.Module):
    def __init__(self, CH=32):
        super(CostCompute, self).__init__()
        CH *= 2
        self.fusion = Conv3D(2*CH,CH,3,1,1)
        convs = []
        convs += [Conv3D(CH,CH,3,1,1),
                        Conv3D(CH,CH,3,1,1),
                        Conv3D(CH,CH,3,1,1)]
        convs += [Conv3D(CH,2*CH,3,2,1),
                        Conv3D(2*CH,2*CH,3,1,1),
                        Conv3D(2*CH,2*CH,3,1,1)]
        convs += [Conv3D(2*CH,2*CH,3,2,1),
                        Conv3D(2*CH,2*CH,3,1,1),
                        Conv3D(2*CH,2*CH,3,1,1)]
        convs += [Conv3D(2*CH,2*CH,3,2,1),
                        Conv3D(2*CH,2*CH,3,1,1),
                        Conv3D(2*CH,2*CH,3,1,1)]
        convs += [Conv3D(2*CH,4*CH,3,2,1),
                        Conv3D(4*CH,4*CH,3,1,1),
                        Conv3D(4*CH,4*CH,3,1,1)]
        self.convs = torch.nn.ModuleList(convs)
        self.deconv1 = DeConv3D(4*CH,2*CH,3,2,1,out_pad=1)
        self.deconv2 = DeConv3D(2*CH,2*CH,3,2,1,out_pad=1)
        self.deconv3 = DeConv3D(2*CH,2*CH,3,2,1,out_pad=1)
        self.deconv4 = DeConv3D(2*CH,CH,3,2,1,out_pad=1)
        self.deconv5 = DeConv3D(CH,1,3,2,1,out_pad=1,bn=False,relu=False)

    def forward(self, feats):
        c = self.fusion(feats)
        c = self.convs[0](c)
        c1 = self.convs[1](c)
        c1 = self.convs[2](c1)
        c = self.convs[3](c)
        c2 = self.convs[4](c)
        c2 = self.convs[5](c2)
        c = self.convs[6](c)
        c3 = self.convs[7](c)
        c3 = self.convs[8](c3)
        c = self.convs[9](c)
        c4 = self.convs[10](c)
        c4 = self.convs[11](c4)
        c = self.convs[12](c)
        c5 = self.convs[13](c)
        c5 = self.convs[14](c5)
        c = self.deconv1(c5, residual=c4)
        c = self.deconv2(c, residual=c3)
        c = self.deconv3(c, residual=c2)
        c = self.deconv4(c, residual=c1)
        costs = self.deconv5(c)
        return costs

class OmniMVSNet(torch.nn.Module):

    def __init__(self, varargin=None):
        super(OmniMVSNet, self).__init__()
        opts = Edict()
        opts.CH = 32
        opts.num_invdepth = 192
        opts.use_rgb = False
        self.opts = argparse(opts, varargin)
        self.feature_layers = FeatureLayers(self.opts.CH, self.opts.use_rgb)
        self.spherical_sweep = SphericalSweep(self.opts.CH)
        self.cost_computes = CostCompute(self.opts.CH)
        self.disps = torch.arange(0, self.opts.num_invdepth,
            requires_grad=False).view((1, -1, 1, 1)).float().cuda()
        
    def forward(self, imgs, grids, upsample=False, out_cost=False):
        feats = [self.feature_layers(x) for x in imgs]
        spherical_feats = torch.cat([self.spherical_sweep(feats[i], grids[i]) \
            for i in range(len(imgs))],1).permute([1,0,2,3])
        spherical_feats = torch.unsqueeze(spherical_feats, 0)
        costs = self.cost_computes(spherical_feats)
        # softargmax
        if upsample:
            costs = F.interpolate(costs.squeeze(1), scale_factor=2,
                    mode='bilinear', align_corners=True)
        else:
            costs = torch.squeeze(costs, 1)
        prob = F.softmax(costs, 1)
        disp = torch.mul(prob, self.disps)
        disp = torch.sum(disp, 1)
        if out_cost:
            return disp, prob.squeeze(), costs.squeeze()
        else:
            return disp
        
        
