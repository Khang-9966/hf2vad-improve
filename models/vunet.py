import torch
from torch import nn
from torch.nn import ModuleDict, ModuleList, Conv2d
import numpy as np
from edflow.util import retrieve
from models.basic_modules import (
    VUnetResnetBlock,
    Upsample,
    Downsample,
    NormConv2d,
    SpaceToDepth,
    DepthToSpace,
)
import math
from models.basic_modules import *
from models.simple_unet import SimpleUnet
# relu based hard shrinkage function, only works for positive values
def hard_shrink_relu(input, lambd=0., epsilon=1e-12):
    output = (F.relu(input - lambd) * input) / (torch.abs(input - lambd) + epsilon)
    return output

class Memory(nn.Module):
    def __init__(self, num_slots, slot_dim, shrink_thres=0.0025):
        super(Memory, self).__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim

        self.memMatrix = nn.Parameter(torch.empty(num_slots, slot_dim))  # M,C
        self.shrink_thres = shrink_thres

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.memMatrix.size(1))
        self.memMatrix.data.uniform_(-stdv, stdv)

    def forward(self, x):
        """
        :param x: query features with size [N,C], where N is the number of query items,
                  C is same as dimension of memory slot

        :return: query output retrieved from memory, with the same size as x.
        """
        # dot product
        att_weight = F.linear(input=x, weight=self.memMatrix)  # [N,C] by [M,C]^T --> [N,M]
        att_weight = F.softmax(att_weight, dim=1)  # NxM

        # if use hard shrinkage
        if self.shrink_thres > 0:
            att_weight = hard_shrink_relu(att_weight, lambd=self.shrink_thres)  # [N,M]
            # normalize
            att_weight = F.normalize(att_weight, p=1, dim=1)  # [N,M]

        # out slot
        out = F.linear(att_weight, self.memMatrix.permute(1, 0))  # [N,M] by [M,C]  --> [N,C]

        return dict(out=out, att_weight=att_weight)

class VUnetEncoder(nn.Module):
    def __init__(
            self,
            n_stages,
            nf_in=3,
            nf_start=64,
            nf_max=128,
            n_rnb=2,
            conv_layer=NormConv2d,
            dropout_prob=0.0,
    ):
        super().__init__()
        self.in_op = conv_layer(nf_in, nf_start, kernel_size=1)
        nf = nf_start
        self.blocks = ModuleDict()
        self.downs = ModuleDict()
        self.n_rnb = n_rnb
        self.n_stages = n_stages
        for i_s in range(self.n_stages):
            # prepare resnet blocks per stage
            if i_s > 0:
                self.downs.update(
                    {
                        f"s{i_s + 1}": Downsample(
                            nf, min(2 * nf, nf_max), conv_layer=conv_layer
                        )
                    }
                )
                nf = min(2 * nf, nf_max)

            for ir in range(self.n_rnb):
                stage = f"s{i_s + 1}_{ir + 1}"
                self.blocks.update(
                    {
                        stage: VUnetResnetBlock(
                            nf, conv_layer=conv_layer, dropout_prob=dropout_prob
                        )
                    }
                )

    def forward(self, x):
        out = {}
        h = self.in_op(x)
        for ir in range(self.n_rnb):
            h = self.blocks[f"s1_{ir + 1}"](h)
            out[f"s1_{ir + 1}"] = h

        for i_s in range(1, self.n_stages):

            h = self.downs[f"s{i_s + 1}"](h)

            for ir in range(self.n_rnb):
                stage = f"s{i_s + 1}_{ir + 1}"
                h = self.blocks[stage](h)
                out[stage] = h

        return out


class ZConverter(nn.Module):
    def __init__(self, n_stages, nf, device, conv_layer=NormConv2d, dropout_prob=0.0):
        super().__init__()
        self.n_stages = n_stages
        self.device = device
        self.blocks = ModuleList()
        for i in range(3):  # three res block
            self.blocks.append(
                VUnetResnetBlock(
                    nf, use_skip=True, conv_layer=conv_layer, dropout_prob=dropout_prob
                )
            )
        self.conv1x1 = conv_layer(nf, nf, 1)
        self.up = Upsample(in_channels=nf, out_channels=nf, conv_layer=conv_layer)
        self.channel_norm = conv_layer(2 * nf, nf, 1)

        self.d2s = DepthToSpace(block_size=2)
        self.s2d = SpaceToDepth(block_size=2)

    def forward(self, x_f):
        params = {}
        zs = {}
        h = self.conv1x1(x_f[f"s{self.n_stages}_2"])
        # distribution inference
        for n, i_s in enumerate(range(self.n_stages, self.n_stages - 2, -1)):
            stage = f"s{i_s}"

            spatial_size = x_f[stage + "_2"].shape[-1]
            spatial_stage = "%dby%d" % (spatial_size, spatial_size)

            h = self.blocks[2 * n](h, x_f[stage + "_2"])

            params[spatial_stage] = h  # parameters of posterior
            z = self._latent_sample(params[spatial_stage])  # sampling
            zs[spatial_stage] = z
            # post
            if n == 0:
                gz = torch.cat([x_f[stage + "_1"], z], dim=1)
                gz = self.channel_norm(gz)
                h = self.blocks[2 * n + 1](h, gz)
                h = self.up(h)

        return params, zs

    def _latent_sample(self, mean):
        normal_sample = torch.randn(mean.size()).to(self.device)
        return mean + normal_sample


class VUnetDecoder(nn.Module):
    def __init__(
            self,
            n_stages,
            nf=128,
            nf_out=3,
            n_rnb=2,
            conv_layer=NormConv2d,
            spatial_size=256,
            final_act=True,
            dropout_prob=0.0,
    ):
        super().__init__()

        self.final_act = final_act
        self.blocks = ModuleDict()
        self.ups = ModuleDict()
        self.n_stages = n_stages
        self.n_rnb = n_rnb
        for i_s in range(self.n_stages - 2, 0, -1):
            # for final stage, bisect number of filters
            if i_s == 1:
                # upsampling operations
                self.ups.update(
                    {
                        f"s{i_s + 1}": Upsample(
                            in_channels=nf, out_channels=nf // 2, conv_layer=conv_layer,
                        )
                    }
                )
                nf = nf // 2
            else:
                # upsampling operations
                self.ups.update(
                    {
                        f"s{i_s + 1}": Upsample(
                            in_channels=nf, out_channels=nf, conv_layer=conv_layer,
                        )
                    }
                )

            # resnet blocks
            for ir in range(self.n_rnb, 0, -1):
                stage = f"s{i_s}_{ir}"
                skip_ = True
                if stage == "s1_2" or stage == "s1_1" : # or stage == "s2_1" :
                  skip_ = False
                  print(stage,skip_)
                self.blocks.update(
                    {
                        stage: VUnetResnetBlock(
                            nf,
                            use_skip=skip_,
                            conv_layer=conv_layer,
                            dropout_prob=dropout_prob,
                        )
                    }
                )

        # final 1x1 convolution
        self.final_layer = conv_layer(nf, nf_out, kernel_size=1)

        # conditionally: set final activation
        # if self.final_act:
        self.final_act = nn.Sigmoid()

    def forward(self, x, skips):
        out = x
        for i_s in range(self.n_stages - 2, 0, -1):
            out = self.ups[f"s{i_s + 1}"](out)

            for ir in range(self.n_rnb, 0, -1):
                stage = f"s{i_s}_{ir}"
                out = self.blocks[stage](out, skips[stage])

        out = self.final_layer(out)
        if self.final_act:  # final activation
            out = self.final_act(out)
        return out


class VUnetBottleneck(nn.Module):
    def __init__(
            self,
            n_stages,
            nf,
            device,
            n_rnb=2,
            n_auto_groups=4,
            conv_layer=NormConv2d,
            dropout_prob=0.0,
    ):
        super().__init__()
        self.device = device  # gpu or cpu
        self.blocks = ModuleDict()
        self.channel_norm = ModuleDict()
        self.conv1x1 = conv_layer(nf, nf, 1)
        self.up = Upsample(in_channels=nf, out_channels=nf, conv_layer=conv_layer)
        self.depth_to_space = DepthToSpace(block_size=2)
        self.space_to_depth = SpaceToDepth(block_size=2)
        self.n_stages = n_stages
        self.n_rnb = n_rnb  # 2
        # number of autoregressively modeled groups
        self.n_auto_groups = n_auto_groups
        for i_s in range(self.n_stages, self.n_stages - 2, -1):  # only 2 lyrs
            self.channel_norm.update({f"s{i_s}": conv_layer(2 * nf, nf, 1)})
            for ir in range(self.n_rnb):
                self.blocks.update(
                    {
                        f"s{i_s}_{ir + 1}": VUnetResnetBlock(
                            nf,
                            use_skip=True,
                            conv_layer=conv_layer,
                            dropout_prob=dropout_prob,
                        )
                    }
                )

        self.auto_blocks = ModuleList()
        # model the autoregressively groups rnb
        for i_a in range(4):
            if i_a < 1:
                self.auto_blocks.append(
                    VUnetResnetBlock(
                        nf, conv_layer=conv_layer, dropout_prob=dropout_prob
                    )
                )
                self.param_converter = conv_layer(4 * nf, nf, kernel_size=1)
            else:
                self.auto_blocks.append(
                    VUnetResnetBlock(
                        nf,
                        use_skip=True,
                        conv_layer=conv_layer,
                        dropout_prob=dropout_prob,
                    )
                )

    def forward(self, x_e, z_post):
        p_params = {}
        z_prior = {}
        # use z from posterior
        use_z = True

        h = self.conv1x1(x_e[f"s{self.n_stages}_2"])

        for i_s in range(self.n_stages, self.n_stages - 2, -1):
            stage = f"s{i_s}"
            spatial_size = x_e[stage + "_2"].shape[-1]
            spatial_stage = "%dby%d" % (spatial_size, spatial_size)

            h = self.blocks[stage + "_2"](h, x_e[stage + "_2"])

            if spatial_size == 1:
                p_params[spatial_stage] = h
                # posterior_params[stage] = z_post[stage + "_2"]
                prior_samples = self._latent_sample(p_params[spatial_stage])

                z_prior[spatial_stage] = prior_samples
                # posterior_samples = self._latent_sample(posterior_params[stage])
            else:
                if use_z:
                    z_flat = (
                        self.space_to_depth(z_post[spatial_stage])
                        if z_post[spatial_stage].shape[2] > 1
                        else z_post[spatial_stage]
                    )
                    sec_size = z_flat.shape[1] // 4
                    z_groups = torch.split(
                        z_flat, [sec_size, sec_size, sec_size, sec_size], dim=1
                    )  # split into 4 groups

                param_groups = []
                sample_groups = []

                param_features = self.auto_blocks[0](h)
                param_features = self.space_to_depth(param_features)
                # convert to fit depth
                param_features = self.param_converter(param_features)

                for i_a in range(len(self.auto_blocks)):
                    param_groups.append(param_features)

                    prior_samples = self._latent_sample(param_groups[-1])

                    sample_groups.append(prior_samples)

                    if i_a + 1 < len(self.auto_blocks):  # 0,1,2
                        if use_z:
                            feedback = z_groups[i_a]
                        else:
                            feedback = prior_samples

                        param_features = self.auto_blocks[i_a + 1](param_features, feedback)

                p_params_stage = self.__merge_groups(param_groups)
                prior_samples = self.__merge_groups(sample_groups)
                p_params[spatial_stage] = p_params_stage  # prior params
                z_prior[spatial_stage] = prior_samples

            if use_z:
                z = (
                    self.depth_to_space(z_post[spatial_stage])
                    if z_post[spatial_stage].shape[-1] != h.shape[-1]
                    else z_post[spatial_stage]
                )
            else:
                z = prior_samples

            gz = torch.cat([x_e[stage + "_1"], z], dim=1)  # cat z and E(y_{1:t})
            gz = self.channel_norm[stage](gz)
            h = self.blocks[stage + "_1"](h, gz)
  
            if i_s == self.n_stages:
                h = self.up(h)

        return h, p_params, z_prior

    def __split_groups(self, x):
        # split along channel axis
        sec_size = x.shape[1] // 4
        return torch.split(
            self.space_to_depth(x), [sec_size, sec_size, sec_size, sec_size], dim=1,
        )

    def __merge_groups(self, x):
        # merge groups along channel axis
        return self.depth_to_space(torch.cat(x, dim=1))

    def _latent_sample(self, mean):
        normal_sample = torch.randn(mean.size()).to(self.device)
        return mean + normal_sample


class VUnet(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        final_act = retrieve(config, "model_paras/final_act", default=False)
        nf_max = retrieve(config, "model_paras/nf_max", default=128)
        nf_start = retrieve(config, "model_paras/nf_start", default=64)
        spatial_size = retrieve(config, "model_paras/spatial_size", default=256)
        dropout_prob = retrieve(config, "model_paras/dropout_prob", default=0.1)
        img_channels = retrieve(config, "model_paras/img_channels", default=3)
        motion_channels = retrieve(config, "model_paras/motion_channels", default=2)
        clip_hist = retrieve(config, "model_paras/clip_hist", default=4)
        clip_pred = retrieve(config, "model_paras/clip_pred", default=1)
        num_flows = retrieve(config, "model_paras/num_flows", default=4)
        device = retrieve(config, "device", default="cuda:0")
        output_channels = img_channels * clip_pred

        # define required parameters
        n_stages = 1 + int(np.round(np.log2(spatial_size))) - 2

        conv_layer_type = Conv2d if final_act else NormConv2d

        # prosterior p( z | x_{1:t},y_{1:t} )
        self.f_phi = VUnetEncoder(
            n_stages=n_stages,
            nf_in=img_channels * clip_hist + motion_channels * num_flows,
            nf_start=nf_start,
            nf_max=nf_max,
            conv_layer=conv_layer_type,
            dropout_prob=dropout_prob,
        )

        # prior p(z|y_{1:t})
        self.e_theta = VUnetEncoder(
            n_stages=n_stages,
            nf_in=motion_channels * num_flows,
            nf_start=nf_start,
            nf_max=nf_max,
            conv_layer=conv_layer_type,
            dropout_prob=dropout_prob,
        )

        # zconverter
        self.zc = ZConverter(
            n_stages=n_stages,
            nf=nf_max,
            device=device,
            conv_layer=conv_layer_type,
            dropout_prob=dropout_prob,
        )

        # bottleneck
        self.bottleneck = VUnetBottleneck(
            n_stages=n_stages,
            nf=nf_max,
            device=device,
            conv_layer=conv_layer_type,
            dropout_prob=dropout_prob,
        )

        # decoder
        self.decoder = VUnetDecoder(
            n_stages=n_stages,
            nf=nf_max,
            nf_out=output_channels,
            conv_layer=conv_layer_type,
            spatial_size=spatial_size,
            final_act=final_act,
            dropout_prob=dropout_prob,
        )
        
        self.saved_tensors = None
     
        self.s2_1_unet = SimpleUnet(features_root=128, raw_channel_num=128)
        self.s2_2_unet = SimpleUnet(features_root=128, raw_channel_num=128)
        # self.s1_1_unet = SimpleUnet(features_root=64, raw_channel_num=64)
        # self.s1_2_unet = SimpleUnet(features_root=64, raw_channel_num=64)

    def mem_forward(self,mem,x):
      # flatten [bs,C,H,W] --> [bs,C*H*W]
      bs, C, H, W = x.shape
      x3 = x.view(bs, -1)
      mem_out = mem(x3)
      x = mem_out["out"]
      # attention weight size [bs,N], N is num_slots
      att_weight = mem_out["att_weight"]
      # unflatten
      x = x.view(bs, C, H, W)
      return x,att_weight


    def forward(self, inputs, mode="train"):
        '''
        Two possible usage：

        1. train stage, sampling z from the posterior p(z | x_{1:t},y_{1:t} )
        2. test stage, use the mean of the posterior as sampled z
        '''
        # posterior
        x_f_in = torch.cat((inputs['appearance'], inputs['motion']), dim=1)
        x_f = self.f_phi(x_f_in)
        # params and samples of the posterior
        q_means, zs = self.zc(x_f)
        loss_sparsity = 0
        # encoding features of flows
        x_e = self.e_theta(inputs['motion'])

        x_f['s2_2']  = self.s2_2_unet(x_f['s2_2'])
        x_f['s2_1']  = self.s2_1_unet(x_f['s2_1'])
        # x_f['s1_2']  = self.s1_2_unet(x_f['s1_2'])
        # x_f['s1_1']  = self.s1_1_unet(x_f['s1_1'])

        # att_w_s3_1 = torch.cat([att_w_s3_1], dim=0)
        # att_w_s4_1 = torch.cat([att_w_s4_1], dim=0)

        # loss_sparsity = torch.mean(
        #     torch.sum(-att_w_s3_1 * torch.log(att_w_s3_1 + 1e-12), dim=1)
        # ) + torch.mean(
        #     torch.sum(-att_w_s4_1 * torch.log(att_w_s4_1 + 1e-12), dim=1)
        # )

        if mode == "train":
            out_b, p_means, ps = self.bottleneck(x_e, zs)  # h, p_params, z_prior
        else:
            out_b, p_means, ps = self.bottleneck(x_e, q_means)
        
        # print([name for name in x_f])
        # for name in x_f:
        #   print(name)
        #   print(x_f[name].shape)

        # decode, feed in the output of bottleneck and the skip connections
        out_img = self.decoder(out_b, x_f)

        self.saved_tensors = dict(q_means=q_means, p_means=p_means)
        return out_img,loss_sparsity
