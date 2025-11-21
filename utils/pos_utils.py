import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.rigid_utils import exp_se3
from typing import Optional, Tuple, List
from utils.encoders import PositionalEncoder, FreqGate

def get_embedder(multires, i=1):
    if i == -1:
        return nn.Identity(), 3

    embed_kwargs = {
        'include_input': True,
        'input_dims': i,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj: eo.embed(x)
    return embed, embedder_obj.out_dim


class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)
    
class DeformNetwork(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, output_ch=59, multires=10, is_blender=True, is_6dof=False):
        super(DeformNetwork, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.output_ch = output_ch
        self.t_multires = 6 if is_blender else 10
        self.skips = [D // 2]
        # self.skips = [D // 3, (D * 2) // 3]

        # self.embed_rx_fn, rx_input_ch = get_embedder(self.t_multires, 3)
        self.embed_fn, xyz_input_ch = get_embedder(multires, 3)
        self.embed_info_fn, info_input_ch = get_embedder(10, 7)  # 4 for orientation, 3 for position
        # self.embed_info_fn, info_input_ch = get_embedder(self.t_multires, 7)
        # self.embed_rxdir_fn, dir_input_ch = get_embedder(4, 3)
        self.input_ch = xyz_input_ch # + rx_input_ch # + info_input_ch

        if is_blender:
           
            # self.rx_out = 90
            # self.dir_out = 90
            self.info_out = 128
            
            self.infonet = nn.Sequential(
                nn.Linear(info_input_ch, 256), nn.ReLU(inplace=True),
                nn.Linear(256, self.info_out))
            
            # self.rxnet = nn.Sequential(
            #     nn.Linear(rx_input_ch, 256), nn.ReLU(inplace=True),
            #     nn.Linear(256, self.rx_out))
            
            # self.dirnet = nn.Sequential(
            #     nn.Linear(dir_input_ch, 256), nn.ReLU(inplace=True),
            #     nn.Linear(256, self.dir_out))
            
            self.linear = nn.ModuleList(
                [nn.Linear(xyz_input_ch + self.info_out, W)] + [
                    nn.Linear(W, W) if i not in self.skips else nn.Linear(W + xyz_input_ch + self.info_out, W)
                    for i in range(D - 1)]
            )

        else:
            self.linear = nn.ModuleList(
                [nn.Linear(self.input_ch, W)] + [
                    nn.Linear(W, W) if i not in self.skips else nn.Linear(W + self.input_ch, W)
                    for i in range(D - 1)]
            )

        self.is_blender = is_blender
        self.is_6dof = is_6dof

        self.gaussian_signal = nn.Sequential(
            nn.Linear(W, W // 2),  
            nn.ReLU(),             
            nn.Linear(W // 2, 1)  
        )

        self.gaussian_phase = nn.Sequential(
            nn.Linear(W, W // 2),  
            nn.ReLU(),             
            nn.Linear(W // 2, 1)  
        )

    def forward(self, x, rx_info):
        info_emb = self.embed_info_fn(rx_info)
        if self.is_blender:
            info_emb = self.infonet(info_emb)  
        x_emb = self.embed_fn(x)


        h = torch.cat([x_emb, info_emb], dim=-1)
        for i, l in enumerate(self.linear):
            h = self.linear[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([x_emb, info_emb, h], -1)

        signal_real = self.gaussian_signal(h)
        signal_img = self.gaussian_phase(h)
        signal_complex = signal_real*torch.exp(1j*signal_img)
        signal = torch.abs(signal_complex)

        return signal
    
class DeformNetwork_real_only(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, output_ch=59, multires=10, is_blender=True, is_6dof=False):
        super(DeformNetwork_real_only, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.output_ch = output_ch
        self.t_multires = 6 if is_blender else 10
        self.skips = [D // 2]
        # self.skips = [D // 3, (D * 2) // 3]

        self.embed_rx_fn, rx_input_ch = get_embedder(self.t_multires, 3)
        self.embed_fn, xyz_input_ch = get_embedder(multires, 3)
        self.input_ch = xyz_input_ch + rx_input_ch 
        if is_blender:
            self.rx_out = 128
            
            self.rxnet = nn.Sequential(
                nn.Linear(rx_input_ch, 256), nn.ReLU(inplace=True),
                nn.Linear(256, self.rx_out))

            self.linear = nn.ModuleList(
                [nn.Linear(xyz_input_ch + self.rx_out, W)] + [
                    nn.Linear(W, W) if i not in self.skips else nn.Linear(W + xyz_input_ch + self.rx_out, W)
                    for i in range(D - 1)]
            )

        else:
            self.linear = nn.ModuleList(
                [nn.Linear(self.input_ch, W)] + [
                    nn.Linear(W, W) if i not in self.skips else nn.Linear(W + self.input_ch, W)
                    for i in range(D - 1)]
            )

        self.is_blender = is_blender
        self.is_6dof = is_6dof

        self.gaussian_signal = nn.Sequential(
            nn.Linear(W, W // 2),  
            nn.ReLU(),             
            nn.Linear(W // 2, 1)   
        )


    def forward(self, x, rx_pos):
        rx_emb = self.embed_rx_fn(rx_pos)
        if self.is_blender:
            rx_emb = self.rxnet(rx_emb)  
        x_emb = self.embed_fn(x)


        h = torch.cat([x_emb, rx_emb], dim=-1)
        for i, l in enumerate(self.linear):
            h = self.linear[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([x_emb, rx_emb, h], -1)

        signal_real = self.gaussian_signal(h)

        return signal_real
    
class DeformNetwork_HumanSimple(nn.Module):
    def __init__(self, D=8, W=256, multires_xyz=10, multires_rx=6, multires_man=10,
                 is_blender=True):
        super().__init__()
        self.D = D
        self.W = W
        self.is_blender = is_blender
        self.skips = [D // 2]

        self.embed_x_fn,   x_ch   = get_embedder(multires_xyz, 3)
        self.embed_rx_fn,  rx_ch  = get_embedder(multires_rx,  3)
        self.embed_man_fn, man_ch = get_embedder(multires_man, 7)

        self.rx_out = 64
        self.rxnet = nn.Sequential(
            nn.Linear(rx_ch, 128), nn.ReLU(inplace=True),
            nn.Linear(128, self.rx_out)
        )
        
        self.info_out = 128
        self.infonet = nn.Sequential(
            nn.Linear(man_ch, 256), nn.ReLU(inplace=True),
            nn.Linear(256, self.info_out)
        )

        trunk_in = x_ch + self.info_out + self.rx_out
        self.linear = nn.ModuleList(
            [nn.Linear(trunk_in, W)] + [
                nn.Linear(W, W) if i not in self.skips else nn.Linear(W + trunk_in, W)
                for i in range(D - 1)
            ]
        )

        self.gaussian_warp     = nn.Linear(W, 3)  
        self.gaussian_scaling  = nn.Linear(W, 3)  
        self.gaussian_rotation = nn.Linear(W, 4)  
        self.gaussian_signal   = nn.Sequential(    
            nn.Linear(W, W // 2), nn.ReLU(inplace=True),
            nn.Linear(W // 2, 1)
        )

    def forward(self, x, rx_pos, man_info):
        x_emb   = self.embed_x_fn(x)           
        rx_emb  = self.embed_rx_fn(rx_pos)     
        rx_emb = self.rxnet(rx_emb)          
        info_emb= self.embed_man_fn(man_info)  
        info_emb= self.infonet(info_emb)       

        h_in = torch.cat([x_emb, rx_emb, info_emb], dim=-1)
        h = h_in
        for i, layer in enumerate(self.linear):
            h = layer(h)
            h = F.relu(h, inplace=True)
            if i in self.skips:
                h = torch.cat([h, h_in], dim=-1)

        d_xyz   = self.gaussian_warp(h)
        d_scale = self.gaussian_scaling(h)
        d_rot   = self.gaussian_rotation(h)   
        signal  = self.gaussian_signal(h)

        return d_xyz, d_scale, d_rot, signal

