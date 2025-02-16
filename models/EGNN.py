import os
import sys
import math
from typing import List, Dict, Tuple, Set, Union, Optional, Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


class EGCL(nn.Module):

    def __init__(self,
            in_dim: int,
            out_dim: int,
            h_dim: int = 128,
            use_attention: bool = False,
            use_tanh: bool = False,
            implementation: str = 'paper') -> None:

        super(EGCL, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.h_dim = h_dim
        self.use_attention = use_attention
        self.use_tanh = use_tanh
        self.implementation = implementation

        # original EGNN code does not follow the paper; therefore, <implementation> option has added
        # if implementaition == 'paper', it follows EGNN paper
        # if implementation == 'repo', it folows EGNN original code
        assert implementation in ['paper', 'repo'], \
            '[error] <implementation> must be "paper" or "repo"!'

        self.message_mlp = nn.Sequential(
            nn.Linear(in_dim * 2 + 1, h_dim),
            nn.SiLU(),
            nn.Linear(h_dim, h_dim),
            nn.SiLU())

        self.feat_mlp = nn.Sequential(
            nn.Linear(in_dim + h_dim, h_dim),
            nn.SiLU(),
            nn.Linear(h_dim, out_dim))

        self.coord_mlp = list()
        self.coord_mlp.append(nn.Linear(h_dim, h_dim))
        self.coord_mlp.append(nn.SiLU())
        self.coord_mlp.append(nn.Linear(h_dim, 1, bias = False))
        torch.nn.init.xavier_uniform_(self.coord_mlp[-1].weight, gain = 0.001)

        if use_tanh:

            self.coord_mlp.append(nn.Tanh())

        self.coord_mlp = nn.Sequential(*self.coord_mlp)

        if use_attention:
        
            self.attention_mlp = nn.Sequential(
                nn.Linear(h_dim, 1),
                nn.Sigmoid())

    def compute_message(self,
            feat: torch.Tensor,
            radial: torch.Tensor) -> torch.Tensor:

        # equation 3 in the EGNN paper

        num_batch, num_atom = feat.size(0), feat.size(1)

        # message: (batch, num_atom, num_atom, dim)
        # use empty tensor to avoid torch.cat method
        message = torch.empty(num_batch, num_atom, num_atom, self.in_dim * 2 + 1, device = feat.device)
        message[:, :, :, :self.in_dim] = feat.unsqueeze(2).repeat(1, 1, num_atom, 1)
        message[:, :, :, self.in_dim:self.in_dim * 2] = feat.unsqueeze(1).repeat(1, num_atom, 1, 1)
        message[:, :, :, self.in_dim * 2:] = radial

        message = self.message_mlp(message)

        if self.use_attention:

            attn = self.attention_mlp(message)
            message = message * attn

        return message

    def compute_coord(self,
            message: torch.Tensor,
            coord: torch.Tensor,
            radial: torch.Tensor,
            disp: torch.Tensor,
            mask2d: torch.Tensor,
            ) -> torch.Tensor:

        # equation 4 in the EGNN paper
        # C = 1 / (M - 1) where M is # of atoms in the molecule
        # However, the original implmentation code uses C = 1
        # if disp is None:
        #     disp = coord.unsqueeze(2) - coord.unsqueeze(1) # (batch, num_atom, num_atom, 3)

        match self.implementation:

            case 'paper':

                update = self.coord_mlp(message) # (batch, num_atom, num_atom, 1)

                update = torch.clamp(update * disp, min = -100, max = 100) * mask2d # (batch, num_atom, num_atom, 3)

                # mask2d shape 유지하면서 self-connection 제거
                mask2d = mask2d * (~torch.eye(mask2d.size(1), dtype=torch.bool, device=mask2d.device).unsqueeze(0).unsqueeze(-1))
                # C 계산 시 shape 유지
                C = torch.sum(mask2d[:, 0, :, 0], dim=-1, keepdim=True).unsqueeze(-1)  # (batch, 1, 1)

            case 'repo':

                update = self.coord_mlp(message) # (batch, num_atom, num_atom, 1)
                update = torch.clamp(update * disp, min = -100, max = 100) # (batch, num_atom, num_atom, 3)

                C = 1
        # print("update", update.size())
        # print("mask2d", mask2d.size())
        update = C * torch.sum(update * mask2d, dim = 2) # (batch, num_atom, 3)

        return update

    def compute_feat(self,
            feat: torch.Tensor,
            message: torch.Tensor,
            adj_mat: torch.Tensor,
            mask2d: torch.Tensor) -> torch.Tensor:

        # equation 5-6 in the EGNN paper

        num_batch, num_atom = feat.size(0), feat.size(1)

        match self.implementation:

            case 'paper':
                # print("message", message.size())
                # print("mask2d", mask2d.size())
                # print("adj_mat", adj_mat.size())

                message = message * mask2d * adj_mat

            case 'repo':

                message = message * mask2d.unsqueeze(-1)

        message = torch.sum(message, dim = 2) # (batch, num_atom, dim)

        new_feat = torch.empty(num_batch, num_atom, self.in_dim + self.h_dim, device = feat.device)
        new_feat[:, :, :self.in_dim] = feat
        new_feat[:, :, self.in_dim:] = message

        new_feat = self.feat_mlp(new_feat)

        return new_feat

    def forward(self,
            feat: torch.Tensor,
            coord: torch.Tensor,
            radial: torch.Tensor,
            disp: torch.Tensor,
            adj_mat: torch.Tensor,
            mask: torch.Tensor,
            mask2d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        # feat: (batch, num_atom, dim)
        # coord: (batch, num_atom, 3)
        # radial: (batch, num_atom, 3)
        # adj_mat: (batch, num_atom, num_atom)
        # mask: (batch, num_atom)
        # mask2d: (batch, num_atom, num_atom)

        message = self.compute_message(feat, radial)
        new_coord = self.compute_coord(message, coord, radial, disp, mask2d)
        new_feat = self.compute_feat(feat, message, adj_mat, mask2d)

        new_feat = new_feat * mask.unsqueeze(-1)
        new_coord = new_coord * mask.unsqueeze(-1)

        return new_feat, new_coord


class EAL(nn.Module):

    def __init__(self,
        h_dim: int,
        num_head: int,
        layer_norm: str = 'pre') -> None:

        super(EAL, self).__init__()
        
        assert h_dim % num_head == 0, \
            '[error] <h_dim> must be divisible by <num_head>'
        assert layer_norm in ['pre', 'post', 'none'], \
            '[error] <layer_norm> must be "pre", "post", or "none"!'

        self.h_dim = h_dim
        self.num_head = num_head
        self.head_dim = h_dim // num_head
        self.layer_norm_type = layer_norm

        self.scale = self.head_dim ** (-0.5)
        self.to_qkv = nn.Linear(h_dim, h_dim * 3, bias = False)
        self.to_out = nn.Linear(h_dim, h_dim)
        self.layer_norm = nn.LayerNorm(h_dim)

    def forward(self, 
            feat: torch.Tensor,
            coord: torch.Tensor,
            radial: torch.Tensor,
            disp: torch.Tensor,
            adj_mat: torch.Tensor, 
            mask: torch.Tensor,
            mask2d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        # feat: (batch, num_atom, dim)
        # coord: (batch, num_atom, 3)
        # radial: (batch, num_atom, 3)
        # adj_mat: (batch, num_atom, num_atom)
        # mask: (batch, num_atom)
        # mask2d: (batch, num_atom, num_atom)

        num_batch, num_atom = feat.size(0), feat.size(1)

        if self.layer_norm_type == 'pre':

            out = self.layer_norm(feat)

        q, k, v = self.to_qkv(out).chunk(3, dim = -1)
        q = q.reshape(num_batch, self.num_head, num_atom, self.head_dim)
        k = k.reshape(num_batch, self.num_head, num_atom, self.head_dim)
        v = v.reshape(num_batch, self.num_head, num_atom, self.head_dim)

        #print(mask2d.size())

        mask2d = mask2d.permute(0,3,1,2).expand(num_batch, self.num_head, num_atom, num_atom)

        attn = torch.einsum('bhnd,bhmd->bhnm', q, k) * self.scale # (batch, head, num_atom, num_atom)
        attn = attn.masked_fill(~mask2d, torch.finfo(attn.dtype).min)
        attn = attn.softmax(dim = -1)

        out = torch.einsum('bhnm,bhmd->bhnd', attn, v)
        out = out.view(num_batch, num_atom, -1)
        out = F.silu(self.to_out(out))
        out = feat + out # residual connection
        
        if self.layer_norm_type == 'post':

            out = self.layer_norm(out)

        out = out * mask.unsqueeze(-1)

        return out, 0.



class TimestepEncoding(nn.Module):

    def __init__(self,
            enc_dim: int = 128,
            num_timestep: int = 1000) -> None:

        super(TimestepEncoding, self).__init__()

        self.enc_dim = enc_dim
        self.num_timestep = num_timestep

        assert enc_dim % 2 == 0, \
            '[error] <enc_dim> must be even number! received: {}'.format(enc_dim)

        self.freq_bands = nn.Parameter(torch.linspace(0, 1, enc_dim // 2).unsqueeze(0), requires_grad = False)

    def forward(self,
            x: torch.Tensor,
            t: torch.Tensor) -> torch.Tensor:

        assert x.size(-1) == self.enc_dim, \
            '[error] wrong <enc_dim>! x: {}, enc_dim: {}'.format(x.size(-1), self.enc_dim)

        out = torch.zeros_like(x)

        sin = torch.sin(((t / self.num_timestep) + self.freq_bands) * 0.5 * math.pi).unsqueeze(1)
        cos = torch.cos(((t / self.num_timestep) + self.freq_bands) * 0.5 * math.pi).unsqueeze(1)

        out[..., :sin.size(-1)] = sin
        out[..., sin.size(-1):] = cos

        return out



class EGNN(nn.Module):

    def __init__(self,
            in_dim: int,
            out_dim: int,
            h_dim: int = 128,
            num_layer: int = 6,
            num_timesteps: int = 300,
            temperature: float = 300.0,
            update_coord: str = 'last',
            use_tanh: bool = False,
            use_pbc: bool = False,
            use_rinv: bool = True,
            use_attention: bool = False,
            num_head: Optional[int] = 4,
            use_condition: bool = False,
            layer_norm: str = 'pre') -> None:

        super(EGNN, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.h_dim = h_dim
        self.num_layer = num_layer
        self.update_coord = update_coord
        self.use_tanh = use_tanh
        self.use_pbc = use_pbc
        self.use_rinv = use_rinv
        self.use_attention = use_attention
        self.use_condition = use_condition

        self.layer = list()
        self.emb = nn.Linear(in_dim, h_dim) # Not using this for now
        self.atomic_emb = nn.Linear(100, h_dim) # Not using this for now
        # encoding layers
        self.time_enc = TimestepEncoding(h_dim, num_timesteps)
        self.enc_dim = 15 if use_condition else 6
        self.feat_enc = nn.Linear(self.enc_dim, h_dim) # encoding [force, velocity, condition(optional)]
        self.combine = nn.Linear(2*h_dim, h_dim)

        for idx in range(num_layer):

            self.layer.append(EGCL(
                h_dim,
                h_dim,
                h_dim,
                use_attention = use_attention,
                use_tanh = use_tanh))

            if self.use_attention:

                self.layer.append(EAL(
                    h_dim = h_dim,
                    num_head = num_head,
                    layer_norm = layer_norm))

        self.layer = nn.ModuleList(self.layer)
        self.dec = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.SiLU(),
            nn.Linear(h_dim, out_dim))
        # self.equation = EGCL(
        #     h_dim,
        #     4,
        #     h_dim,
        #     use_attention = use_attention,
        #     use_tanh = use_tanh)

    def forward(self,
        atom_feat: torch.Tensor,
        coord: torch.Tensor,
        radial: torch.Tensor,
        disp: torch.Tensor,
        t: torch.Tensor,
        adj_mat: torch.Tensor,
        mask: torch.Tensor,
        mask2d: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        lattice: Optional[torch.Tensor] = None,) -> Tuple[torch.Tensor, torch.Tensor]:

        # atom_feat: (batch, num_atom, 6) [force, velocity], originally h_dim
        # coord: (batch, num_atom, 3)
        # mask: (batch, num_atom)
        # mask2d: (batch, num_atom, num_atom)
        # adj_mat: (batch, num_atom, num_atom)
        # lattice: (batch, 3, 3)

        new_coord = coord
        if radial is None:
            disp = coord.unsqueeze(2) - coord.unsqueeze(1) # (batch, num_atom, num_atom, 3)
            radial = torch.sum(disp ** 2, dim = -1, keepdim = True) # (batch, num_atom, num_atom, 1)
            mask2d = radial < 0.3
            adj_mat = mask2d
        disp = disp.squeeze()           # (batch, num_atom, num_atom, 3)
         # torch.sum(disp ** 2, dim = -1, keepdim = True) # (batch, num_atom, num_atom, 1)
        # print("atom_feat", atom_feat.size())
        # print("coord", coord.size())
        # print("radial", radial.size())
        # print("disp", disp.size())
        # print("adj_mat", adj_mat.size())
        # print("mask", mask.size())
        # print("mask2d", mask2d.size())

        if self.use_rinv:

            radial = 1 / (radial + 0.3)

        if self.use_condition:
                #atom_feat = atom_feat[..., -3:] # only velocities
                #condition = torch.cat([condition[..., :3], condition[..., -3:]], dim = -1) # coord, vel
                atom_feat = torch.cat([atom_feat, condition], dim = -1) # (batch, num_atom, 12)
        atom_feat = self.feat_enc(atom_feat)
        t_feat = self.time_enc(atom_feat, t)

        feat = torch.empty(atom_feat.size(0), atom_feat.size(1), self.h_dim * 2).to(atom_feat)
        # combine atom_feat and t_feat
        feat[:, :, :self.h_dim] = atom_feat
        feat[:, :, self.h_dim:] = t_feat
        feat = self.combine(feat)


        for idx in range(self.num_layer):

            feat, coord_update = self.layer[idx](feat, new_coord, radial, disp, adj_mat, mask, mask2d)
            #print(coord_update)
            #new_coord = new_coord.clone() + coord_update

        new_feat = self.dec(feat) * mask.unsqueeze(-1)
        #if self.out_dim == 6: # if only predict coord and vel,
        new_coord = new_coord * mask.unsqueeze(-1)
        # new_feat[..., :3] = new_coord

        return new_feat, new_coord


if __name__ == '__main__':

    import random

    in_dim = 64
    out_dim = 64
    h_dim = 128
    layer = EGCL(in_dim, out_dim, h_dim)
    attention = EAL(h_dim, 4, layer_norm = 'pre')
    model = EGNN(in_dim, out_dim, h_dim, use_attention = True)

    num_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('[info] # of parameters: {} M'.format(num_param / 10 ** 6))

    feat = torch.randn(3, 7, in_dim)
    latent = torch.randn(3, 7, h_dim)
    coord = torch.randn(3, 7, 3)
    radial = torch.randn(3, 7, 7, 1)

    mask = torch.Tensor([[False for _ in range(7)] for _ in range(3)]).to(torch.bool)
    
    for idx in range(3):

        for jdx in range(random.randint(2, 6)):

            mask[idx][jdx] = True

    mask2d = torch.zeros(mask.size(0), mask.size(1), mask.size(1), dtype = torch.bool)
    mask2d[~mask] = 1
    mask2d = mask2d.transpose(1, 2)
    mask2d[~mask] = 1
    mask2d = ~mask2d

    adj_mat = (torch.randn(3, 7, 7) > 0) * mask2d

    new_feat, new_coord = layer(feat, coord, radial, adj_mat, mask, mask2d)
    print(new_feat.size(), new_coord.size())

    new_feat, new_coord = attention(latent, coord, radial, adj_mat, mask, mask2d)
    print(new_feat.size(), new_coord.size())

    new_feat, new_coord = model(feat, coord, adj_mat, mask, mask2d)
    print(new_feat.size(), new_coord.size())
