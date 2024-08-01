import torch
from torch import nn

class MCGRU(nn.Module):
    def __init__(self, lab_dim, demo_dim, hidden_dim: int=32, feat_dim: int=8, act_layer=nn.GELU, drop=0.0, **kwargs):
        super().__init__()
        self.lab_dim = lab_dim
        self.demo_dim = demo_dim
        self.hidden_dim = hidden_dim
        self.feat_dim = feat_dim
        self.act = act_layer()
        self.demo_proj = nn.Linear(demo_dim, hidden_dim)
        self.lab_proj = nn.Linear(lab_dim, lab_dim)
        self.grus = nn.ModuleList(
            [
                nn.GRU(1, feat_dim, num_layers=1, batch_first=True)
                for _ in range(lab_dim)
            ]
        )
        self.out_proj = nn.Linear(lab_dim*feat_dim+hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(drop)
    def forward(self, x, static, **kwargs):
        # x: [bs, time_steps, lab_dim]
        # static: [bs, demo_dim]
        bs, time_steps, lab_dim = x.shape
        demo = self.demo_proj(static) # [bs, hidden_dim]
        x = self.lab_proj(x)
        out = torch.zeros(bs, time_steps, self.lab_dim, self.feat_dim).to(x.device)
        for i, gru in enumerate(self.grus):
            cur_feat = x[:, :, i].unsqueeze(-1)
            cur_feat = gru(cur_feat)[0]
            out[:, :, i] = cur_feat
        out = out.flatten(2) # b t l f -> b t (l f)
        # concat demo and out
        out = torch.cat([demo.unsqueeze(1).repeat(1, time_steps, 1), out], dim=-1)
        out = self.out_proj(out)
        return out
