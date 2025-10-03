
# models/relation_hybrid.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class LowRankBiaffine(nn.Module):
    """
    Считает для каждой пары (i,j) и класса r:
      score_r(i,j) = sum_k <hi, U[r,:,k]> * <hj, V[r,:,k]> + b_r
    где hi,hj \in R^H, ранг k << H.
    """
    def __init__(self, hidden_dim: int, num_classes: int, rank: int = 32, dropout: float = 0.0, bias: bool = True):
        super().__init__()
        self.H = hidden_dim
        self.C = num_classes
        self.R = rank

        # [C, H, R]
        self.U = nn.Parameter(torch.randn(num_classes, hidden_dim, rank) * 0.01)
        self.V = nn.Parameter(torch.randn(num_classes, hidden_dim, rank) * 0.01)
        self.drop = nn.Dropout(dropout)
        self.bias = nn.Parameter(torch.zeros(num_classes)) if bias else None

    def forward(self, hi_e: torch.Tensor, hj_e: torch.Tensor) -> torch.Tensor:
        """
        hi_e, hj_e: [B, N, N, H]
        return:     [B, N, N, C]
        """
        hi_e = self.drop(hi_e)
        hj_e = self.drop(hj_e)

        # [B,N,N,H] x [C,H,R] -> [B,N,N,C,R]
        s1 = torch.einsum('bnih,chr->bnicr', hi_e, self.U)
        s2 = torch.einsum('bnjh,chr->bnjcr', hj_e, self.V)
        out = (s1 * s2).sum(dim=-1)  # [B,N,N,C]
        if self.bias is not None:
            out = out + self.bias
        return out


class GroupedHybridRelationHead(nn.Module):
    """
    Общие роль-проекции/контент/геометрия → три головы:
      - attention:   [B,N,N,3]  (softmax)
      - spatial:     [B,N,N,6]  (softmax)
      - contacting:  [B,N,N,16] (sigmoid)
    Теперь + резидуальная биаффинность для каждой семьи.
    """
    def __init__(self, input_dim, n_att=3, n_spa=6, n_con=16, hidden=256,
                 geom_dim=8, geom_hidden=64, p_drop=0.2,
                 # биаффинные настройки:
                 biaff_rank_att=32, biaff_rank_spa=32, biaff_rank_con=32,
                 biaff_dropout=0.1,
                 # смешивание линейной и биаффинной частей
                 init_alpha_att=0.5, init_alpha_spa=0.5, init_alpha_con=0.5,
                 # температурные масштабы (логиты / T)
                 temp_att=1.0, temp_spa=1.0, temp_con=1.0):
        super().__init__()
        self.n_att, self.n_spa, self.n_con = n_att, n_spa, n_con
        self.temp_att, self.temp_spa, self.temp_con = temp_att, temp_spa, temp_con

        Dh = input_dim // 2
        H = hidden

        self.proj_s = nn.Linear(Dh, H, bias=False)
        self.proj_o = nn.Linear(Dh, H, bias=False)

        # контентный MLP 
        self.mlp_content = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, H),
            nn.GELU(),
            nn.Dropout(p_drop),
        )

        # геометрический MLP
        self.use_geom = bool(geom_dim and geom_dim > 0)
        if self.use_geom:
            self.mlp_geom = nn.Sequential(
                nn.LayerNorm(geom_dim),
                nn.Linear(geom_dim, geom_hidden),
                nn.GELU(),
                nn.Dropout(p_drop),
                nn.Linear(geom_hidden, H)
            )

        # Линейные головы 
        self.out_att = nn.Linear(H, n_att)
        self.out_spa = nn.Linear(H, n_spa)
        self.out_con = nn.Linear(H, n_con)

        self.biaff_att = LowRankBiaffine(H, n_att, rank=biaff_rank_att, dropout=biaff_dropout)
        self.biaff_spa = LowRankBiaffine(H, n_spa, rank=biaff_rank_spa, dropout=biaff_dropout)
        self.biaff_con = LowRankBiaffine(H, n_con, rank=biaff_rank_con, dropout=biaff_dropout)

        self._alpha_att = nn.Parameter(torch.tensor(init_alpha_att).float())
        self._alpha_spa = nn.Parameter(torch.tensor(init_alpha_spa).float())
        self._alpha_con = nn.Parameter(torch.tensor(init_alpha_con).float())

        self.pre_fuse_norm = nn.LayerNorm(H)

    @staticmethod
    def make_geom_features(boxes):
        x1,y1,x2,y2 = [boxes[...,k] for k in range(4)]
        w = (x2-x1).clamp_min(1e-6); h = (y2-y1).clamp_min(1e-6)
        cx = (x1+x2)*0.5; cy = (y1+y2)*0.5
        dx = cx[:,:,None]-cx[:,None,:]; dy = cy[:,:,None]-cy[:,None,:]
        log_dw = torch.log(w[:,:,None]/w[:,None,:]); log_dh = torch.log(h[:,:,None]/h[:,None,:])
        xx1 = torch.maximum(x1[:,:,None], x1[:,None,:]); yy1 = torch.maximum(y1[:,:,None], y1[:,None,:])
        xx2 = torch.minimum(x2[:,:,None], x2[:,None,:]); yy2 = torch.minimum(y2[:,:,None], y2[:,None,:])
        inter = (xx2-xx1).clamp_min(0)*(yy2-yy1).clamp_min(0)
        area = w*h; union = (area[:,:,None]+area[:,None,:]-inter).clamp_min(1e-6)
        iou = inter/union
        center_dist = torch.sqrt(dx**2+dy**2)
        overlap_x = (xx2-xx1).clamp_min(0); overlap_y = (yy2-yy1).clamp_min(0)
        feats = torch.stack([dx,dy,log_dw,log_dh,iou,center_dist,overlap_x,overlap_y], dim=-1)
        return torch.nan_to_num(feats, nan=0.0, posinf=10.0, neginf=-10.0)

    def _alpha(self, p: nn.Parameter) -> torch.Tensor:
        return torch.sigmoid(p)

    def forward(self, z_refined, boxes=None):
        B,N,D = z_refined.shape
        Dh = D//2
        zi, zj = z_refined[...,:Dh], z_refined[...,Dh:]

        # роль-проекции
        hi = self.proj_s(zi)  # [B,N,H]
        hj = self.proj_o(zj)  # [B,N,H]
        hi_e = hi.unsqueeze(2).expand(-1,-1,N,-1)  # [B,N,N,H]
        hj_e = hj.unsqueeze(1).expand(-1,N,-1,-1)  # [B,N,N,H]

        # контент
        zi_e = zi.unsqueeze(2).expand(-1,-1,N,-1)
        zj_e = zj.unsqueeze(1).expand(-1,N,-1,-1)
        cat = torch.cat([zi_e, zj_e], dim=-1)             # [B,N,N,D]
        h_cont = self.mlp_content(cat)                    # [B,N,N,H]

        # геометрия 
        h = hi_e + hj_e + h_cont
        if self.use_geom and boxes is not None:
            g = self.make_geom_features(boxes)            # [B,N,N,8]
            g2h = self.mlp_geom(g)                        # [B,N,N,H]
            h = h + g2h

        h = self.pre_fuse_norm(h)

        lin_att = self.out_att(h)     # [B,N,N,3]
        lin_spa = self.out_spa(h)     # [B,N,N,6]
        lin_con = self.out_con(h)     # [B,N,N,16]

        bia_att = self.biaff_att(hi_e, hj_e)  # [B,N,N,3]
        bia_spa = self.biaff_spa(hi_e, hj_e)  # [B,N,N,6]
        bia_con = self.biaff_con(hi_e, hj_e)  # [B,N,N,16]

        a_att = self._alpha(self._alpha_att)
        a_spa = self._alpha(self._alpha_spa)
        a_con = self._alpha(self._alpha_con)

        att_logits = (lin_att + a_att * bia_att) / max(self.temp_att, 1e-6)
        spa_logits = (lin_spa + a_spa * bia_spa) / max(self.temp_spa, 1e-6)
        con_logits = (lin_con + a_con * bia_con) / max(self.temp_con, 1e-6)  

        return att_logits, spa_logits, con_logits
