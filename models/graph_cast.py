import torch
import torch.nn as nn
from models.forecast_module import ForecastModule
import torch.nn.functional as F
from models.relation_hybrid import GroupedHybridRelationHead


class ContextualPresenceClassifier(nn.Module):
    def __init__(self, input_dim, nhead=4, p_drop=0.2):
        super().__init__()
        self.norm1 = nn.LayerNorm(input_dim)
        self.attn1 = nn.MultiheadAttention(input_dim, num_heads=nhead, batch_first=True)
        self.norm2 = nn.LayerNorm(input_dim)
        self.attn2 = nn.MultiheadAttention(input_dim, num_heads=nhead, batch_first=True)
        self.drop  = nn.Dropout(p_drop)
        self.mlp   = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Dropout(p_drop),
            nn.Linear(input_dim, 1)
        )

    def forward(self, x, key_padding_mask=None):
        x = self.norm1(x)
        h, _ = self.attn1(x, x, x, key_padding_mask=key_padding_mask)
        x = self.norm2(x + h)
        h, _ = self.attn2(x, x, x, key_padding_mask=key_padding_mask)
        x = x + self.drop(h)
        return self.mlp(x).squeeze(-1)


        

class GraphCast(nn.Module):
    def __init__(self, triplet_embed_dim, num_relation_classes):
        super().__init__()
        self.num_relation_classes = num_relation_classes
        self.forecast_module = ForecastModule(triplet_embed_dim)

        D = triplet_embed_dim
        D_half = D // 2

        self.presence_head = ContextualPresenceClassifier(D_half)
        
        self.relation_head = GroupedHybridRelationHead(
            input_dim=D, n_att=3, n_spa=6, n_con=16, hidden=256, geom_dim=8, geom_hidden=64, p_drop=0.2
        )

    # def forward(self, z_seq, oc_last, ob_last, z_target=None, z_reference=None, mask=None):
    def forward(self, z_seq, oc_last, ob_last, z_target=None, z_reference=None, mask=None, frame_mask=None, object_pad: torch.Tensor | None = None):

    
        # forecast_out = self.forecast_module(z_seq, oc_last, ob_last, mask=None)
        forecast_out = self.forecast_module(z_seq, oc_last, ob_last, mask=None, frame_mask=frame_mask)
        z_pred = forecast_out["z_pred"]

        z_refined = z_pred  

        B, N, D = z_refined.shape
        D_half = D // 2
        oc_j = z_refined[:, :, D_half:]  # [B, N, D//2]

        presence_logits = self.presence_head(oc_j, key_padding_mask=object_pad)  # [B, N]
        
        att_logits, spa_logits, con_logits = self.relation_head(z_refined, boxes=ob_last)



        return {
            "forecast": forecast_out,
            "refined": {
                "presence_logits": presence_logits,     # [B,N]
                "rel_att_logits": att_logits,           # [B,N,N,3]
                "rel_spa_logits": spa_logits,           # [B,N,N,6]
                "rel_con_logits": con_logits,           # [B,N,N,16]
                "z_refined": z_refined
            }
        }



def focal_bce_with_logits(logits, targets, mask=None, alpha=0.25, gamma=2.0, reduction='mean'):
    """
    Focal Binary Cross-Entropy for logits.
    logits: [B, N] - raw logits
    targets: [B, N] - 0/1 floats
    mask: optional [B, N] boolean mask to ignore padding
    alpha: balance factor (0.25 for rare positive class)
    gamma: focusing parameter (higher → more focus on hard examples)
    """
    p = torch.sigmoid(logits)
    ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')  # [B, N]
    
    pt = p * targets + (1 - p) * (1 - targets)  # [B, N]
    focal_weight = alpha * targets + (1 - alpha) * (1 - targets)
    focal_weight = focal_weight * (1 - pt).pow(gamma)
    
    loss = focal_weight * ce_loss  # [B, N]

    if mask is not None:
        loss = loss[mask]
    
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss


def compute_sgf_loss(output,
                     presence_labels,
                     rel_att_idx,              # [B,N,N] 
                     rel_spa_mhot,             # [B,N,N,NUM_SPA] 
                     rel_con_mhot,             # [B,N,N,NUM_CON] 
                     presence_mask=None, pair_mask=None,
                     att_none_weight=0.1):
    pres_logits = output["refined"]["presence_logits"]      # [B,N]
    att_logits  = output["refined"]["rel_att_logits"]       # [B,N,N,NUM_ATT]
    spa_logits  = output["refined"]["rel_spa_logits"]       # [B,N,N,NUM_SPA]
    con_logits  = output["refined"]["rel_con_logits"]       # [B,N,N,NUM_CON]

    presence_loss = focal_bce_with_logits(pres_logits, presence_labels, mask=presence_mask)

    B,N = pres_logits.shape
    att_valid = (rel_att_idx >= 0)                   
    if pair_mask is not None:
        att_valid = att_valid & pair_mask.bool()

    if att_valid.any():
        att_targets = rel_att_idx[att_valid].reshape(-1)            # [M]
        att_logits_sel = att_logits[att_valid].reshape(-1, att_logits.size(-1))  # [M, NUM_ATT]
        loss_att = F.cross_entropy(att_logits_sel, att_targets, reduction='mean')
    else:
        loss_att = torch.tensor(0.0, device=att_logits.device)


    # ===== SPA: multi-label BCE
    bce_spa = F.binary_cross_entropy_with_logits(spa_logits, rel_spa_mhot, reduction='none')  # [B,N,N,NUM_SPA]
    if pair_mask is not None:
        bce_spa = bce_spa * pair_mask.unsqueeze(-1).float()
    denom_spa = (pair_mask.sum()*spa_logits.size(-1)).clamp_min(1) if pair_mask is not None \
                else spa_logits.numel()/spa_logits.size(-1)
    loss_spa = bce_spa.sum() / denom_spa

    # ===== CON: multi-label BCE 
    bce_con = F.binary_cross_entropy_with_logits(con_logits, rel_con_mhot, reduction='none')  # [B,N,N,NUM_CON]
    if pair_mask is not None:
        bce_con = bce_con * pair_mask.unsqueeze(-1).float()
    denom_con = (pair_mask.sum()*con_logits.size(-1)).clamp_min(1) if pair_mask is not None \
                else con_logits.numel()/con_logits.size(-1)
    loss_con = bce_con.sum() / denom_con

    total = presence_loss + loss_att + loss_spa + loss_con
    return total, {
        "presence_loss": float(presence_loss),
        "loss_att": float(loss_att),
        "loss_spa": float(loss_spa),
        "loss_con": float(loss_con),
        "loss_rel": float(loss_att + loss_spa + loss_con)
    }



def multilabel_relation_loss(logits, targets, pair_mask=None, pos_weight=None, reduction='mean'):
    """
    logits:  [B,N,N,R]
    targets: [B,N,N,R]  (multi-hot 0/1)
    pair_mask: [B,N,N]  (True/1 там, где пару учитываем), опционально
    pos_weight: Tensor[R] или скаляр для балансировки классов, опционально
    """
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none', pos_weight=pos_weight)  # [B,N,N,R]
    if pair_mask is not None:
        loss = loss * pair_mask.unsqueeze(-1)  

    if reduction == 'mean':
        if pair_mask is not None:
            denom = (pair_mask.unsqueeze(-1).expand_as(loss)).sum().clamp_min(1.0)
        else:
            denom = torch.numel(loss) / loss.size(-1)  
        return loss.sum() / denom
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss

