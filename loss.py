import torch
import torch.nn.functional as F
from torch import nn

from Baseline.utils import gen_triplets


class TripletLoss(nn.Module):

    def __init__(self, margin=0.25):
        super().__init__()
        self.margin = margin

    def forward(self, sim_mat, triplets):
        anc_idxes, pos_idxes, neg_idxes = triplets
        S_ap = sim_mat[anc_idxes, pos_idxes]
        S_an = sim_mat[anc_idxes, neg_idxes]
        losses = F.relu(S_an - S_ap + self.margin)
        mask = losses > 0  # all
        N = mask.sum()
        if N == 0:
            loss = torch.tensor(0.0, requires_grad=True, device=sim_mat.device)
        else:
            loss = losses[mask].mean()
        return loss


class KLdivergence(nn.Module):
    def __init__(self):
        super().__init__()

    def kl_divergence(self, mu, logsigma):
        return -0.5 * (1 + logsigma - mu.pow(2) - logsigma.exp()).sum()

    def forward(self, img_samples, img_logsigma, txt_samples, txt_logsigma):
        vib_loss = self.kl_divergence(img_samples.mean(dim=1), img_logsigma) + self.kl_divergence(
            txt_samples.mean(dim=1), txt_logsigma
        )
        return vib_loss


class UATVRLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.tri_loss_fct = TripletLoss()
        self.vib_loss_fct = KLdivergence()

    def forward(
        self, img_embs, txt_embs, img_output, txt_output, img_samples, txt_samples, img_logsigma, txt_logsigma, labels
    ):
        triplets = gen_triplets(labels, labels)

        sim_mat = img_embs @ txt_embs.T
        tri_loss1 = self.tri_loss_fct(sim_mat, triplets)
        tri_loss2 = self.tri_loss_fct(sim_mat.T, triplets)
        tri_loss = (tri_loss1 + tri_loss2) / 2

        # from code: weighted_token_wise_intersection wo mask
        sim_mat = torch.einsum("aid,btd->abit", img_output, txt_output)
        sim_mat = torch.amax(sim_mat, dim=(2, 3))
        dsa_loss1 = self.tri_loss_fct(sim_mat, triplets)
        dsa_loss2 = self.tri_loss_fct(sim_mat.T, triplets)
        dsa_loss = (dsa_loss1 + dsa_loss2) / 2

        # token-wise DRL (extra v & t cls token)
        prob_sim_mat = torch.einsum("ad,bd->ab", img_samples.flatten(0, -2), txt_samples.flatten(0, -2))

        batch_size = labels.shape[0]
        offset1 = sim_mat.shape[0] // batch_size
        offset2 = sim_mat.shape[1] // batch_size
        la = torch.ones((offset1, offset2), device=labels.device, dtype=torch.bool)

        pos_mask = (labels @ labels.T) > 0
        # pos_mask.fill_diagonal_(False)
        pos_mask = torch.kron(pos_mask, la)  # 克罗内克积
        neg_mask = ~pos_mask
        prob_triplets = torch.where(pos_mask.unsqueeze(2) * neg_mask.unsqueeze(1))

        dua_loss1 = self.tri_loss_fct(prob_sim_mat, prob_triplets)
        dua_loss2 = self.tri_loss_fct(prob_sim_mat.T, prob_triplets)
        dua_loss = (dua_loss1 + dua_loss2) / 2

        # kl divergence
        vib_loss = self.vib_loss_fct(img_samples, img_logsigma, txt_samples, txt_logsigma)

        return tri_loss, dsa_loss, dua_loss, vib_loss


if __name__ == "__main__":
    pass
