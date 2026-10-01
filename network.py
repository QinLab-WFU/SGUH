from argparse import Namespace

import torch
import torch.nn.functional as F
from torch import nn

from clip.model import Transformer
from Baseline_CM.network import HashClip
from PCME.uncertainty_module import UncertaintyModuleImage
from PCME.utils import sample_gaussian_tensors
from PVSE.network import PIENet


class HashClipMod(HashClip):

    def __init__(self, output_dim):
        super().__init__(output_dim)

        embed_dim = self.embed_dim

        self.img_position_embeddings = nn.Embedding(128, embed_dim)
        self.txt_position_embeddings = nn.Embedding(128, embed_dim)

        self.seqTransf = Transformer(width=embed_dim, layers=4, heads=embed_dim // 64)

        self.pie_net_img = PIENet(1, embed_dim, embed_dim, embed_dim // 2, 0.1)
        self.uncertain_net_img = UncertaintyModuleImage(embed_dim, embed_dim, embed_dim // 2)

        self.pie_net_txt = PIENet(1, embed_dim, embed_dim, embed_dim // 2, 0.1)
        # Note: not UncertaintyModuleText, because input is not text sequence
        self.uncertain_net_txt = UncertaintyModuleImage(embed_dim, embed_dim, embed_dim // 2)

    def probabilistic_xxx(self, pooled_feats, feats, _type):
        out, _, _ = getattr(self, f"pie_net_{_type}")(pooled_feats, feats)
        out = F.normalize(out)

        uncertain_out = getattr(self, f"uncertain_net_{_type}")(
            pooled_feats, feats
        )  # (B 512) (B 12 512)   multiheadatt + fc + (residual)
        logsigma = uncertain_out["logsigma"]

        samples = sample_gaussian_tensors(out, logsigma, 7)  # B 7 512    从高斯分布中采样N个embedding

        return samples, logsigma

    def encode_img(self, image, return_hidden=False):
        x = self.clip.visual.conv1(image.type(self.clip.dtype))  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [
                self.clip.visual.class_embedding.to(x.dtype)
                + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
                x,
            ],
            dim=1,
        )  # shape = [*, grid ** 2 + 1, width]
        x = x + self.clip.visual.positional_embedding.to(x.dtype)
        x = self.clip.visual.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.clip.visual.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        # x = self.clip.visual.ln_post(x[:, 0, :])

        # if self.clip.visual.proj is not None:
        #     x = x @ self.clip.visual.proj

        # return x

        hidden = self.clip.visual.ln_post(x) @ self.clip.visual.proj

        x0 = hidden[:, 0, :]

        x = self.img_hash(x0)  # TODO: j4t!

        if return_hidden:
            return x, x0, hidden

        return x

    def encode_txt(self, text, return_hidden=False):
        x = self.clip.token_embedding(text).type(self.clip.dtype)  # [batch_size, n_ctx, d_model]

        pos_emd = self.clip.positional_embedding[: x.size(1), :].type(self.clip.dtype)  # [batch_size, d_mdoel]
        x = x + pos_emd
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.clip.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        hidden = self.clip.ln_final(x).type(self.clip.dtype) @ self.clip.text_projection

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x0 = hidden[torch.arange(hidden.shape[0]), text.argmax(dim=-1)]
        x = self.txt_hash(x0)  # TODO: j4t!

        if return_hidden:
            return x, x0, hidden

        return x

    def forward(self, imgs, txts, return_hidden=False, add_extra_token=False):
        if return_hidden:
            # for training
            img_embs, img_feats, img_output = self.encode_img(imgs, True)
            txt_embs, txt_feats, txt_output = self.encode_txt(txts, True)

            img_embs, txt_embs = F.normalize(img_embs), F.normalize(txt_embs)

            if add_extra_token:
                img_output_original = img_output
                img_seq_length = img_output.size(1) + 2  # extra 2 learnable token
                img_position_ids = torch.arange(img_seq_length, dtype=torch.long, device=imgs.device)
                img_position_ids = img_position_ids.unsqueeze(0).expand(img_output.size(0), -1)
                img_position_embeddings = self.img_position_embeddings(img_position_ids)  # bs num+extra_token_num dim
                img_position_embeddings[:, 0 : img_output.size(1), :] += img_output
                img_output = img_position_embeddings

                img_output = img_output.permute(1, 0, 2)  # NLD -> LND
                img_output = self.seqTransf(img_output)
                img_output = img_output.permute(1, 0, 2).contiguous()  # LND -> NLD
                img_output[:, : img_output_original.size(1), :] += img_output_original

                img_output = F.normalize(img_output, dim=-1)[:, : img_output_original.size(1), :].contiguous()
                img_pooled = img_output.mean(1)
                img_pooled = F.normalize(img_pooled, dim=-1)

                img_samples, img_logsigma = self.probabilistic_xxx(img_pooled, img_output, "img")

                txt_output_original = txt_output  # save original
                txt_seq_length = txt_output.size(1) + 2  # extra 2 learnable token
                txt_position_ids = torch.arange(txt_seq_length, dtype=torch.long, device=txts.device)
                txt_position_ids = txt_position_ids.unsqueeze(0).expand(txt_output.size(0), -1)
                txt_position_embeddings = self.txt_position_embeddings(txt_position_ids)
                txt_position_embeddings[:, 0 : txt_output.size(1), :] += txt_output
                txt_output = txt_position_embeddings

                txt_output = txt_output.permute(1, 0, 2)
                txt_output = self.seqTransf(txt_output)
                txt_output = txt_output.permute(1, 0, 2).contiguous()
                txt_output[:, : txt_output_original.size(1), :] += txt_output_original

                txt_output = F.normalize(txt_output, dim=-1)[:, : txt_output_original.size(1), :].contiguous()
                txt_pooled = txt_output.mean(1)
                txt_pooled = F.normalize(txt_pooled, dim=-1)

                txt_samples, txt_logsigma = self.probabilistic_xxx(txt_pooled, txt_output, "txt")
            else:
                img_feats, img_output = F.normalize(img_feats), F.normalize(img_output, dim=-1)
                txt_feats, txt_output = F.normalize(txt_feats), F.normalize(txt_output, dim=-1)
                img_samples, img_logsigma = self.probabilistic_xxx(img_feats, img_output, "img")
                txt_samples, txt_logsigma = self.probabilistic_xxx(txt_feats, txt_output, "txt")

            return img_embs, txt_embs, img_output, txt_output, img_samples, txt_samples, img_logsigma, txt_logsigma

        img_embs = self.encode_img(imgs, False)
        txt_embs = self.encode_txt(txts, False)
        # img_embs = F.normalize(img_embs)
        # txt_embs = F.normalize(txt_embs)

        return img_embs, txt_embs


def build_model(args: Namespace):
    if args.backbone != "clip":
        raise NotImplementedError(f"Not support: {args.backbone}")
    net = HashClipMod(args.n_bits)
    return net.to(args.device)


if __name__ == "__main__":
    imgs = torch.randn(2, 3, 224, 224)
    txts = torch.randint(1, 10, (2, 32))
    net = HashClipMod(8)
    # z1 = net.encode_img(imgs, True)
    # print(z1[0].shape, z1[1].shape)
    # z2 = net.encode_txt(txts, True)
    # print(z2[0].shape, z2[1].shape)
    out = net(imgs, txts, True)
    for x in out:
        print(x.shape)
