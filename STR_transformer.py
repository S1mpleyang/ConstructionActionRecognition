import torch.nn.functional as F
import torch
import torch.nn as nn
from torchvision.models.video.resnet import r3d_18, r2plus1d_18
import timm


class STR_Transformer(nn.Module):
    def __init__(self, num_classes=1000, at_type="DTM", lstm_channel=16):
        super(STR_Transformer, self).__init__()
        self.swintransformer = timm.models.swin_base_patch4_window7_224_in22k(
            pretrained=True)
        self.backbone = r2plus1d_18(pretrained=True)
        self.swin_out = 1024
        self.backbone_out = 400
        self.at_type = at_type

        self.dropout = nn.Dropout(0.5)

        # no-attention
        self.no_attention_pred = nn.Linear(self.swin_out, num_classes)

        # CAM Attention
        self.alpha = nn.Sequential(
            nn.Linear(self.swin_out, 1),
            nn.Sigmoid()
        )
        self.alpha_pred = nn.Linear(self.swin_out, num_classes)

        # RAM Attention
        self.beta = nn.Sequential(
            nn.Linear(self.swin_out * 2, 1),
            nn.Sigmoid()
        )
        self.beta_pred = nn.Linear(self.swin_out * 2, num_classes)

        # DTM Attention
        self.gama = nn.Sequential(
            nn.Linear(self.swin_out * 6, 1),
            nn.Sigmoid()
        )
        self.gama_pred = nn.Linear(self.swin_out * 6, num_classes)

        self.theta = nn.Sequential(
            nn.Linear(self.backbone_out, self.swin_out * 2),
            nn.Sigmoid(),
        )

        # lstm
        self.lstm = nn.LSTM(input_size=lstm_channel, hidden_size=lstm_channel,
                            num_layers=1)  # (input_size,hidden_size,num_layers)

    def freeze_model(self, x):
        for i, j in x.named_parameters():
            j.requires_grad = False
        x.fc = nn.Linear(in_features=512, out_features=(self.swin_out + self.moblenet_out), bias=True)

    def forward(self, x_raw):
        # x_raw = [N, c, T, h, w]
        n, c, T, h, w = x_raw.shape
        x = x_raw.permute(0, 2, 1, 3, 4).reshape(-1, c, h, w)  # x = [N*T, c, h, w]

        vs_stack = self.swintransformer.forward_features(x)  # [N*T, 1024]

        if self.at_type == "no_attention":
            y = self.no_attention_pred(self.dropout(vs_stack))  # [N*T, class]
            y = y.reshape(n, T, -1)  # [N, T, class]
            y = torch.max(y, dim=1).values  # [N, 1, class]
            y = torch.softmax(y, dim=1)  # [N, class]
            return y

        else:
            fuse_feature = vs_stack  # [N*T, 1024]
            # CAM
            alphas = self.alpha(self.dropout(fuse_feature))  # [N*T, 1]
            attention1 = fuse_feature.mul(alphas)  # [N*T, 1024]
            fv = attention1.sum(0, keepdim=True).div(alphas.sum(0))  # [N, 1024]
            if self.at_type == "CAM":
                y = self.alpha_pred(fv)
                y = torch.softmax(y, dim=1)
                return y

            # RAM
            fv = fv.expand_as(fuse_feature)  # [N*T, 1024]
            fv = torch.cat((fuse_feature, fv), dim=1)  # [N*T, 1024*2]
            betas = self.beta(self.dropout(fv))  # [N*T, 1]
            """reshape"""
            attention2 = fv.mul(betas * alphas)  # [N*T, 1024*2]
            fv2 = attention2.sum(0, keepdim=True).div(alphas.T.mm(betas))  # [N, 1024*2]

            if self.at_type == "RAM":
                rela_att = self.beta_pred(fv2)  # [N, class]
                y = torch.softmax(rela_att, dim=1)
                return y

            # DTM
            elif self.at_type == "DTM":
                x_diff = x_raw[:, :, 1:T, :, :] - x_raw[:, :, 0:T - 1, :, :]
                x_diff = F.pad(x_diff, pad=(0, 0, 0, 0, 1, 0, 0, 0), mode="constant", value=0)
                diff_stack = self.backbone(x_diff).flatten(1)  # diff_stack = [N,400]
                diff_f = self.theta(self.dropout(diff_stack))  # [N,1024*2]
                out_t, (h_t, c_t) = self.lstm(attention2.T.unsqueeze(1))
                out_t = out_t.squeeze(1).T  # [N*T, 1024*2]
                fv2 = fv2.expand_as(out_t)  # [N*T, 1024*2]
                diff_f = diff_f.expand_as(out_t)
                fv_3 = torch.cat((fv2, diff_f, out_t), dim=1)  # [N*T, 1024*6]
                gamas = self.gama(self.dropout(fv_3))  # [N*T, 1]
                attention3 = (fv_3 * gamas * betas * alphas).sum(0, keepdim=True).div(
                    (alphas * betas * gamas).sum(0))  # [N*T, 1024*4]
                y3 = self.gama_pred(attention3)  # [N, class]
                y3 = torch.softmax(y3, dim=1)
                return y3