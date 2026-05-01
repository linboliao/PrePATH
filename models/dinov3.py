import os
import torch
import torch.nn as nn
from torchvision import transforms


# ==============================================================
# =============== ViT 基础模块定义（兼容旧版Torch）===============
# ==============================================================

class PatchEmbed(nn.Module):
    """将输入图像划分为patch，并进行线性投影"""

    def __init__(self, img_size=256, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: [B, 3, H, W]
        x = self.proj(x)  # [B, embed_dim, H/patch, W/patch]
        x = x.flatten(2).transpose(1, 2)  # [B, N, embed_dim]
        return x


class SimpleViT(nn.Module):
    """简化版 ViT-B/16 (支持256输入)"""

    def __init__(self, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=True, img_size=256):
        super().__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer

        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=16, in_chans=3, embed_dim=embed_dim)
        self.num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.num_patches, embed_dim))
        self.norm = nn.LayerNorm(embed_dim)

        encoder_layer = TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            activation='gelu',
            batch_first=True
        )
        self.encoder = TransformerEncoder(encoder_layer, num_layers=depth)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        x = self.patch_embed(x)  # [B, N, C]
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.encoder(x)
        x = self.norm(x)
        return x[:, 0]  # 仅返回 CLS token 表征


# ==============================================================
# ===============   加载本地 DINOv3 权重模型函数   ===============
# ==============================================================

def get_dinov3_model(device, weights_path='/NAS2/Data1/hukang/Code/dinov3_train/outputs/industrial_ssl/checkpoint_epoch_11.pth'):
    print(f"✅ 使用本地 DINOv3 权重加载 (兼容 torchvision==0.11.2, 输入256x256)")
    model = SimpleViT(embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, img_size=256)
    model.to(device)

    if weights_path is not None and os.path.exists(weights_path):
        print(f"🔍 正在加载权重: {weights_path}")
        state = torch.load(weights_path, map_location="cpu")
        if "state_dict" in state:
            state = state["state_dict"]
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"✅ 已加载 DINOv3 权重 ({len(state)} 个参数)")
        if len(missing) > 0:
            print(f"⚠️ 缺少 {len(missing)} 个参数未匹配（通常为head或teacher模块）")
        if len(unexpected) > 0:
            print(f"⚠️ 发现 {len(unexpected)} 个额外参数（通常是投影层）")
    else:
        print("⚠️ 未找到权重文件，模型使用随机初始化")

    model.eval()
    return model


def get_dinov3_trans():
    """DINOv3 图像预处理 (输入256x256)"""
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
    return transform


def dinov3_baseline(device="cuda", weights_path="/NAS/xk/gastritis/CLAM/models/DINOv3/best_model.pth"):
    """加载 DINOv3 模型及对应 transform"""
    model = get_dinov3_model(device, weights_path)
    eval_transforms = get_dinov3_trans()
    return model, eval_transforms


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tfm = dinov3_baseline(device)
    dummy = torch.randn(1, 3, 256, 256).to(device)
    with torch.no_grad():
        out = model(dummy)
    print("✅ 测试输出特征维度:", out.shape)
