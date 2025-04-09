
import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, mlp_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x2 = self.norm1(x)
        attn_output, _ = self.attn(x2, x2, x2)
        x = x + attn_output
        x2 = self.norm2(x)
        x = x + self.mlp(x2)
        return x

def CBR2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    )

class UNetTransformer(nn.Module):
    def __init__(self, img_size=256, patch_size=16, in_channels=1, base_dim=64):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.base_dim = base_dim

        # Contracting path
        self.enc1_1 = CBR2d(in_channels, base_dim)
        self.enc1_2 = CBR2d(base_dim, base_dim)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2_1 = CBR2d(base_dim, base_dim*2)
        self.enc2_2 = CBR2d(base_dim*2, base_dim*2)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3_1 = CBR2d(base_dim*2, base_dim*4)
        self.enc3_2 = CBR2d(base_dim*4, base_dim*4)
        self.pool3 = nn.MaxPool2d(2)

        self.enc4_1 = CBR2d(base_dim*4, base_dim*8)
        self.enc4_2 = CBR2d(base_dim*8, base_dim*8)
        self.pool4 = nn.MaxPool2d(2)

        self.enc5_1 = CBR2d(base_dim*8, base_dim*16)

        # Transformer block
        self.flatten = nn.Flatten(2)  # [B, C, H*W]
        self.trans_block = TransformerBlock(dim=base_dim*16, heads=8, mlp_dim=base_dim*16*2)
        self.unflatten = lambda x, H, W: x.view(x.size(0), -1, H, W)

        # Expansive path
        self.dec5_1 = CBR2d(base_dim*16, base_dim*8)
        self.unpool4 = nn.ConvTranspose2d(base_dim*8, base_dim*8, 2, 2)
        self.dec4_2 = CBR2d(base_dim*16, base_dim*8)
        self.dec4_1 = CBR2d(base_dim*8, base_dim*4)

        self.unpool3 = nn.ConvTranspose2d(base_dim*4, base_dim*4, 2, 2)
        self.dec3_2 = CBR2d(base_dim*8, base_dim*4)
        self.dec3_1 = CBR2d(base_dim*4, base_dim*2)

        self.unpool2 = nn.ConvTranspose2d(base_dim*2, base_dim*2, 2, 2)
        self.dec2_2 = CBR2d(base_dim*4, base_dim*2)
        self.dec2_1 = CBR2d(base_dim*2, base_dim)

        self.unpool1 = nn.ConvTranspose2d(base_dim, base_dim, 2, 2)
        self.dec1_2 = CBR2d(base_dim*2, base_dim)
        self.dec1_1 = CBR2d(base_dim, base_dim)

        self.fc = nn.Conv2d(base_dim, 1, 1)

    def forward(self, x):
        # Encoder
        enc1 = self.enc1_2(self.enc1_1(x)); pool1 = self.pool1(enc1)
        enc2 = self.enc2_2(self.enc2_1(pool1)); pool2 = self.pool2(enc2)
        enc3 = self.enc3_2(self.enc3_1(pool2)); pool3 = self.pool3(enc3)
        enc4 = self.enc4_2(self.enc4_1(pool3)); pool4 = self.pool4(enc4)
        enc5 = self.enc5_1(pool4)

        # Transformer
        B, C, H, W = enc5.shape
        x_flat = self.flatten(enc5).permute(0, 2, 1)  # [B, H*W, C]
        x_trans = self.trans_block(x_flat)
        enc5 = x_trans.permute(0, 2, 1).view(B, C, H, W)

        # Decoder
        x = self.dec5_1(enc5)
        x = self.unpool4(x)
        x = self.dec4_2(torch.cat([x, enc4], dim=1))
        x = self.dec4_1(x)
        x = self.unpool3(x)
        x = self.dec3_2(torch.cat([x, enc3], dim=1))
        x = self.dec3_1(x)
        x = self.unpool2(x)
        x = self.dec2_2(torch.cat([x, enc2], dim=1))
        x = self.dec2_1(x)
        x = self.unpool1(x)
        x = self.dec1_2(torch.cat([x, enc1], dim=1))
        x = self.dec1_1(x)

        out = self.fc(x)
        return out
