import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """Two convolutional layers with ReLU activations, used in encoder/decoder blocks."""
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class AttentionGate(nn.Module):
    """Attention gate that filters skip-connection features using the gating signal
    from the decoder. Helps the model focus on lesion regions and suppress
    irrelevant background features."""
    def __init__(self, x_channels, g_channels, inter_channels):
        super(AttentionGate, self).__init__()

        self.theta_x = nn.Conv2d(x_channels, inter_channels, kernel_size=1)
        self.phi_g = nn.Conv2d(g_channels, inter_channels, kernel_size=1)
        self.psi = nn.Conv2d(inter_channels, 1, kernel_size=1)

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, g):
        theta_x = self.theta_x(x)
        phi_g = self.phi_g(g)

        f = self.relu(theta_x + phi_g)
        attention_map = self.sigmoid(self.psi(f))

        return x * attention_map


class AttentionUNet(nn.Module):
    """Attention U-Net for binary segmentation.

    Compared to the baseline U-Net, attention gates are added on every
    skip connection so the decoder can selectively use encoder features
    that are most relevant to the lesion region.
    """
    def __init__(self, in_channels=3, out_channels=1):
        super(AttentionUNet, self).__init__()

        # Encoder
        self.encoder1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.encoder2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.encoder3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        self.encoder4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(512, 1024)

        # Decoder with attention gates on each skip connection
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.att4 = AttentionGate(x_channels=512, g_channels=512, inter_channels=256)
        self.decoder4 = DoubleConv(1024, 512)

        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.att3 = AttentionGate(x_channels=256, g_channels=256, inter_channels=128)
        self.decoder3 = DoubleConv(512, 256)

        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.att2 = AttentionGate(x_channels=128, g_channels=128, inter_channels=64)
        self.decoder2 = DoubleConv(256, 128)

        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.att1 = AttentionGate(x_channels=64, g_channels=64, inter_channels=32)
        self.decoder1 = DoubleConv(128, 64)

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder path
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool1(e1))
        e3 = self.encoder3(self.pool2(e2))
        e4 = self.encoder4(self.pool3(e3))

        # Bottleneck
        b = self.bottleneck(self.pool4(e4))

        # Decoder path with attention on each skip connection
        d4 = self.upconv4(b)
        e4_att = self.att4(e4, d4)
        d4 = self.decoder4(torch.cat((e4_att, d4), dim=1))

        d3 = self.upconv3(d4)
        e3_att = self.att3(e3, d3)
        d3 = self.decoder3(torch.cat((e3_att, d3), dim=1))

        d2 = self.upconv2(d3)
        e2_att = self.att2(e2, d2)
        d2 = self.decoder2(torch.cat((e2_att, d2), dim=1))

        d1 = self.upconv1(d2)
        e1_att = self.att1(e1, d1)
        d1 = self.decoder1(torch.cat((e1_att, d1), dim=1))

        # Raw logits returned (sigmoid is applied inside the loss / metric functions
        # to stay consistent with the baseline U-Net pipeline).
        return self.final_conv(d1)
