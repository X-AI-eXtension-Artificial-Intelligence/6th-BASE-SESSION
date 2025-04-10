import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        def CBR2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True, dropout_rate=0.2):
            layers = []
            # Conv
            layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                 kernel_size=kernel_size, stride=stride, padding=padding,
                                 bias=bias)]
            # BatchNorm
            layers += [nn.BatchNorm2d(num_features=out_channels)]
            # ReLU
            layers += [nn.ReLU()]
            # Dropout
            layers += [nn.Dropout2d(p=dropout_rate)]

            cbr = nn.Sequential(*layers)
            return cbr

        ### Contracting path (Encoder)
        self.enc1_1 = CBR2d(in_channels=1, out_channels=32, dropout_rate=0.1)
        self.enc1_2 = CBR2d(in_channels=32, out_channels=32, dropout_rate=0.1)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.enc2_1 = CBR2d(in_channels=32, out_channels=64, dropout_rate=0.1)
        self.enc2_2 = CBR2d(in_channels=64, out_channels=64, dropout_rate=0.1)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.enc3_1 = CBR2d(in_channels=64, out_channels=128, dropout_rate=0.2)
        self.enc3_2 = CBR2d(in_channels=128, out_channels=128, dropout_rate=0.2)
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.enc4_1 = CBR2d(in_channels=128, out_channels=256, dropout_rate=0.2)
        self.enc4_2 = CBR2d(in_channels=256, out_channels=256, dropout_rate=0.2)
        self.pool4 = nn.MaxPool2d(kernel_size=2)

        self.enc5_1 = CBR2d(in_channels=256, out_channels=512, dropout_rate=0.3)
        self.pool5 = nn.MaxPool2d(kernel_size=2)

        self.enc6_1 = CBR2d(in_channels=512, out_channels=1024, dropout_rate=0.3)
        self.pool6 = nn.MaxPool2d(kernel_size=2)

        self.enc7_1 = CBR2d(in_channels=1024, out_channels=2048, dropout_rate=0.3)

        ### Expansive path (Decoder)
        self.dec7_1 = CBR2d(in_channels=2048, out_channels=1024, dropout_rate=0.3)
        self.unpool6 = nn.ConvTranspose2d(in_channels=1024, out_channels=1024,
                                          kernel_size=2, stride=2, padding=0, bias=True)

        self.dec6_2 = CBR2d(in_channels=2 * 1024, out_channels=512, dropout_rate=0.3)
        self.unpool5 = nn.ConvTranspose2d(in_channels=512, out_channels=512,
                                          kernel_size=2, stride=2, padding=0, bias=True)

        self.dec5_2 = CBR2d(in_channels=2 * 512, out_channels=256, dropout_rate=0.2)
        self.unpool4 = nn.ConvTranspose2d(in_channels=256, out_channels=256,
                                          kernel_size=2, stride=2, padding=0, bias=True)

        self.dec4_2 = CBR2d(in_channels=2 * 256, out_channels=128, dropout_rate=0.2)
        self.unpool3 = nn.ConvTranspose2d(in_channels=128, out_channels=128,
                                          kernel_size=2, stride=2, padding=0, bias=True)

        self.dec3_2 = CBR2d(in_channels=2 * 128, out_channels=64, dropout_rate=0.1)
        self.unpool2 = nn.ConvTranspose2d(in_channels=64, out_channels=64,
                                          kernel_size=2, stride=2, padding=0, bias=True)

        self.dec2_2 = CBR2d(in_channels=2 * 64, out_channels=32, dropout_rate=0.1)
        self.unpool1 = nn.ConvTranspose2d(in_channels=32, out_channels=32,
                                          kernel_size=2, stride=2, padding=0, bias=True)

        self.dec1_2 = CBR2d(in_channels=2 * 32, out_channels=32, dropout_rate=0.1)
        self.dec1_1 = CBR2d(in_channels=32, out_channels=32, dropout_rate=0.1)

        # 최종 1x1 Convolution
        self.fc = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        enc1_1 = self.enc1_1(x)
        enc1_2 = self.enc1_2(enc1_1)
        pool1 = self.pool1(enc1_2)

        enc2_1 = self.enc2_1(pool1)
        enc2_2 = self.enc2_2(enc2_1)
        pool2 = self.pool2(enc2_2)

        enc3_1 = self.enc3_1(pool2)
        enc3_2 = self.enc3_2(enc3_1)
        pool3 = self.pool3(enc3_2)

        enc4_1 = self.enc4_1(pool3)
        enc4_2 = self.enc4_2(enc4_1)
        pool4 = self.pool4(enc4_2)

        enc5_1 = self.enc5_1(pool4)
        pool5 = self.pool5(enc5_1)

        enc6_1 = self.enc6_1(pool5)
        pool6 = self.pool6(enc6_1)

        enc7_1 = self.enc7_1(pool6)

        dec7_1 = self.dec7_1(enc7_1)
        unpool6 = self.unpool6(dec7_1)
        cat6 = torch.cat((unpool6, enc6_1), dim=1)

        dec6_2 = self.dec6_2(cat6)
        unpool5 = self.unpool5(dec6_2)
        cat5 = torch.cat((unpool5, enc5_1), dim=1)

        dec5_2 = self.dec5_2(cat5)
        unpool4 = self.unpool4(dec5_2)
        cat4 = torch.cat((unpool4, enc4_2), dim=1)

        dec4_2 = self.dec4_2(cat4)
        unpool3 = self.unpool3(dec4_2)
        cat3 = torch.cat((unpool3, enc3_2), dim=1)

        dec3_2 = self.dec3_2(cat3)
        unpool2 = self.unpool2(dec3_2)
        cat2 = torch.cat((unpool2, enc2_2), dim=1)

        dec2_2 = self.dec2_2(cat2)
        unpool1 = self.unpool1(dec2_2)
        cat1 = torch.cat((unpool1, enc1_2), dim=1)

        dec1_2 = self.dec1_2(cat1)
        dec1_1 = self.dec1_1(dec1_2)

        x = self.fc(dec1_1)

        return x
