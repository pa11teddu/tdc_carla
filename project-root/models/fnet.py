import torch
import torch.nn as nn
import torch.nn.functional as F

class FPN(nn.Module):
    def __init__(self, num_classes):
        super(FPN, self).__init__()
        
        # Encoder (backbone)
        self.encoder1 = self.conv_block(3, 64)
        self.encoder2 = self.conv_block(64, 128)
        self.encoder3 = self.conv_block(128, 256)
        self.encoder4 = self.conv_block(256, 512)
        self.encoder5 = self.conv_block(512, 1024)
        
        # Lateral connections (1x1 convs to reduce channels)
        self.lateral5 = nn.Conv2d(1024, 256, kernel_size=1)
        self.lateral4 = nn.Conv2d(512, 256, kernel_size=1)
        self.lateral3 = nn.Conv2d(256, 256, kernel_size=1)
        self.lateral2 = nn.Conv2d(128, 256, kernel_size=1)
        self.lateral1 = nn.Conv2d(64, 256, kernel_size=1)
        
        # FPN output layers
        self.fpn_conv = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        
        # Final prediction layers
        self.predict = nn.Conv2d(256, num_classes, kernel_size=1)
    
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def upsample_add(self, x, y):
        """Upsample x and add it to y."""
        return F.interpolate(x, size=y.shape[2:], mode='bilinear', align_corners=False) + y
    
    def forward(self, x):
        # Bottom-up pathway (encoder)
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(F.max_pool2d(enc1, kernel_size=2))
        enc3 = self.encoder3(F.max_pool2d(enc2, kernel_size=2))
        enc4 = self.encoder4(F.max_pool2d(enc3, kernel_size=2))
        enc5 = self.encoder5(F.max_pool2d(enc4, kernel_size=2))
        
        # Lateral connections
        lat5 = self.lateral5(enc5)
        lat4 = self.lateral4(enc4)
        lat3 = self.lateral3(enc3)
        lat2 = self.lateral2(enc2)
        lat1 = self.lateral1(enc1)
        
        # Top-down pathway and lateral connections
        map5 = self.fpn_conv(lat5)
        map4 = self.fpn_conv(self.upsample_add(map5, lat4))
        map3 = self.fpn_conv(self.upsample_add(map4, lat3))
        map2 = self.fpn_conv(self.upsample_add(map3, lat2))
        map1 = self.fpn_conv(self.upsample_add(map2, lat1))
        
        # Final prediction
        # Upscale all maps to the original input resolution
        output_size = x.shape[2:]
        
        pred1 = F.interpolate(self.predict(map1), size=output_size, mode='bilinear', align_corners=False)
        pred2 = F.interpolate(self.predict(map2), size=output_size, mode='bilinear', align_corners=False)
        pred3 = F.interpolate(self.predict(map3), size=output_size, mode='bilinear', align_corners=False)
        pred4 = F.interpolate(self.predict(map4), size=output_size, mode='bilinear', align_corners=False)
        pred5 = F.interpolate(self.predict(map5), size=output_size, mode='bilinear', align_corners=False)
        
        # Combine predictions (you can modify this based on your needs)
        return (pred1 + pred2 + pred3 + pred4 + pred5) / 5.0

if __name__ == "__main__":
    model = FPN(num_classes=3)
    input_tensor = torch.randn(1, 3, 512, 512)
    output = model(input_tensor)
    print("Output shape:", output.shape)  # Expected shape: [1, num_classes, 512, 512] 
