import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models
import torch.nn.functional as F
class inception_model(nn.Module):
    def __init__(self):
    	super(inception_model, self).__init__()
    	self.network = models.inception_v3(pretrained=True)
        for param in self.network.parameters():
            param.requires_grad = True

    def forward(self, img):
        x = self.network.Conv2d_1a_3x3(img)
        # 149 x 149 x 32
        x = self.network.Conv2d_2a_3x3(x)
        # 147 x 147 x 32
        x = self.network.Conv2d_2b_3x3(x)
        # 147 x 147 x 64
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 73 x 73 x 64
        x = self.network.Conv2d_3b_1x1(x)
        # 73 x 73 x 80
        x = self.network.Conv2d_4a_3x3(x)
        # 71 x 71 x 192
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 35 x 35 x 192
        x = self.network.Mixed_5b(x)
        # 35 x 35 x 256
        x = self.network.Mixed_5c(x)
        # 35 x 35 x 288
        x = self.network.Mixed_5d(x)
        # 35 x 35 x 288
        x = self.network.Mixed_6a(x)
        # 17 x 17 x 768
        x = self.network.Mixed_6b(x)
        # 17 x 17 x 768
        x = self.network.Mixed_6c(x)
        # 17 x 17 x 768
        x = self.network.Mixed_6d(x)
        # 17 x 17 x 768
        x = self.network.Mixed_6e(x)
        # 17 x 17 x 768
        if self.training and self.network.aux_logits:
            aux = self.network.AuxLogits(x)
        # 17 x 17 x 768
        x = self.network.Mixed_7a(x)
        # 8 x 8 x 1280
        x = self.network.Mixed_7b(x)
        # 8 x 8 x 2048
        feature = self.network.Mixed_7c(x)
        # 8 x 8 x 2048
        x = F.avg_pool2d(feature, kernel_size=8)
        # 1 x 1 x 2048
        x = F.dropout(x, training=self.training)
        # 1 x 1 x 2048
        x = x.view(x.size(0), -1)
        # 2048
        logits = self.network.fc(x)
        if self.training and self.network.aux_logits:
            return logits, aux
        return logits, feature

