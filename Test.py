
import torch
import torch.nn as nn
import torchinfo

class ResidualConv(nn.Module):
    def __init__(self, input_dim, output_dim, stride, padding):
        super(ResidualConv, self).__init__()

        self.conv_block = nn.Sequential(
            nn.BatchNorm2d(input_dim),
            nn.ReLU(),
            nn.Conv2d(
                input_dim, output_dim, kernel_size=3, stride=stride, padding=padding, groups=4    ## added groups
            ),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
            nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1, groups=4)  ## added groups,
        )
        self.conv_skip = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=stride, padding=1, groups=4),  ## added groups
            nn.BatchNorm2d(output_dim),                                   
        )


    def forward(self, x):

        return self.conv_block(x) + self.conv_skip(x)
    


if __name__=="__main__":
    x = torch.randn(1, 32, 256, 256)
    model = ResidualConv(32, 64, 1, 1)
    output=model(x)
    print(output.shape)
    torchinfo.summary(model, input_size=(1, 32, 256, 256))