"""
Module containing neural network architectures (MLP and ConvNet)
"""
import torch.nn as nn



class ConvNet(nn.Module):
    def __init__(self, out_dim, channel_sizes, activation=nn.ReLU):
        super(ConvNet, self).__init__()

        def get_output_dim(input_dim, kernel_size, stride,pad=0):
            output_dim = (input_dim +2*pad-kernel_size) // stride +1
            return output_dim

        output_dim = get_output_dim(get_output_dim(32, kernel_size=3, stride=1,pad=1), kernel_size=2, stride=2)
        output_dim = get_output_dim(get_output_dim(output_dim, kernel_size=3, stride=1,pad=1), kernel_size=2, stride=2)
        output_dim = get_output_dim(get_output_dim(output_dim, kernel_size=3, stride=1,pad=1), kernel_size=2, stride=2)
        output_dim = get_output_dim(get_output_dim(output_dim, kernel_size=3, stride=1,pad=1), kernel_size=2, stride=2)
        output_dim = get_output_dim(get_output_dim(output_dim, kernel_size=3, stride=1,pad=1), kernel_size=2, stride=2)
        layer_list = [
         nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3),padding=1, bias=False),
         activation(),
         nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3),padding=1, bias=False),
         activation(),
         nn.MaxPool2d(kernel_size=(2, 2), stride=(2,2)),
        
         nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1,bias=False),
         activation(),
         nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1,bias=False),
         activation(),
         nn.MaxPool2d(kernel_size=(2, 2), stride=(2,2)),
        
         nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=1,bias=False),
         activation(),
         nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding=1,bias=False),
         activation(),
         nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding=1,bias=False),
         activation(),
         nn.MaxPool2d(kernel_size=(2, 2), stride=(2,2)),

         nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), padding=1,bias=False),
         activation(),
         nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1,bias=False),
         activation(),
         nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1,bias=False),
         activation(),
         nn.MaxPool2d(kernel_size=(2, 2), stride=(2,2)),

         nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1,bias=False),
         activation(),
         nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1,bias=False),
         activation(),
         nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1,bias=False),
         activation(),
         nn.MaxPool2d(kernel_size=(2, 2), stride=(2,2)),
         
         nn.Flatten(),
         nn.Linear(output_dim*output_dim*512, out_dim, bias=False)
        ]

        self.net = nn.Sequential(*layer_list)


    def forward(self, x):
        return self.net(x)