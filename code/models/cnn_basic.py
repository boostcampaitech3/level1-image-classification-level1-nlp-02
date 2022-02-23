import torch 
import torch.nn as nn

class CNN_Basic(nn.Module):
    def __init__(self, name='cnn', xdim=[3, 256, 256], ksize=3, cdims=[16, 32, 64], hdims=[576, 128], ydim=18, USE_BATCHNORM=False):
        super().__init__()
        self.name = name
        self.xdim = xdim
        self.ksize = ksize
        self.cdims = cdims
        self.hdims = hdims
        self.ydim = ydim
        self.USE_BATCHNORM = USE_BATCHNORM
        
        self.layers = []

        prev_cdim = self.xdim[0]
        for cdim in self.cdims:
            self.layers.append(
                nn.Conv2d(
                    in_channels=prev_cdim,
                    out_channels=cdim,
                    kernel_size = self.ksize,
                    padding=self.ksize//2
                )
            )
            if self.USE_BATCHNORM:
                self.layers.append(nn.BatchNorm2d(cdim))

            self.layers.append(nn.ReLU(True))
            self.layers.append(nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)))
            #self.layers.append(nn.Dropout2d(p=0.5))
            prev_cdim = cdim

        # Dense layers
        self.layers.append(nn.Flatten())
        prev_hdim = prev_cdim * (self.xdim[1]//(2**len(self.cdims))) * (self.xdim[2]//(2 ** len(self.cdims)))
        for hdim in self.hdims:
            self.layers.append(nn.Linear(prev_hdim, hdim, bias=True))
            self.layers.append(nn.ReLU(True))
            prev_hdim = hdim

        self.layers.append(nn.Linear(prev_hdim, self.ydim, bias=True))
        
        # LogSoftmax
        self.layers.append(nn.LogSoftmax(dim=-1))
        
        # connect all layers
        self.net = nn.Sequential()
        for l_idx, layer in enumerate(self.layers):
            layer_name = "%s_%02d" % (type(layer).__name__.lower(), l_idx)
            self.net.add_module(layer_name, layer)
        self.init_param()

    def init_param(self):
        for m in self.modules():
            if isinstance(m,nn.Conv2d): # init conv
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m,nn.BatchNorm2d): # init BN
                nn.init.constant_(m.weight,1)
                nn.init.constant_(m.bias,0)
            elif isinstance(m,nn.Linear): # lnit dense
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)
            
    def forward(self,x):
        return self.net(x)
        

class CNN_Boostcamp(nn.Module):
    def __init__(self, name='cnn', xdim=[3, 256, 256], ksize=3, cdims=[32, 64, 128, 256, 512, 1024], hdims=[1024, 128], ydim=18):
        super().__init__()

        # identity mapping, input과 output의 feature map size, filter 수 동일
        self.ksize = ksize
        

        self.layers1 = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.Conv2D(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2, 2))        
        )

        self.residual1 = nn.Sequential(
            nn.Conv2D(in_channels=64, out_channels=32, kernel_size=1),
            nn.Conv2D(in_channels=32, out_channels=64, kernel_size=3, padding=1),      
        )

        self.layers2 = nn.Conv2D(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.shortcut = nn.Sequential()
