import torch.nn as nn

class AlexNet(nn.Module):
    def __init__(self,num_classes=10):
        super(AlexNet,self).__init__()
        self.features=nn.Sequential(
            nn.Conv2d(3,96,kernel_size=11,stride=4,padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2),

            nn.Conv2d(96,256,kernel_size=5,stride=1,padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2),

            nn.Conv2d(256,384,kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(384,256,kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(384,256,kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2),
        )

        self.classifier=nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256*6*6,256*4*4),
            nn.ReLU(inplace=True),

            nn.Dropout(p=0.5),
            nn.Linear(256*4*4,256*4*4),
            nn.ReLU(inplace=True),

            nn.Linear(256*4*4,num_classes)
        )

    def forward(self,x):
        x=self.features(x)
        x=x.view(x.size(0),256*6*6)
        x=self.classifier(x)
        return x