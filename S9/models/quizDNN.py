import torch.nn as nn
import torch.nn.functional as F


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.con1=nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=0), # input: 3x3 output: 32	Rec.field: 1	JumpOut: 1
            nn.ReLU(),
            nn.BatchNorm2d(16), 
         )
        
        self.con2=nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1), # input: 32  output: 32	Rec.field: 3	JumpOut: 1
            nn.ReLU(),
            nn.BatchNorm2d(32), 
         )
        
        self.con3=nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0), # input: 32  output: 30	Rec.field: 5	JumpOut: 1
            nn.ReLU(),
            nn.BatchNorm2d(64), 
         )
         
        self.pool1=nn.MaxPool2d(2,2) # input: 30  output: 15	Rec.field: 6	JumpOut: 1
         
        self.con4=nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1), # input: 15  output: 15	Rec.field: 10	JumpOut: 2
            nn.ReLU(),
            nn.BatchNorm2d(128), 
         )
         
        self.con5=nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1), # input: 15  output: 15	Rec.field: 14	JumpOut: 2
            nn.ReLU(),
            nn.BatchNorm2d(128), 
         )
         
        self.con6=nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1), # input: 15  output: 15	Rec.field: 18	JumpOut: 2
            nn.ReLU(),
            nn.BatchNorm2d(64), 
         )
         
        self.pool2=nn.MaxPool2d(2,2) # input: 7   output: 7	Rec.field: 20	JumpOut: 2
        
        self.con7=nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1), # input: 7   output: 7	Rec.field: 28	JumpOut: 4
            nn.ReLU(),
            nn.BatchNorm2d(32), 
         )
         
        self.con8=nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1), # input: 7   output: 7	Rec.field: 36	JumpOut: 4
            nn.ReLU(),
            nn.BatchNorm2d(16), 
         )
         
        self.con9=nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=3, stride=1, padding=1), # input: 7   output: 7	Rec.field: 44	JumpOut: 4
         )

        self.avg_pool=nn.AvgPool2d(kernel_size=7)
        
        

    def forward(self, x):
        x = self.con1(x)
        x = self.con2(x)
        x = self.con3(x)
        x = self.pool1(x)
        x = self.con4(x)
        x = self.con5(x)
        x = self.con6(x)
        x = self.pool2(x)
        x = self.con7(x)
        x = self.con8(x)
        x = self.con9(x)
        x = self.avg_pool(x)
        x = x.view(-1, 10)
        return x
    
def ResNet18():
    return Net()