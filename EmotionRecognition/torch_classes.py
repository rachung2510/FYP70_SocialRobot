from torch import nn
import torch

class FCNNModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super(FCNNModel, self).__init__()
        
        input_layer_size = kwargs['input_layer_size']
        hidden_layer_size = kwargs['hidden_layer_size']
        num_classes = kwargs['num_classes']
        
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_layer_size, hidden_layer_size),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, hidden_layer_size),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, num_classes),
        )
    
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

def conv_block(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), 
              nn.BatchNorm2d(out_channels), 
              nn.ELU(inplace=True)]
    if pool: layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)

##class ImageClassificationBase(nn.Module)
##    def training_step(self, batch):
##        images, labels = batch 
##        out = self(images)
##        loss = F.cross_entropy(out, labels)
##        acc = accuracy(out, labels)
##        return loss, acc
##    
##    def validation_step(self, batch):
##        images, labels = batch 
##        out = self(images)
##        loss = F.cross_entropy(out, labels)
##        acc = accuracy(out, labels)
##        return {'val_loss': loss.detach(), 'val_acc': acc}
##        
##    def validation_epoch_end(self, outputs):
##        batch_losses = [x['val_loss'] for x in outputs]
##        epoch_loss = torch.stack(batch_losses).mean()
##        batch_accs = [x['val_acc'] for x in outputs]
##        epoch_acc = torch.stack(batch_accs).mean()
##        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
##    
##    def epoch_end(self, epoch, result, total, print_epoch=1):
##        if (not epoch % print_epoch) or epoch==total-1:
##            print('epoch=%d  ' % epoch + (' ' if epoch<10 else ''), end="")
##            print('last_lr={:.5f}   train loss={:.4f}   test loss={:.4f}   train accuracy={:.4f}   test accuracy={:.4f}'.format(
##                result['lrs'][-1], result['train_loss'], result['val_loss'], result['train_acc'], result['val_acc']))

class ResNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        
        self.conv1 = conv_block(in_channels, 128)
        self.conv2 = conv_block(128, 128, pool=True)
        self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))
        self.drop1 = nn.Dropout(0.5)
        
        self.conv3 = conv_block(128, 256)
        self.conv4 = conv_block(256, 256, pool=True)
        self.res2 = nn.Sequential(conv_block(256, 256), conv_block(256, 256))
        self.drop2 = nn.Dropout(0.5)
        
        self.conv5 = conv_block(256, 512)
        self.conv6 = conv_block(512, 512, pool=True)
        self.res3 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))
        self.drop3 = nn.Dropout(0.5)
        
        self.classifier = nn.Sequential(nn.MaxPool2d(6), 
                                        nn.Flatten(),
                                        nn.Linear(512, num_classes))
        
    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.drop1(out)
        
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.drop2(out)
        
        out = self.conv5(out)
        out = self.conv6(out)
        out = self.res3(out) + out
        out = self.drop3(out)
        
        out = self.classifier(out)
        return out

def get_default_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    
def to_device(data, device):
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)
