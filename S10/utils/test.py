import torch
import torchvision
import torchvision.transforms as transforms
from .data_loader import *


def __init__(self,accuracy_max):
    self.accuracy_max = 0 
    
def test(net,testloader, device,criterion,losses,accuracy,correct_samples, incorrect_samples, sample_count=25, last_epoch=False):
    #testloader = dataloader_cifar10(root='./data',, split='test', batch_size=dataloader_args.batch_size)
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    accuracy = 0
    accuracy_max = 0
    classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            img_batch = inputs  # This is done to keep data in CPU
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
#             total += targets.size(0)
            result = predicted.eq(targets.view_as(predicted))
            
             # Save correct and incorrect samples
            if last_epoch:
                for i in range(len(list(result))):
                    if not list(result)[i] and len(incorrect_samples) < sample_count:
                        incorrect_samples.append({
                            'prediction': list(predicted)[i],
                            'label': list(targets.view_as(predicted))[i],
                            'image': img_batch[i]
                        })
                    elif list(result)[i] and len(correct_samples) < sample_count:
                        correct_samples.append({
                            'prediction': list(predicted)[i],
                            'label': list(targets.view_as(predicted))[i],
                            'image': img_batch[i]
                        })
            
            correct += result.sum().item()
            
        test_loss /= len(testloader.dataset)
        losses.append(test_loss)
        accuracy=100.00 * correct / len(testloader.dataset)
        
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                test_loss, correct, len(testloader.dataset),
                accuracy))
            
        if accuracy >=88:
            class_correct = list(0. for i in range(10))
            class_total = list(0. for i in range(10))
            with torch.no_grad():
                for data in testloader:
                    images, labels = data
                    images, labels = images.to(device), labels.to(device)
                    outputs = net(images)
                    _, predicted = torch.max(outputs, 1)
                    c = (predicted == labels).squeeze()
                    for i in range(4):
                        label = labels[i]
                        class_correct[label] += c[i].item()
                        class_total[label] += 1


            for i in range(10):
                print('Accuracy of %5s : %2d %%' % (
                    classes[i], 100 * class_correct[i] / class_total[i]))
                
        return accuracy