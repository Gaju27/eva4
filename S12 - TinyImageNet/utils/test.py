import torch
import torchvision
import torchvision.transforms as transforms
from .data_loader import *

test_losses = []
test_acc = []

# def test(net, device,criterion, testloader):
#     net.eval()
#     test_loss = 0
#     correct = 0
#     total = 0
#     accuracy = 0
#     
#     with torch.no_grad():
#         for batch_idx, (inputs, targets) in enumerate(testloader):
#             inputs, targets = inputs.to(device), targets.to(device)
#             outputs = net(inputs)
#             loss = criterion(outputs, targets)
# 
#             test_loss += loss.item()
#             _, predicted = outputs.max(1)
#             total += targets.size(0)
#             correct += predicted.eq(targets).sum().item()
# 
#             # progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
#             #     % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
#         test_loss /= len(testloader.dataset)
#         test_losses.append(test_loss)
# 
#         print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
#         test_loss, correct, len(testloader.dataset), 100. * correct / len(testloader.dataset)))
#     
#         test_acc.append(100. * correct / len(testloader.dataset))
#         
#     return accuracy
        
def test(net, device,criterion, testloader):
    net.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            pred = outputs.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(targets.view_as(pred)).sum().item()

    test_loss /= len(testloader.dataset)
    test_losses.append(test_loss)

    print('\nTest set: Average loss: {:.4f}, Test Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(testloader.dataset),
        100. * correct / len(testloader.dataset)))
    
    test_acc.append(100. * correct / len(testloader.dataset))
    
    return test_acc
                   