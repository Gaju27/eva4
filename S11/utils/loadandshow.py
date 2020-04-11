import datetime
import torch
# import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets


class loadandshow:

    # helper function to un-normalize and display an image
    @staticmethod
    def imshow(img):
        img = img / 2 + 0.5  # unnormalize
        plt.imshow(np.transpose(img, (1, 2, 0)))  # convert from Tensor image

    def printdatetime(self):
        print("Model execution started at:" + datetime.datetime.today().ctime())


    def savemodel(model, epoch, path, optimizer_state_dict=None, train_losses=None, train_acc=None, test_acc=None,
                  test_losses=None):
        # Prepare model model saving directory.
        # save_dir = os.path.join(os.getcwd(), 'saved_models')
        t = datetime.datetime.today()

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer_state_dict,
            'train_losses': train_losses,
            'train_acc': train_acc,
            'test_losses': test_losses,
            'test_acc': test_acc,
            # 'lr_data': lr_data,
            # 'reg_loss_l1': reg_loss_l1
        }, path)

    def loadmodel(path):
        checkpoint = torch.load(path)
        # epoch = checkpoint['epoch']
        # model_state_dict = checkpoint['model_state_dict']
        # optimizer_state_dict = checkpoint['optimizer_state_dict']
        # train_losses = checkpoint['train_losses']
        # train_acc = checkpoint['train_acc']
        # test_losses = checkpoint['test_losses']
        # test_acc = checkpoint['test_acc']
        # lr_data = checkpoint['lr_data']
        return checkpoint