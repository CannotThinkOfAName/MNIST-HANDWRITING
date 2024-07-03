import PIL.Image
import torch
import torchvision.transforms.functional as functional
from torchvision.datasets import mnist
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import PIL as PIL
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.style.use("classic")


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    test_dataset = mnist.MNIST(root = "./test", train=False, transform=ToTensor())
    test_loader = DataLoader(test_dataset, shuffle=True)
    model = torch.load('.\models\mnist_0.983.pkl')
    model.eval()
    i = 1
    fig=None
    for idx, (test_x, test_label) in enumerate(test_loader):
        test_x = test_x.to(device)
        test_label = test_label.to(device)
        predict_y = model(test_x.float()).detach()
        predict_y = torch.argmax(predict_y[0], dim=-1)
        image = functional.to_pil_image(test_x[0])
        image = image.resize((50, 50))
        plt.figure("预测结果")
        plt.subplot(1, 10, i)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(image)
        plt.title(predict_y.item())
        i = i+1
        if(i==11):
            break
    
    plt.tight_layout() 
    plt.show()

