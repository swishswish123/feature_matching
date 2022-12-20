import torchvision.transforms as transforms
from utils import config
import torch

def main():
    #model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
    #                       in_channels=6, out_channels=3, init_features=32, pretrained=False)
    #model.load_state_dict(torch.load('best_model.pth.tar'))

    model = torch.load('/Users/aure/Documents/CARES/feature_matching/fake_out/best_model.pth.tar').to(config.DEVICE)
    model.eval()

    (input, label) = [30]
    input = input.cuda()

if __name__ == '__main__':

    main()
