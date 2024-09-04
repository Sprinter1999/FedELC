from resnets.lenet import LeNet
from resnets.model_resnet import ResNet18, ResNet34, ResNet50, ResNet20
# from resnets.model_resnet_official import ResNet50
import torchvision.models as models
import torch.nn as nn



def build_model(args):
    # choose different Neural network model for different args or input
    if args.model == 'lenet':
        netglob = LeNet(args.num_classes)
        # netglob = netglob.to(args.device)

    elif args.model == 'resnet18':
        netglob = ResNet18(args.num_classes)
        # netglob = netglob.to(args.device)

    elif args.model == 'resnet20':
        netglob = ResNet20(args.num_classes)
        # netglob = netglob.to(args.device)

    elif args.model == 'resnet34':
        netglob = ResNet34(args.num_classes)
        # netglob = netglob.to(args.device)

    elif args.model == 'resnet50':
        netglob = ResNet50(args.num_classes)
        # if args.pretrained:
        #     model = models.resnet50(pretrained=True)
        #     netglob.load_state_dict(model.state_dict())
        # netglob.fc = nn.Linear(2048, args.num_classes)
        # netglob = netglob.to(args.device)

    elif args.model == 'vgg11':
        netglob = models.vgg11()
        netglob.fc = nn.Linear(4096, args.num_classes)
        # netglob = netglob.to(args.device)

    else:
        exit('Error: unrecognized model')

    return netglob
