import _init_paths
import os
import argparse
import pickle
from tqdm import tqdm

import torch
from torchvision import datasets, transforms
from torchvision.datasets import MNIST, FashionMNIST
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

from model import net, embedding
from utils.gen_utils import make_dir_if_not_exist
from config.base_config import cfg, cfg_from_file
import cv2
import numpy as np
from sklearn.manifold import TSNE
import matplotlib as mpl
import matplotlib.pyplot as plt
from dataloader import custom_dset  # Import your custom dataset module
from dataloader.triplet_img_loader import BaseLoader

def main():
    torch.manual_seed(1)
    if args.cuda:
        torch.cuda.manual_seed(1)

    exp_dir = os.path.join("data", args.exp_name)
    make_dir_if_not_exist(exp_dir)

    if args.pkl is not None:
        input_file = open(args.pkl, 'rb')
        final_data = pickle.load(input_file)
        input_file.close()
        embeddings = final_data['embeddings']
        labels = final_data['labels']
        vis_tSNE(embeddings, labels)
    else:
        embeddingNet = None
        if args.dataset in ['s2s', 'vggface2', 'custom']:
            embeddingNet = embedding.EmbeddingResnet()
        elif args.dataset in ['mnist', 'fmnist']:
            embeddingNet = embedding.EmbeddingLeNet()
        else:
            print(f"Dataset {args.dataset} not supported")
            return

        model_dict = None
        if args.ckp is not None:
            if os.path.isfile(args.ckp):
                print(f"=> Loading checkpoint '{args.ckp}'")
                try:
                    model_dict = torch.load(args.ckp, weights_only=True)['state_dict']
                except Exception:
                    model_dict = torch.load(args.ckp, map_location='cpu',weights_only=True)['state_dict']
                print(f"=> Loaded checkpoint '{args.ckp}'")
            else:
                print(f"=> No checkpoint found at '{args.ckp}'")
                return
        else:
            print("Please specify a model")
            return

        model_dict_mod = {}
        for key, value in model_dict.items():
            new_key = '.'.join(key.split('.')[2:])
            model_dict_mod[new_key] = value
        model = embeddingNet.to(device)
        model.load_state_dict(model_dict_mod)

        kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

        if args.dataset in ['mnist', 'fmnist']:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            if args.dataset == 'mnist':
                train_dataset = MNIST('data/MNIST', train=True, download=True, transform=transform)
            elif args.dataset == 'fmnist':
                train_dataset = FashionMNIST('data/FashionMNIST', train=True, download=True, transform=transform)
            data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, **kwargs)
        elif args.dataset == 'custom':
            dset_obj = custom_dset.Custom()  # Instantiate your custom dataset
            dset_obj.load()  # Load dataset and generate triplets
            train_triplets = [dset_obj.getTriplet() for _ in range(args.num_train_samples)]
            transform = transforms.Compose([
                transforms.ToTensor(),  # Customize as needed
                transforms.Normalize((0.5,), (0.5,))
            ])
            data_loader = torch.utils.data.DataLoader(
                BaseLoader(train_triplets, transform=transform),
                batch_size=64, shuffle=True, **kwargs
            )
        else:
            print(f"Dataset {args.dataset} not supported")
            return

        embeddings, labels = generate_embeddings(data_loader, model)

        final_data = {
            'embeddings': embeddings,
            'labels': labels
        }

        dst_dir = os.path.join('data', args.exp_name, 'tSNE')
        make_dir_if_not_exist(dst_dir)

        output_file = open(os.path.join(dst_dir, 'tSNE.pkl'), 'wb')
        pickle.dump(final_data, output_file)
        output_file.close()

        vis_tSNE(embeddings, labels)


def generate_embeddings(data_loader, model):
    with torch.no_grad():
        model.eval()
        labels = []
        embeddings = []
        for batch_idx, data in tqdm(enumerate(data_loader)):
            batch_imgs1, batch_imgs2, batch_imgs3 = data
            batch_imgs1 = Variable(batch_imgs1.to(device))
            batch_imgs2 = Variable(batch_imgs2.to(device))
            batch_imgs3 = Variable(batch_imgs3.to(device))
            
            # Concatenate the triplet images
            batch_imgs = torch.cat((batch_imgs1, batch_imgs2, batch_imgs3), dim=0)
            batch_embeddings = model(batch_imgs)
            
            batch_embeddings1, batch_embeddings2, batch_embeddings3 = torch.split(batch_embeddings, batch_imgs1.size(0))
            
            embeddings.append(batch_embeddings1.cpu().numpy())
            embeddings.append(batch_embeddings2.cpu().numpy())
            embeddings.append(batch_embeddings3.cpu().numpy())
            
            labels.extend([0] * batch_imgs1.size(0))  # Placeholder labels for embeddings
            labels.extend([1] * batch_imgs2.size(0))  # Placeholder labels for embeddings
            labels.extend([2] * batch_imgs3.size(0))  # Placeholder labels for embeddings
            
        embeddings = np.concatenate(embeddings, axis=0)
        labels = np.array(labels)
        
    return embeddings, labels


def vis_tSNE(embeddings, labels):
    num_samples = args.tSNE_ns if args.tSNE_ns < embeddings.shape[0] else embeddings.shape[0]
    X_embedded = TSNE(n_components=2).fit_transform(embeddings[0:num_samples, :])

    fig, ax = plt.subplots()
    x, y = X_embedded[:, 0], X_embedded[:, 1]
    colors = plt.cm.rainbow(np.linspace(0, 1, 3))  # Adjust number of colors based on labels
    sc = ax.scatter(x, y, c=labels[0:num_samples], cmap=mpl.colors.ListedColormap(colors))
    plt.colorbar(sc)
    plt.savefig(os.path.join('data', args.exp_name, 'tSNE', 'tSNE_' + str(num_samples) + '.jpg'))
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Siamese Example')
    parser.add_argument('--exp_name', default='exp0', type=str,
                        help='name of experiment')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--ckp', default=None, type=str,
                        help='path to load checkpoint')
    parser.add_argument('--dataset', type=str, default='custom', metavar='M',
                        help='Dataset (default: custom)')
    parser.add_argument('--pkl', default=None, type=str,
                        help='Path to load embeddings')
    parser.add_argument('--tSNE_ns', default=5000, type=int,
                        help='Num samples to create a tSNE visualisation')
    parser.add_argument('--num_train_samples', default=1000, type=int,
                        help='Number of training samples to generate')
    parser.add_argument('--num_test_samples', default=100, type=int,
                        help='Number of test samples to generate')
    global args, device
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()
    cfg_from_file("config/test.yaml")

    if args.cuda:
        device = 'cuda'
        if args.gpu_devices is None:
            args.gpu_devices = [0]
    else:
        device = 'cpu'
    main()
