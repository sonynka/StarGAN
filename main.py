import os
import argparse
from solver import Solver
from data_loader import get_loaders
from torch.backends import cudnn
from datetime import datetime
from logger import create_logger


def str2bool(v):
    return v.lower() in ('true')

def main(config):
    # For fast training
    cudnn.benchmark = True

    # create logging folders
    if not os.path.exists(config.output_path):
        os.makedirs(config.output_path)
    subfolders = ['logs', 'samples', 'models', 'results']
    for subfolder in subfolders:
        subfolder_path = os.path.join(config.output_path, subfolder, config.output_name)
        if not os.path.exists(subfolder_path):
            os.makedirs(subfolder_path)

    print_logger = create_logger(
        os.path.join(config.output_path, 'logs', config.output_name,
                     'train{}.log'.format(datetime.now().strftime("%Y%m%d-%H%M%S"))))
    print_logger.info('============ Initialized logger ============')
    print_logger.info('\n'.join('%s: %s' % (k, str(v)) for k, v
                          in sorted(dict(vars(config)).items())))

    # Data loader
    data_loaders = get_loaders(config.root, config.attrs, config.image_size,
                              config.batch_size)

    # Solver
    solver = Solver(data_loaders, config)

    if config.mode == 'train':
        solver.train()
    elif config.mode == 'test':
        solver.test()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model hyper-parameters
    parser.add_argument('--attrs', type=str, default='*',
                        help='attributes to train on')
    parser.add_argument('--c_dim', type=int, default=43)
    parser.add_argument('--c2_dim', type=int, default=8)
    parser.add_argument('--crop_size', type=int, default=256)
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--g_conv_dim', type=int, default=64)
    parser.add_argument('--d_conv_dim', type=int, default=64)
    parser.add_argument('--g_repeat_num', type=int, default=6)
    parser.add_argument('--d_repeat_num', type=int, default=6)
    parser.add_argument('--g_lr', type=float, default=0.0001)
    parser.add_argument('--d_lr', type=float, default=0.0001)
    parser.add_argument('--lambda_cls', type=float, default=1)
    parser.add_argument('--lambda_rec', type=float, default=10)
    parser.add_argument('--lambda_gp', type=float, default=10)
    parser.add_argument('--d_train_repeat', type=int, default=5)

    # Training settings
    parser.add_argument('--num_epochs', type=int, default=40)
    parser.add_argument('--num_epochs_decay', type=int, default=5)
    parser.add_argument('--num_iters', type=int, default=200000)
    parser.add_argument('--num_iters_decay', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)

    # pre-trained models
    parser.add_argument('--pretrained_model', type=str, default=None)
    parser.add_argument('--pretrained_model_path', type=str, default='./stargan/models/')

    # Test settings
    parser.add_argument('--test_model', type=str, default='20_1000')

    # Misc
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--use_tensorboard', type=str2bool, default=True)

    # Path

    parser.add_argument('--root', type=str, default='./data/fashion/')
    parser.add_argument('--metadata_path', type=str, default='./data/fashion/img_attr.csv')
    parser.add_argument('--output_path', type=str, default='./stargan/outputs/')
    parser.add_argument('--output_name', type=str, default='test')
    parser.add_argument('--num_val_imgs', type=int, default=10)

    # Step size
    parser.add_argument('--log_step', type=int, default=50)
    parser.add_argument('--sample_step', type=int, default=50)
    parser.add_argument('--model_save_step', type=int, default=100)

    config = parser.parse_args()
    print(config)
    main(config)