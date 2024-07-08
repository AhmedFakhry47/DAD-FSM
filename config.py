import argparse
from utils import process_config


def get_train_config():
    parser = argparse.ArgumentParser("Visual Transformer Train/Fine-tune")

    # basic config
    parser.add_argument("--exp-name", type=str, default="ft", help="experiment name")
    parser.add_argument("--train", type=int, default=1, help="train or not")
    parser.add_argument("--n-gpu", type=int, default=1, help="number of gpus to use")
    parser.add_argument("--tensorboard", default=False, action='store_true', help='flag of turnning on tensorboard')
    parser.add_argument("--checkpoint_dir", type=str, default="", help="model checkpoint to load weights")
    parser.add_argument("--checkpoint-path", type=str, default="", help="model checkpoint to load weights")
    parser.add_argument("--image-size", type=int, default=224, help="input image size", choices=[224, 384, 256]) #384 is the original
    parser.add_argument("--batch-size", type=int, default=8, help="batch size")
    parser.add_argument("--num-frames", type=int, default=4, help="number of frames")
    parser.add_argument("--num-workers", type=int, default=1, help="number of workers")
    parser.add_argument("--train-steps", type=int, default=38000, help="number of training/fine-tunning steps")
    parser.add_argument("--epochs", type=int, default=100, help="number of training/fine-tunning epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--wd", type=float, default=1e-4, help='weight decay')
    parser.add_argument("--warmup-steps", type=int, default=500, help='learning rate warm up steps')
    parser.add_argument("--data-dir", type=str, default='../data', help='data folder')
    parser.add_argument("--dataset", type=str, default='ImageNet', help="dataset for fine-tunning/evaluation")
    config = parser.parse_args()

    # model config
    #config = eval("get_{}_config".format(config.model_arch))(config)
    process_config(config)
    print_config(config)
    return config


def print_config(config):
    message = ''
    message += '----------------- Config ---------------\n'
    for k, v in sorted(vars(config).items()):
        comment = ''
        message += '{:>35}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '----------------- End -------------------'
    print(message)
