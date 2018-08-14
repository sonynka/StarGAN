import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
import datetime
from torch.autograd import grad
from torch.autograd import Variable
from torchvision.utils import save_image
from torchvision import transforms
from model import Generator
from model import Discriminator
from PIL import Image
from logging import getLogger

print_logger = getLogger()


class Solver(object):

    def __init__(self, data_loaders, config):
        # Data loader
        self.data_loaders = data_loaders
        self.attrs = config.attrs

        # Model hyper-parameters
        self.c_dim = len(data_loaders['train'].dataset.class_names)
        self.c2_dim = config.c2_dim
        self.image_size = config.image_size
        self.g_conv_dim = config.g_conv_dim
        self.d_conv_dim = config.d_conv_dim
        self.g_repeat_num = config.g_repeat_num
        self.d_repeat_num = config.d_repeat_num
        self.d_train_repeat = config.d_train_repeat

        # Hyper-parameteres
        self.lambda_cls = config.lambda_cls
        self.lambda_rec = config.lambda_rec
        self.lambda_gp = config.lambda_gp
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2

        # Training settings
        self.num_epochs = config.num_epochs
        self.num_epochs_decay = config.num_epochs_decay
        self.num_iters = config.num_iters
        self.num_iters_decay = config.num_iters_decay
        self.batch_size = config.batch_size
        self.use_tensorboard = config.use_tensorboard
        self.pretrained_model = config.pretrained_model
        self.pretrained_model_path = config.pretrained_model_path

        # Test settings
        self.test_model = config.test_model

        # Path
        self.log_path = os.path.join(config.output_path, 'logs') 
        self.sample_path = os.path.join(config.output_path, 'samples') 
        self.model_save_path = os.path.join(config.output_path, 'models') 
        self.result_path = os.path.join(config.output_path, 'results') 

        # Step size
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step

        self.num_val_imgs = config.num_val_imgs

        # Build tensorboard if use
        self.build_model()
        if self.use_tensorboard:
            self.build_tensorboard()

        # Start with trained model
        if self.pretrained_model:
            self.load_pretrained_model()

    def build_model(self):

        self.G = Generator(self.g_conv_dim, self.c_dim, self.g_repeat_num, self.image_size)
        self.D = Discriminator(self.image_size, self.d_conv_dim, self.c_dim, self.d_repeat_num)

        # Optimizers
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, [self.beta1, self.beta2])
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), self.d_lr, [self.beta1, self.beta2])

        # Print networks
        self.print_network(self.G, 'G')
        self.print_network(self.D, 'D')

        if torch.cuda.is_available():
            self.G.cuda()
            self.D.cuda()

    def print_network(self, model, name):
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print_logger.info('{} - {} - Number of parameters: {}'.format(name, model, num_params))

    def load_pretrained_model(self):
        self.G.load_state_dict(torch.load(os.path.join(
            self.pretrained_model_path, '{}_G.pth'.format(self.pretrained_model))))
        self.D.load_state_dict(torch.load(os.path.join(
            self.pretrained_model_path, '{}_D.pth'.format(self.pretrained_model))))
        print_logger.info('loaded trained models (step: {})..!'.format(self.pretrained_model))

    def build_tensorboard(self):
        from tensorboard_logger import Logger
        self.logger = Logger(self.log_path)

    def update_lr(self, g_lr, d_lr):
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = g_lr
        for param_group in self.d_optimizer.param_groups:
            param_group['lr'] = d_lr

    def reset_grad(self):
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()

    def to_var(self, x, grad=True):
        if torch.cuda.is_available():
            x = x.cuda()
        return Variable(x, requires_grad=grad)

    def denorm(self, x):
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def threshold(self, x):
        x = x.clone()
        x = (x >= 0.5).float()
        return x

    def compute_accuracy(self, x, y):
        x = F.sigmoid(x)
        predicted = self.threshold(x)
        correct = (predicted == y).float()
        accuracy = torch.mean(correct, dim=0) * 100.0
        return accuracy

    def one_hot(self, labels, dim):
        """Convert label indices to one-hot vector"""
        batch_size = labels.size(0)
        out = torch.zeros(batch_size, dim)
        out[np.arange(batch_size), labels.long()] = 1
        return out

    def make_data_labels(self, real_c):
        """Generate domain labels for dataset for debugging/testing.
        """

        y = []
        for dim in range(self.c_dim):
            t = [0] * self.c_dim
            t[dim] = 1
            y.append(torch.FloatTensor(t))

        fixed_c_list = []

        for i in range(self.c_dim):
            fixed_c = real_c.clone()
            for c in fixed_c:
                c[:self.c_dim] = y[i]

            fixed_c_list.append(self.to_var(fixed_c, grad=False))

        return fixed_c_list

    def train(self):
        """Train StarGAN within a single dataset."""

        # The number of iterations per epoch
        data_loader = self.data_loaders['train']

        iters_per_epoch = len(data_loader)

        fixed_x = []
        real_c = []

        num_fixed_imgs = self.num_val_imgs
        for i in range(num_fixed_imgs):
            images, labels = self.data_loaders['val'].dataset.__getitem__(i)
            fixed_x.append(images.unsqueeze(0))
            real_c.append(labels.unsqueeze(0))

        # Fixed inputs and target domain labels for debugging
        fixed_x = torch.cat(fixed_x, dim=0)
        fixed_x = self.to_var(fixed_x, grad=False)
        real_c = torch.cat(real_c, dim=0)

        fixed_c_list = self.make_data_labels(real_c)

        # lr cache for decaying
        g_lr = self.g_lr
        d_lr = self.d_lr

        # Start with trained model if exists
        if self.pretrained_model:
            start = int(self.pretrained_model.split('_')[0])
        else:
            start = 0

        # Start training
        start_time = time.time()
        for e in range(start, self.num_epochs):
            for i, (real_x, real_label) in enumerate(data_loader):
                
                # Generat fake labels randomly (target domain labels)
                rand_idx = torch.randperm(real_label.size(0))
                fake_label = real_label[rand_idx]

                real_c = real_label.clone()
                fake_c = fake_label.clone()

                # Convert tensor to variable
                real_x = self.to_var(real_x)
                real_c = self.to_var(real_c)           # input for the generator
                fake_c = self.to_var(fake_c)
                real_label = self.to_var(real_label)   # this is same as real_c if dataset == 'CelebA'
                fake_label = self.to_var(fake_label)
                
                # ================== Train D ================== #

                # Compute loss with real images
                out_src, out_cls = self.D(real_x)
                d_loss_real = - torch.mean(out_src)

                d_loss_cls = F.binary_cross_entropy_with_logits(
                    out_cls, real_label, size_average=False) / real_x.size(0)

                # Compute classification accuracy of the discriminator
                if (i+1) % (self.log_step*10) == 0:
                    accuracies = self.compute_accuracy(out_cls, real_label)
                    log = ["{}: {:.2f}".format(attr, acc) for (attr, acc) in
                           zip(data_loader.dataset.class_names, accuracies.data.cpu().numpy())]
                    print_logger.info('Discriminator Accuracy: {}'.format(log))

                # Compute loss with fake images
                fake_x = self.G(real_x, fake_c)
                fake_x = Variable(fake_x.data)
                out_src, out_cls = self.D(fake_x)
                d_loss_fake = torch.mean(out_src)

                # Backward + Optimize
                d_loss = d_loss_real + d_loss_fake + self.lambda_cls * d_loss_cls
                self.reset_grad()
                d_loss.backward()
                self.d_optimizer.step()

                # Compute gradient penalty
                alpha = torch.rand(real_x.size(0), 1, 1, 1).cuda().expand_as(real_x)
                interpolated = Variable(alpha * real_x.data + (1 - alpha) * fake_x.data, requires_grad=True)
                out, out_cls = self.D(interpolated)

                grad = torch.autograd.grad(outputs=out,
                                           inputs=interpolated,
                                           grad_outputs=torch.ones(out.size()).cuda(),
                                           retain_graph=True,
                                           create_graph=True,
                                           only_inputs=True)[0]

                grad = grad.view(grad.size(0), -1)
                grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
                d_loss_gp = torch.mean((grad_l2norm - 1)**2)

                # Backward + Optimize
                d_loss = self.lambda_gp * d_loss_gp
                self.reset_grad()
                d_loss.backward()
                self.d_optimizer.step()

                # Logging
                loss = {}
                loss['D/loss_real'] = d_loss_real.data.item()
                loss['D/loss_fake'] = d_loss_fake.data.item()
                loss['D/loss_cls'] = d_loss_cls.data.item()
                loss['D/loss_gp'] = d_loss_gp.data.item()

                # ================== Train G ================== #
                if (i+1) % self.d_train_repeat == 0:

                    # Original-to-target and target-to-original domain
                    fake_x = self.G(real_x, fake_c)
                    rec_x = self.G(fake_x, real_c)

                    # Compute losses
                    out_src, out_cls = self.D(fake_x)
                    g_loss_fake = - torch.mean(out_src)
                    g_loss_rec = torch.mean(torch.abs(real_x - rec_x))

                    g_loss_cls = F.binary_cross_entropy_with_logits(
                        out_cls, fake_label, size_average=False) / fake_x.size(0)

                    # Backward + Optimize
                    g_loss = g_loss_fake + self.lambda_rec * g_loss_rec + self.lambda_cls * g_loss_cls
                    self.reset_grad()
                    g_loss.backward()
                    self.g_optimizer.step()

                    # Logging
                    loss['G/loss_fake'] = g_loss_fake.data.item()
                    loss['G/loss_rec'] = g_loss_rec.data.item()
                    loss['G/loss_cls'] = g_loss_cls.data.item()

                # Print out log info
                if (i+1) % self.log_step == 0:
                    elapsed = time.time() - start_time
                    elapsed = str(datetime.timedelta(seconds=elapsed))

                    log = "Elapsed [{}], Epoch [{}/{}], Iter [{}/{}]".format(
                        elapsed, e+1, self.num_epochs, i+1, iters_per_epoch)

                    for tag, value in loss.items():
                        log += ", {}: {:.4f}".format(tag, value)
                    print_logger.info(log)

                    if self.use_tensorboard:
                        for tag, value in loss.items():
                            self.logger.scalar_summary(tag, value, e * iters_per_epoch + i + 1)

                # Translate fixed images for debugging
                if (i+1) % self.sample_step == 0:
                    fake_image_list = [fixed_x]

                    for fixed_c in fixed_c_list:
                        gen_imgs = self.G(fixed_x, fixed_c)
                        fake_image_list.append(gen_imgs)

                    # fake_images = torch.cat(fake_image_list, dim=3)
                    # save_image(self.denorm(fake_images.data.cpu()),
                    #     os.path.join(self.sample_path, '{}_{}_fake.png'.format(e+1, i+1)),nrow=1, padding=0)
                    print_logger.info('Translated images and saved into {}..!'.format(self.sample_path))

                    if self.use_tensorboard:
                        tb_imgs = [t.unsqueeze(0) for t in fake_image_list]
                        tb_imgs = torch.cat(tb_imgs)
                        tb_imgs = tb_imgs.permute(1, 0, 2, 3, 4)
                        tb_imgs_list = torch.unbind(tb_imgs, dim=0)
                        tb_imgs_list = [torch.cat(torch.unbind(t, dim=0), dim=2) for t in tb_imgs_list]

                        self.logger.image_summary('fixed_imgs', tb_imgs_list, e * iters_per_epoch + i + 1)

                # Save model checkpoints
                if (i+1) % self.model_save_step == 0:
                    torch.save(self.G.state_dict(),
                        os.path.join(self.model_save_path, '{}_{}_G.pth'.format(e+1, i+1)))
                    torch.save(self.D.state_dict(),
                        os.path.join(self.model_save_path, '{}_{}_D.pth'.format(e+1, i+1)))

            # Decay learning rate
            if (e+1) > (self.num_epochs - self.num_epochs_decay):
                g_lr -= (self.g_lr / float(self.num_epochs_decay))
                d_lr -= (self.d_lr / float(self.num_epochs_decay))
                self.update_lr(g_lr, d_lr)
                print_logger.info('Decay learning rate to g_lr: {}, d_lr: {}.'.format(g_lr, d_lr))

    def test(self):
        """Facial attribute transfer on CelebA or facial expression synthesis on RaFD."""
        # Load trained parameters
        G_path = os.path.join(self.model_save_path, '{}_G.pth'.format(self.test_model))
        self.G.load_state_dict(torch.load(G_path))
        self.G.eval()

        data_loader = self.data_loaders['test']

        for i, (real_x, org_c) in enumerate(data_loader):
            real_x = self.to_var(real_x, grad=False)
            target_c_list = self.make_data_labels(org_c)

            # Start translations
            fake_image_list = [real_x]
            for target_c in target_c_list:
                fake_image_list.append(self.G(real_x, target_c))
            fake_images = torch.cat(fake_image_list, dim=3)
            save_path = os.path.join(self.result_path, '{}_fake.png'.format(i+1))
            save_image(self.denorm(fake_images.data), save_path, nrow=1, padding=0)
            print_logger.info('Translated test images and saved into "{}"..!'.format(save_path))
