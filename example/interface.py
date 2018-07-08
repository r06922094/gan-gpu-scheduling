import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

def G_f(optimizer, nn_input, nn_graph, batch_size, device_id):
    optimizer.zero_grad()
    #gen_imgs = generator(z)
    gen_imgs = nn_graph(nn_input)
    return gen_imgs

def G_b(optimizer, nn_input_1, nn_input_2, nn_graph, batch_size, device_id):
    #gen.forward(z).backward(g_grad_recv)
    nn_graph.forward(nn_input_1).backward(nn_input_2)
    optimizer.step()

def D_upG_fb(nn_input, nn_graph, batch_size, device_id):
    adversarial_loss = torch.nn.BCELoss()
    valid = Variable(Tensor(batch_size, 1).fill_(1.0), requires_grad=False).cuda(device_id)
    #g_loss = adversarial_loss(discriminator(d_input.cuda(1)), valid).cuda(1)
    g_loss = adversarial_loss(nn_graph(nn_input.cuda(device_id)), valid).cuda(device_id)
    g_loss.backward()

    return nn_input.grad.data

def D_upD_fb(optimizer, gen_input, real_input, nn_graph, batch_size, device_id):
    adversarial_loss = torch.nn.BCELoss()
    valid = Variable(Tensor(batch_size, 1).fill_(1.0), requires_grad=False).cuda(device_id)
    fake  = Variable(Tensor(batch_size, 1).fill_(0.0), requires_grad=False).cuda(device_id)

    real_loss = adversarial_loss(nn_graph(real_input), valid).cuda(device_id)
    fake_loss = adversarial_loss(nn_graph(gen_input ), fake ).cuda(device_id)
    d_loss = (real_loss + fake_loss) / 2

    d_loss.backward()
    optimizer.step()

