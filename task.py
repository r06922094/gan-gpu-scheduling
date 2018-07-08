import example.interface
import torch
from torch.autograd import Variable
import numpy as np

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

class task_t:
    # Description of tasks
    # f : feedforward @ Generator
    # b : backforward @ Generator
    # fbnu : feedforward + backforward (no-update) @ Discriminator
    # fbu : feedforward + backforward @ Discriminator
    def __init__(self, generator_index, task_type, gen_network, dis_network, optimizer, nn_input_1, nn_input_2, device_id):
        self.generator_index = generator_index
        self.task_type = task_type
        self.gen_network = gen_network
        self.dis_network = dis_network
        self.nn_input_1 = nn_input_1
        self.nn_input_2 = nn_input_2

        self.batch_size = 32
        self.optimizer = optimizer
        self.device_id = device_id



    def learn(self):

        # G task
        if self.task_type == 'f':
            self.nn_input_1 = Variable(Tensor(np.random.normal(0, 1, (self.batch_size, 100))), requires_grad=True).cuda(self.device_id)
            gen_imgs = example.interface.G_f(self.optimizer, self.nn_input_1, self.gen_network, self.batch_size, self.device_id)
            d_input = Variable(gen_imgs, requires_grad = True)
            # input_noise, input_fake_img
            return self.nn_input_1, d_input

        elif self.task_type == 'b':
            self.nn_input_1 = self.nn_input_1.cuda(self.device_id)
            self.nn_input_2 = self.nn_input_2.cuda(self.device_id)
            example.interface.G_b(self.optimizer, self.nn_input_1, self.nn_input_2, self.gen_network, self.batch_size, self.device_id)

        # D task
        elif self.task_type == 'fbnu':
            gen_grad = example.interface.D_upG_fb(self.nn_input_1, self.dis_network, self.batch_size, self.device_id)
            g_grad_send = Variable(gen_grad, requires_grad = True)
            # need to return input_noise
            return g_grad_send, self.nn_input_2

        elif self.task_type == 'fbu':
            self.nn_input_1 = self.nn_input_1.cuda(self.device_id)
            self.nn_input_2 = self.nn_input_2.cuda(self.device_id)
            example.interface.D_upD_fb(self.optimizer, self.nn_input_1, self.nn_input_2, self.dis_network, self.batch_size, self.device_id)

class queue_t:
    def __init__(self):
        self.impl = []

    def num(self):
        return len(self.impl)

    def enqueue(self, task):
        self.impl.append(task)

    def dequeue(self):
        if len(self.impl) > 0:
            return self.impl.pop(0)
        return None

    def num_by_index(self, generator_index):
        count = 0
        for tsk in self.impl:
            if tsk.generator_index == generator_index:
                count += 1
        return count

    def dequeue_by_index(self, generator_index):
        for i in range(len(self.impl)):
            if self.impl[i].generator_index == generator_index:
                return self.impl.pop(i)
        return None
