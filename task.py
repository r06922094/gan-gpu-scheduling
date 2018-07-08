import example.interface


class task_t:
    # Description of tasks
    # f : feedforward @ Generator
    # b : backforward @ Generator
    # fbnu : feedforward + backforward (no-update) @ Discriminator
    # fbu : feedforward + backforward @ Discriminator
    def __init__(self, generator_index, task_type, nn_network, nn_input_1, nn_input_2, device_id):
        self.device_id = device_id
        self.generator_index = generator_index
        self.task_type = task_type
        self.nn_network = nn_network
        self.nn_input_1 = nn_input_1
        self.nn_input_2 = nn_input_2

        self.batch_size = 32
        self.optimizer = torch.optim.Adam(self.nn_network.parameters(), lr=0.0002, betas=(0.5, 0.999))

    def learn(self):

        # G task
        if self.task_type == 'f':
            self.nn_input_1 = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))), requires_grad=True).cuda(self.device_id)

            gen_imgs = G_f(self.optimizer, z, self.nn_network, self.batch_size, self.device_id)
            d_input = Variable(gen_imgs, requires_grad = True)

            # input_noise, input_fake_img
            return self.nn_input_1, d_input

        elif self.task_type == 'b':
            self.nn_input_1 = self.nn_input_1.cuda(self.device_id)
            self.nn_input_2 = self.nn_input_2.cuda(self.device_id)

            G_b(self.optimizer, self.nn_input_1, self.nn_input_2, self.nn_network, self.batch_size, self.device_id)

        # D task
        elif self.task_type == 'fbnu':
            self.nn_input_1 = self.nn_input_1.cuda(self.device_id)
            
            # self.nn_graph = discriminator
            gen_grad = D_upG_fb(self.nn_input_1, self.nn_network, self.batch_size, self.device_id)
            g_grad_send = Variable(gen_grad, requires_grad = True)
            # need to return z
            return g_grad_send, self.nn_input_2

        elif self.task_type == 'fbu':
            self.nn_input_1 = self.nn_input_1.cuda(self.device_id)
            self.nn_input_2 = self.nn_input_2.cuda(self.device_id)

            D_upD_fb(self.optimizer, self.nn_input_1, self.nn_input_2, self.nn_network, self.batch_size, self.device_id)

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
