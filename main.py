import task
import session
import threading
import os

import example.model

import torch



print('create model')

dataloader = example.model.load_data()

generator = example.model.Generator()
discriminator = example.model.Discriminator()

generator.apply(example.model.weights_init_normal)
discriminator.apply(example.model.weights_init_normal)

optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

print('send to device')
generator.cuda(0)
discriminator.cuda(1)

print('queue init')

G_upG_tskq = task.queue_t()
G_upD_tskq = task.queue_t()
D_upD_tskq = task.queue_t()
D_upG_tskq = task.queue_t()

G_upG_lock = threading.Lock()
G_upD_lock = threading.Lock()
D_upG_lock = threading.Lock()
D_upD_lock = threading.Lock()
D_internal_lock = threading.Lock()

isUpdated = [False]
whoUpdate = [-1]
isUpdating = [False]
sharedWeight = None

NUM_OF_GENERATOR = 1
NUM_OF_DISCRIMINATOR = 1

for i in range(NUM_OF_GENERATOR):
    # generator_index, task_type, gen_network, dis_network, optimizer, nn_input_1, nn_input_2, device_id
    upG_task = task.task_t(i, 'f', generator, discriminator, optimizer_G, None, None, 0)
    G_upG_tskq.enqueue(upG_task)
    upD_task = task.task_t(i, 'f', generator, discriminator, optimizer_D, None, None, 0)
    G_upD_tskq.enqueue(upD_task)

glist = []
dlist = []

for i in range(NUM_OF_GENERATOR):
    g = session.runGenerator(G_upG_tskq, G_upD_tskq, D_upG_tskq, D_upD_tskq, G_upG_lock, G_upD_lock, D_upG_lock, D_upD_lock, i)
    g.start()
    glist.append(g)

for i in range(NUM_OF_DISCRIMINATOR):
    d = session.runDiscriminator(G_upG_tskq, G_upD_tskq, D_upG_tskq, D_upD_tskq, G_upG_lock, G_upD_lock, D_upG_lock, D_upD_lock, D_internal_lock, isUpdated, whoUpdate, isUpdating, sharedWeight, i, dataloader)
    d.start()
    dlist.append(d)

for g in glist:
    g.join()

for d in dlist:
    d.join()
