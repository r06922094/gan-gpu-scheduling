import task
import time
import threading
import simulate

NUM_OF_ITERATION = 10

class runDiscriminator(threading.Thread):
    def __init__(self, G_upG_tskq, G_upD_tskq, D_upG_tskq, D_upD_tskq, G_upG_lock, G_upD_lock, D_upG_lock, D_upD_lock, D_internal_lock, isUpdated, whoUpdate, isUpdating, sharedWeight, index, img):
        super(runDiscriminator, self).__init__()
        self.G_upG_tskq = G_upG_tskq
        self.G_upD_tskq = G_upD_tskq
        self.D_upG_tskq = D_upG_tskq
        self.D_upD_tskq = D_upD_tskq
        self.G_upG_lock = G_upG_lock
        self.G_upD_lock = G_upD_lock
        self.D_upG_lock = D_upG_lock
        self.D_upD_lock = D_upD_lock
        # Manage multi-thread of Discrimminator
        self.D_internal_lock = D_internal_lock
        # boolean variable
        self.isUpdated = isUpdated
        self.whoUpdate = whoUpdate
        # boolean variable
        self.isUpdating = isUpdating
        self.sharedWeight = sharedWeight
        # Counter of updateG and updateD, respectively
        self.history = [0, 0]
        self.index = index
        self.img = img

    def arbitrator(self):
        while True:
            if self.D_upG_tskq.num() > 0 or self.D_upD_tskq.num() > 0:
                break
        if self.D_upG_tskq.num() == 0:
            self.history[1] += 1
            return 1
        if self.D_upD_tskq.num() == 0:
            self.history[0] += 1
            return 0
        if self.history[0] > self.history[1]:
            self.history[1] += 1
            return 1
        else:
            self.history[0] += 1
            return 0

    def run(self):
        while True:
            self.D_internal_lock.acquire()
            if self.isUpdated[0] == True and self.whoUpdate[0] != self.index:
                # sync up the weights
                print("%6.8f / Discriminator %d Pull Shared Weights / start" % (time.time(), self.index))
                time.sleep(simulate.transfer)
                print("%6.8f / Discriminator %d Pull Shared Weights / end" % (time.time(), self.index))
                self.isUpdated[0] = False
                self.D_internal_lock.release()
                continue
            elif self.isUpdating[0] == True:
                self.D_internal_lock.release()
                if self.D_upG_tskq.num() > 0:
                    self.D_upG_lock.acquire()
                    old_task = self.D_upG_tskq.dequeue()
                    self.D_upG_lock.release()
                    gindex = old_task.generator_index
                    nn_netowrk = None
                    nn_input = None
                    print("%6.8f / Generator %d / updateGenerator @ Discriminator %d / feedforward + backforward / start" % (time.time(), gindex, self.index))
                    # Do the task
                    #############
                    g_grad_send, input_noise = old_task.learn()
                    #time.sleep(simulate.fbnu_D)
                    #############
                    print("%6.8f / Generator %d / updateGenerator @ Discriminator %d / feedforward + backforward / end" % (time.time(), gindex, self.index))
                    new_task = task.task_t(gindex, 'b', old_task.gen_network, old_task.dis_network, old_task.optimizer, input_noise, g_grad_send, 0)
                    self.G_upG_lock.acquire()
                    self.G_upG_tskq.enqueue(new_task)
                    self.G_upG_lock.release()
                continue
            self.D_internal_lock.release()
            doWhat = self.arbitrator()
            # Update the Generator
            if doWhat == 0:
                self.D_upG_lock.acquire()
                old_task = self.D_upG_tskq.dequeue()
                self.D_upG_lock.release()
                gindex = old_task.generator_index
                nn_network = None
                nn_input = None
                print("%6.8f / Generator %d / updateGenerator @ Discriminator %d / feedforward + backforward / start" % (time.time(), gindex, self.index))
                # Do the task
                #############
                g_grad_send, input_noise = old_task.learn()
                #############
                print("%6.8f / Generator %d / updateGenerator @ Discriminator %d / feedforward + backforward / end" % (time.time(), gindex, self.index))
                # generator_index, task_type, gen_network, dis_network, optimizer, nn_input_1, nn_input_2, device_id
                new_task = task.task_t(gindex, 'b', old_task.gen_network, old_task.dis_network, old_task.optimizer, input_noise, g_grad_send, 0)
                self.G_upG_lock.acquire()
                self.G_upG_tskq.enqueue(new_task)
                self.G_upG_lock.release()
            # Update the Discrimminator
            else: # Update Discrimminator
                self.D_upD_lock.acquire()
                old_task = self.D_upD_tskq.dequeue()
                self.D_upD_lock.release()
                gindex = old_task.generator_index
                nn_network = None
                nn_input = None

                ###################################
                self.D_internal_lock.acquire()
                self.isUpdating[0] = True
                self.D_internal_lock.release()
                # Do the task (Updating weights...)
                print("%6.8f / Generator %d / updateDiscriminator @ Discriminator %d / feedforward + backforward / start" % (time.time(), gindex, self.index))
                ############
                # need to load real img to nn_input_2
                old_task.nn_input_2 = self.img.__iter__().__next__()[0]
                old_task.learn()
                #time.sleep(simulate.fbu_D)
                ############
                print("%6.8f / Generator %d / updateDiscriminator @ Discriminator %d / feedforward + backforward / end" % (time.time(), gindex, self.index))

                self.D_internal_lock.acquire()
                self.isUpdating[0] = False
                # Update the shared Weights
                print("%6.8f / Discriminator %d Push Shared Weights / start" % (time.time(), self.index))
                time.sleep(simulate.transfer)
                print("%6.8f / Discriminator %d Push Shared Weights / end" % (time.time(), self.index))
                self.isUpdated[0] = True
                self.whoUpdate[0] = self.index
                self.D_internal_lock.release()
                ###################################

                new_task = task.task_t(gindex, 'f', old_task.gen_network, old_task.dis_network, old_task.optimizer, None, None, 0)
                self.G_upD_lock.acquire()
                self.G_upD_tskq.enqueue(new_task)
                self.G_upD_lock.release()

class runGenerator(threading.Thread):
    def __init__(self, G_upG_tskq, G_upD_tskq, D_upG_tskq, D_upD_tskq, G_upG_lock, G_upD_lock, D_upG_lock, D_upD_lock, index):
        super(runGenerator, self).__init__()
        self.G_upG_tskq = G_upG_tskq
        self.G_upD_tskq = G_upD_tskq
        self.D_upG_tskq = D_upG_tskq
        self.D_upD_tskq = D_upD_tskq
        self.G_upG_lock = G_upG_lock
        self.G_upD_lock = G_upD_lock
        self.D_upG_lock = D_upG_lock
        self.D_upD_lock = D_upD_lock
        # Counter of updateG and updateD, respectively
        self.history = [0, 0]
        self.index = index

    def arbitrator(self):
        while True:
            if self.G_upG_tskq.num_by_index(self.index) > 0 or self.G_upD_tskq.num_by_index(self.index) > 0:
                break
        if self.G_upG_tskq.num_by_index(self.index) == 0:
            self.history[1] += 1
            return 1
        if self.G_upD_tskq.num_by_index(self.index) == 0:
            self.history[0] += 1
            return 0
        if self.history[0] > self.history[1]:
            self.history[1] += 1
            return 1
        else:
            self.history[0] += 1
            return 0
        
    def run(self):
        while True:
            doWhat = self.arbitrator()
            # Update the Generator
            if doWhat == 0:
                self.G_upG_lock.acquire()
                old_task = self.G_upG_tskq.dequeue_by_index(self.index)
                self.G_upG_lock.release()
                gindex = old_task.generator_index
                nn_network = None
                nn_input = None
                if old_task.task_type is 'b':
                    print("%6.8f / Generator %d / updateGenerator @ Generator %d / backforward / start" % (time.time(), gindex, self.index))
                    # Do the task
                    #############
                    old_task.learn()
                    #############
                    print("%6.8f / Generator %d / updateGenerator @ Generator %d / backforward / end" % (time.time(), gindex, self.index))
                    new_task = task.task_t(gindex, 'f', old_task.gen_network, old_task.dis_network, old_task.optimizer, None, None, 0)
                    self.G_upG_lock.acquire()
                    self.G_upG_tskq.enqueue(new_task)
                    self.G_upG_lock.release()
                elif old_task.task_type is 'f':
                    print("%6.8f / Generator %d / updateGenerator @ Generator %d / feedforward / start" % (time.time(), gindex, self.index))
                    # Do the task
                    #############
                    input_noise, input_fake_img = old_task.learn()       
                    #############
                    print("%6.8f / Generator %d / updateGenerator @ Generator %d / feedforward / end" % (time.time(), gindex, self.index))
                    # generator_index, task_type, gen_network, dis_network, optimizer, nn_input_1, nn_input_2, device_id

                    new_task = task.task_t(gindex, 'fbnu', old_task.gen_network, old_task.dis_network, old_task.optimizer, input_fake_img, input_noise, 1)

                    self.D_upG_lock.acquire()
                    self.D_upG_tskq.enqueue(new_task)
                    self.D_upG_lock.release()
            # Update the Discrimminator
            else:
                self.G_upD_lock.acquire()
                old_task = self.G_upD_tskq.dequeue_by_index(self.index)
                self.G_upD_lock.release()
                gindex = old_task.generator_index
                nn_network = None
                nn_input = None
                if old_task.task_type is 'f':
                    print("%6.8f / Generator %d / updateDiscriminator @ Genrator %d / feedforward / start" % (time.time(), gindex, self.index))
                    # Do the task
                    #############
                    _, input_fake_img = old_task.learn()
                    #time.sleep(simulate.f_G)
                    #############
                    print("%6.8f / Generator %d / updateDiscriminator @ Genrator %d / feedforward / end" % (time.time(), gindex, self.index))
                    # disc need to load real img
                    new_task = task.task_t(gindex, 'fbu', old_task.gen_network, old_task.dis_network, old_task.optimizer, input_fake_img, None, 1)
                    self.D_upD_lock.acquire()
                    self.D_upD_tskq.enqueue(new_task)
                    self.D_upD_lock.release()
