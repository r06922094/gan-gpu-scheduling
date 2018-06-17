class task_t:
    # Description of tasks
    # f : feedforward @ Generator
    # b : backforward @ Generator
    # fbnu : feedforward + backforward (no-update) @ Discriminator
    # fbu : feedforward + backforward @ Discriminator
    def __init__(self, generator_index, task_type, nn_network, nn_input):
        self.generator_index = generator_index
        self.task_type = task_type
        self.nn_network = nn_network
        self.nn_input = nn_input

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
