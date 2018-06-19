import task
import session
import threading

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

NUM_OF_GENERATOR = 2

for i in range(NUM_OF_GENERATOR):
    upG_task = task.task_t(i, 'f', None, None)
    G_upG_tskq.enqueue(upG_task)
    upD_task = task.task_t(i, 'f', None, None)
    G_upD_tskq.enqueue(upD_task)

g1 = session.runGenerator(G_upG_tskq, G_upD_tskq, D_upG_tskq, D_upD_tskq, G_upG_lock, G_upD_lock, D_upG_lock, D_upD_lock, 0)

g2 = session.runGenerator(G_upG_tskq, G_upD_tskq, D_upG_tskq, D_upD_tskq, G_upG_lock, G_upD_lock, D_upG_lock, D_upD_lock, 1)

d1 = session.runDiscriminator(G_upG_tskq, G_upD_tskq, D_upG_tskq, D_upD_tskq, G_upG_lock, G_upD_lock, D_upG_lock, D_upD_lock, D_internal_lock, isUpdated, whoUpdate, isUpdating, sharedWeight, 0)

d2 = session.runDiscriminator(G_upG_tskq, G_upD_tskq, D_upG_tskq, D_upD_tskq, G_upG_lock, G_upD_lock, D_upG_lock, D_upD_lock, D_internal_lock, isUpdated, whoUpdate, isUpdating, sharedWeight, 1)

g1.start()
g2.start()
d1.start()
d2.start()

g1.join()
g2.join()
d1.join()
d2.join()
