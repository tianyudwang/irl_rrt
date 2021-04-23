import matplotlib.pyplot as plt

class LinearDecayLR:
    def __init__(self, lr, final_lr=0.1, max_steps=100):
        self.init_lr = lr
        self.final_lr = final_lr
        self.max_steps =  max_steps

    def get_lr(self, k):
        """
        Returns a linear decay learning rate
        """
        if k >= self.max_steps:
            return self.final_lr
        else:
            return (self.final_lr - self.init_lr) / self.max_steps * k + self.init_lr 

class Logger:
    def __init__(self, true_weight):
        self.weights = []
        self.true_weight = true_weight

    def log(self, *args):
        """
        Args:
            k: iteration index
            weight: current learnable weight
        """
        k, weight = args
        self.weights.append([k, weight])

    def plot(self):
        idx = [k for k, w in self.weights]
        fig, ax = plt.subplots()
        l1, = ax.plot(idx, [w[0] for k, w in self.weights], 'r.')
        l2, = ax.plot(idx, [w[1] for k, w in self.weights], 'b.')
        l3, = ax.plot(idx, [self.true_weight[0]] * len(idx), 'r-')
        l4, = ax.plot(idx, [self.true_weight[1]] * len(idx), 'b-')

        ax.legend((l1, l2, l3, l4), ('learned w1', 'learned w2', 'true w1', 'true w2'), loc='upper right')
        ax.set_xlim([0, idx[-1]])
        ax.set_ylim([0, 1])
        plt.show()

def visualize_trajs(trajs, filename):
    fig, ax = plt.subplots()
    for traj in trajs:
        ax.plot([x for (x, y) in traj], [y for (x, y) in traj], 'r--')
    plt.savefig(filename)
    #plt.show()
    plt.close()