import os
import time
from functools import partial
import numpy as np
import pickle

import neat                                 # pip install neat-python
import mnist                                # pip install mnist
from sklearn.utils import shuffle           # pip install scikit-learn
import matplotlib
import matplotlib.pyplot as plt             # pip install matplotlib
import skimage as ski                       # pip install scikit-image
from scipy.ndimage import affine_transform  # pip install scipy

##### SETTINGS #####
MULTITHREADING: bool = True          # *can* increase speed but will use all cores by default!
NUM_GENERATIONS: int = 2000          # maximum number of generations to run, if target fitness is not reached before
NUM_SAMPLES: int = 2000              # how many images to test each network on
RESAMPLE: bool = False               # whether to resample NUM_SAMPLES items from the dataset every generation
CONFIG_FILEPATH: str = 'config.cfg'  # relative to cwd
PROCESS: bool = True                 # make the images 16x16 and deskew
####################


def eval_genomes(genomes, config):
    if RESAMPLE:
        prep_dataset()
    for genome_id, genome in genomes:
        eval_genome(genome, config)


def eval_genomes_async(genomes, config):
    if RESAMPLE:
        prep_dataset()
    evaluator.evaluate(genomes, config)


def eval_genome(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    correct = 0
    for xi, yi in xy:
        output = net.activate(xi)
        if max(output) == min(output):
            correct = 0
            break
        prediction = output.index(max(output))
        correct += 1 if prediction == yi else 0
    genome.fitness = correct / total


def eval_genome_async(genome, config, xy=None, total=None):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    correct = 0
    for xi, yi in xy:
        output = net.activate(xi)
        if max(output) == min(output):
            correct = 0
            break
        prediction = output.index(max(output))
        correct += 1 if prediction == yi else 0
    return correct / total


def prep_dataset():
    def moments(image):
        c0, c1 = np.mgrid[:image.shape[0], :image.shape[1]]  # A trick in numPy to create a mesh grid
        totalImage = np.sum(image)  # sum of pixels
        m0 = np.sum(c0 * image) / totalImage  # mu_x
        m1 = np.sum(c1 * image) / totalImage  # mu_y
        m00 = np.sum((c0 - m0) ** 2 * image) / totalImage  # var(x)
        m11 = np.sum((c1 - m1) ** 2 * image) / totalImage  # var(y)
        m01 = np.sum((c0 - m0) * (c1 - m1) * image) / totalImage  # covariance(x,y)
        mu_vector = np.array([m0, m1])  # Notice that these are \mu_x, \mu_y respectively
        covariance_matrix = np.array([[m00, m01], [m01, m11]])  # Do you see a similarity between the covariance matrix
        return mu_vector, covariance_matrix

    def deskew(image):
        c, v = moments(image)
        alpha = v[0, 1] / v[0, 0]
        affine = np.array([[1, 0], [alpha, 1]])
        ocenter = np.array(image.shape) / 2.0
        offset = c - np.dot(affine, ocenter)
        img = affine_transform(image, affine, offset=offset)
        return (img - img.min(initial=0)) / (img.max(initial=1) - img.min(initial=0))

    global xy, total
    x, y = shuffle(mnist.test_images(), mnist.test_labels(), n_samples=NUM_SAMPLES)
    total = len(y)
    xy = []
    for xi, yi in zip(x, y):
        if PROCESS:
            xi = ski.transform.resize(deskew(xi), (16, 16), preserve_range=True, anti_aliasing=False)
        xy.append((xi.flatten(), yi))


def plot_stats(statistics, ylog=False, view=False, filename='avg_fitness.svg'):
    """ Plots the population's average and best fitness. """
    matplotlib.use('TkAgg')  # temp fix for Pycharm
    generation = list(range(len(statistics.most_fit_genomes)))
    best_fitness = [c.fitness for c in statistics.most_fit_genomes]
    avg_fitness = np.array(statistics.get_fitness_mean())
    stdev_fitness = np.array(statistics.get_fitness_stdev())

    plt.plot(generation, avg_fitness, 'b-', label="average")
    plt.plot(generation, avg_fitness - stdev_fitness, 'g-.', label="-1 sd")
    plt.plot(generation, avg_fitness + stdev_fitness, 'g-.', label="+1 sd")
    plt.plot(generation, best_fitness, 'r-', label="best")

    plt.title("Population's average and best fitness")
    plt.xlabel("Generations")
    plt.ylabel("Fitness")
    plt.grid()
    plt.legend(loc="best")
    if ylog:
        plt.gca().set_yscale('symlog')

    plt.savefig(filename)
    if view:
        plt.show()

    plt.close()


def plot_mnist(x, y, pred=None, cols=5):
    matplotlib.use('TkAgg')  # temp fix for Pycharm
    rows = round((len(y)/cols)+0.5)
    fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
    for i, ax in enumerate(fig.axes):
        print(x[i].shape)
        ax.imshow(x[i], cmap='gray_r')
        ax.set_title(f'Pred: {pred[i] if pred else "n/a"} - True: {y[i]}')
        # ax.set_axis_off()
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    plt.show()


class Classifier:
    def __init__(self, genome, config):
        self.net = neat.nn.FeedForwardNetwork.create(genome, config)

    def __call__(self, inputs):
        if isinstance(inputs, list):
            res = []
            for array in inputs:
                output = self.net.activate(array)
                res.append(output.index(max(output)))
            return res
        else:
            output = self.net.activate(inputs)
            return output.index(max(output))

    @classmethod
    def from_checkpoint(cls, filename, config):
        genome = neat.Checkpointer().restore_checkpoint(filename).best_genome
        if genome:
            return cls(genome, config)
        else:
            print(f'WARNING: checkpoint "{filename}" does not have a best_genome, picking random one')
            genome = list(neat.Checkpointer().restore_checkpoint(filename).population.values())[0]
            return cls(genome, config)


def run():
    global evaluator
    prep_dataset()

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         os.path.join(os.path.dirname(__file__), CONFIG_FILEPATH))

    p = neat.Population(config)

    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    timestamp = int(time.time())
    os.makedirs(f'checkpoints/{timestamp}', exist_ok=True)
    checkpointer = neat.Checkpointer(generation_interval=10, filename_prefix=f'checkpoints/{timestamp}/neat_pop_')

    p.add_reporter(checkpointer)

    if MULTITHREADING:
        print(f'### Using {os.cpu_count()} cpu cores for multithreading')
        evaluator = neat.parallel.ParallelEvaluator(num_workers=os.cpu_count(),
                                                    eval_function=partial(eval_genome_async, xy=xy, total=total))
        winner = p.run(eval_genomes_async, NUM_GENERATIONS)
    else:
        winner = p.run(eval_genomes, NUM_GENERATIONS)

    checkpointer.save_checkpoint(config, p, neat.DefaultSpeciesSet, p.generation)

    with open('_winner.pkl', 'wb') as file:
        pickle.dump(winner, file)
    with open('_stats.pkl', 'wb') as file:
        pickle.dump(stats, file)

    plot_stats(stats)

    print('\nBest genome:\n{!s}'.format(winner))

    # Show output of the most fit genome against test data.
    winner_clf = Classifier(winner, config)
    x = []
    y = []
    for xi, yi in xy[:20]:
        x.append(xi.reshape(28, 28))
        y.append(yi)
        output = winner_clf(xi)
        print("expected output {!r}, got {!r}".format(yi, output))
    # plot_mnist(x, y, winner_clf(x))


if __name__ == '__main__':
    run()
