import os
import time
from functools import partial

import neat                        # pip install neat-python
import mnist                       # pip install mnist
from sklearn.utils import shuffle  # pip install scikit-learn
import matplotlib
import matplotlib.pyplot as plt    # pip install matplotlib

##### SETTINGS #####
MULTITHREADING: bool = True          # *can* increase speed but will use all cores by default!
NUM_GENERATIONS: int = 4096          # maximum number of generations to run, if target fitness is not reached before
NUM_SAMPLES: int = 2000              # how many images to test each network on
RESAMPLE: bool = False               # whether to resample NUM_SAMPLES items from the dataset every generation
CONFIG_FILEPATH: str = 'config.cfg'  # relative to cwd
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
    global xy, total
    x, y = shuffle(mnist.test_images(), mnist.test_labels(), n_samples=NUM_SAMPLES)
    total = len(y)
    xy = []
    for xi, yi in zip(x, y):
        xy.append((xi.flatten(), yi))


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
    plot_mnist(x, y, winner_clf(x))


if __name__ == '__main__':
    run()
