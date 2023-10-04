import os
import multiprocessing as mp
from concurrent.futures.thread import ThreadPoolExecutor

import neat  # pip install neat-python
import mnist  # pip install mnist
from sklearn.utils import shuffle  # pip install scikit-learn

##### SETTINGS #####
MULTITHREADING = True           # *can* increase speed but probably uses more cpu power (will use all cores by default!)
NUM_GENERATIONS = 300           # maximum number of generations to run, if target fitness is not reached before
NUM_SAMPLES = 2000              # how many images to test each network on
CONFIG_FILEPATH = 'config.cfg'  # relative to cwd
####################

cores = mp.cpu_count()

config_path = os.path.join(os.path.dirname(__file__), CONFIG_FILEPATH)

x, y = shuffle(mnist.test_images(), mnist.test_labels(), n_samples=NUM_SAMPLES)  # TODO resample each generation?
total = len(y)
xy = []
for xii, yii in zip(x, y):
    xy.append((xii.flatten(), yii))


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        eval_genome(genome, config)


def eval_genomes_async(genomes, config):
    with ThreadPoolExecutor(max_workers=cores) as executor:
        for genome_id, genome in genomes:
            executor.submit(eval_genome, genome, config)


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


def run():
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    p = neat.Population(config)

    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    if MULTITHREADING:
        print(f'### Using {cores} cpu cores for multithreading')
        winner = p.run(eval_genomes_async, NUM_GENERATIONS)
    else:
        winner = p.run(eval_genomes, NUM_GENERATIONS)

    print('\nBest genome:\n{!s}'.format(winner))

    # TODO
    # Show output of the most fit genome against training data.
    # print('\nOutput:')
    # winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    # for xi, yi in xy:
    #     output = winner_net.activate(xi)
    #     print("expected output {!r}, got {!r}".format(xi, yi, output))


if __name__ == '__main__':
    run()
