import os

import neat                        # pip install neat-python
import tensorflow as tf

##### SETTINGS #####
MULTITHREADING: bool = True          # *can* increase speed but will use all cores by default!
NUM_GENERATIONS: int = 300           # maximum number of generations to run, if target fitness is not reached before
NUM_SAMPLES: int = 2000              # how many images to test each network on
RESAMPLE: bool = False               # whether to resample NUM_SAMPLES items from the dataset every generation
CONFIG_FILEPATH: str = 'config.cfg'  # relative to cwd
####################
# input nodes: -num_input -> -1
# output nodes: 0 -> num_output-1
# hidden nodes: num_output -> num_hidden+num_outputs-1

config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         os.path.join(os.path.dirname(__file__), CONFIG_FILEPATH))

genome = neat.DefaultGenome(69)
genome.configure_new(config.genome_config)
genome.connect_full_nodirect(config.genome_config)
print(genome.size())  # (nodes, connections)

cons = genome.connections


mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0



inp_org = tf.keras.layers.Input(shape=(28, 28))
input_layer = tf.keras.layers.Flatten()(inp_org)

nodes = {}

depths = {}

outs = {}

prev_layer = None
for con in genome.connections.values():
    origin = con.key[0]
    destination = con.key[1]
    weight = con.weight
    depth = depths.get(origin, 0)

    inp = tf.keras.layers.Lambda(lambda x: x[abs(origin)-1, :])(input_layer)
    inp = tf.keras.layers.Flatten()(inp)
    layer = tf.keras.layers.Dense(1, activation='sigmoid')(nodes.get(origin, inp))
    prev_layer = layer
    out_list = outs.get(depth, [])
    out_list.append(layer)
    outs[depth] = out_list
    nodes[destination] = layer

    if origin in config.genome_config.input_keys:
        depths[destination] = depth + 1
    elif destination in config.genome_config.output_keys:
        depths[destination] = depth + 1
    else:  # hidden -> hidden
        depths[destination] = depth + 1


out_list = []
for k, v in outs.items():
    out_list.extend(v)

out_list.append(tf.keras.layers.Dense(10)(out_list[-1]))
model = tf.keras.Model(inputs=inp_org, outputs=tf.keras.layers.concatenate(out_list))
print(model(x_train[:1].reshape(28, 28)))
print(model.summary())