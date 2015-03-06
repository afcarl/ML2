import numpy as np
import matplotlib.pyplot as plt
import Node

import warnings
warnings.filterwarnings("ignore", category=FutureWarning) 


# Load the image and binarize
im = np.mean(plt.imread('dalmatian1.png'), axis=2) > 0.5
# plt.imshow(im)
# plt.gray()

# Add some noise
noise = np.random.rand(*im.shape) > 0.9
noise_im = np.logical_xor(noise, im)
# plt.figure()
# plt.imshow(noise_im)

test_im = np.zeros((10,10))
test_im[2:8, 2:8] = 0.0
# test_im[6,5] = 1.0
# plt.figure()
# plt.imshow(test_im)
# plt.gray()
# plt.show()

# Add some noise
noise = np.random.rand(*test_im.shape) > 0.9
noise_test_im = np.logical_xor(noise, test_im)
# plt.figure()
# plt.imshow(noise_test_im)

# W = Node.Variable('Wheezing',2)

# # and Factor nodes
# I_prior = Node.Factor('I_prior',np.array([0.95,0.05]),[I])

test_im = noise_test_im

pixel_x = test_im.shape[0]
pixel_y = test_im.shape[1]

observed_graph = [[None for i in xrange(pixel_y)] for j in xrange(pixel_x)]
latent_graph = [[None for i in xrange(pixel_y)] for j in xrange(pixel_x)]
observed_factor_graph = [[None for i in xrange(pixel_y)] for j in xrange(pixel_x)]
latent_factor_graph = dict()


#Coordinates are of shape (x,y)
init_prob = np.zeros((2,2)) + 0.5

init_prob_unequal = np.empty((2,2))

init_prob_unequal[0,0] = 0.5
init_prob_unequal[0,1] = 0.5
init_prob_unequal[1,0] = 0.5
init_prob_unequal[1,1] = 0.5

for x in xrange(pixel_x):
    for y in xrange(pixel_y):
        observed_graph[x][y] = Node.Variable("V_observed" + str(x) + str(y),2)
        observed_graph[x][y].set_observed(test_im[x,y])

        latent_graph[x][y] = Node.Variable("V_latent" + str(x) + str(y),2)

        observed_factor_graph[x][y] = Node.Factor("F_observed" + str(x) + str(y),init_prob_unequal,[observed_graph[x][y],latent_graph[x][y]])

        observed_graph[x][y].pending.update([observed_factor_graph[x][y]])


for x in xrange(pixel_x):
    for y in xrange(pixel_y):
        name_x = "F_latent" + str(x) + str(y) + "-" + str(x + 1) + str(y)
        name_y = "F_latent" + str(x) + str(y) + "-" + str(x) + str(y + 1)

        if x != (pixel_x - 1):
            latent_factor_graph[name_x] = Node.Factor(name_x,init_prob,[latent_graph[x][y],latent_graph[x + 1][y]])
        if y != (pixel_y - 1):
            latent_factor_graph[name_y] = Node.Factor(name_y,init_prob,[latent_graph[x][y],latent_graph[x][y + 1]])

for x in xrange(pixel_x):
    for y in xrange(pixel_y):
        name = lambda from_x, from_y, to_x, to_y: "F_latent" + str(from_x) + str(from_y) + "-" + str(to_x) + str(to_y)

        if x != (pixel_x - 1):
            latent_graph[x][y].in_msgs[latent_factor_graph[name(x,y,x+1,y)]] = np.array([1.,1.])

        if x != 0:
            latent_graph[x][y].in_msgs[latent_factor_graph[name(x-1,y,x,y)]] = np.array([1.,1.])

        if y != (pixel_y - 1):
            latent_graph[x][y].in_msgs[latent_factor_graph[name(x,y,x,y+1)]] = np.array([1.,1.])

        if y != 0:
            latent_graph[x][y].in_msgs[latent_factor_graph[name(x,y-1,x,y)]] = np.array([1.,1.])



for i in xrange(50):
    print "Iteration", i
    #First the observed variables

    for x in xrange(pixel_x):
        for y in xrange(pixel_y):
            node = observed_graph[x][y]

            pending = set(node.pending)
            for other in pending:
                node.send_ms_msg(other)

    #Then the obs-lat factors
    for x in xrange(pixel_x):
        for y in xrange(pixel_y):
            node = observed_factor_graph[x][y]

            pending = set(node.pending)
            for other in pending:
                node.send_ms_msg(other)


    #Then the latent variables
    for x in xrange(pixel_x):
        for y in xrange(pixel_y):
            node = latent_graph[x][y]

            pending = set(node.pending)
            for other in pending:
                node.send_ms_msg(other)


    #Then the lat-lat factors
    factor_order = latent_factor_graph.keys()
    np.random.shuffle(factor_order)
    for key in factor_order:
            node = latent_factor_graph[key]

            pending = set(node.pending)
            for other in pending:
                node.send_ms_msg(other) 



    rec_im = np.empty((pixel_x,pixel_y))

    for x in xrange(pixel_x):
        for y in xrange(pixel_y):
            node = latent_graph[x][y]
            rec_im[x,y] = node.max()

 
    print test_im.shape[0] * test_im.shape[1]
    print np.sum(test_im)
    print np.sum(rec_im)
    print "-------------------"

print test_im
print rec_im
# plt.figure()
# plt.imshow(rec_im)
# plt.gray()
# plt.show()


