import numpy as np

import matplotlib.pyplot as plt

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
# test_im[5:8, 3:8] = 1.0
# test_im[5,5] = 1.0
# plt.figure()
# plt.imshow(test_im)

# Add some noise
noise = np.random.rand(*test_im.shape) > 0.9
noise_test_im = np.logical_xor(noise, test_im)
# plt.figure()
# plt.imshow(noise_test_im)

# W = Node.Variable('Wheezing',2)

# # and Factor nodes
# I_prior = Node.Factor('I_prior',np.array([0.95,0.05]),[I])

pixel_x = 10
pixel_y = 5

observed_graph = [[None for i in xrange(pixel_y)] for j in xrange(pixel_x)]
latent_graph = [[None for i in xrange(pixel_y)] for j in xrange(pixel_x)]
observed_factor_graph = [[None for i in xrange(pixel_y)] for j in xrange(pixel_x)]
latent_factor_graph = [[None for i in xrange(pixel_y - 1)] for j in xrange(pixel_x - 1)]


#Coordinates are of shape (x,y)

for x in xrange(pixel_x):
	for y in xrange(pixel_y):
		observed_graph[x][y] = Node.Variable("V_observed" + str(x) + str(y),2)
		observed_graph[x][y].set_observed(test_im[x,y])

		latent_graph[x][y] = Node.Variable("V_latent" + str(x) + str(y),2)

		observed_graph[x][y].pending.update(latent_graph[x][y])

		observed_factor_graph[x][y] = Node.Factor("F_observed" + str(x) + str(y),np.array([0.5,0.5]),[observed_graph[x][y],latent_graph[x][y]])

for x in xrange(pixel_x):
	for y in xrange(pixel_y):
		if x != pixel_x:
			latent_factor_graph[x][y] = Node.Factor("F_latent" + str(x) + str(y),np.array([0.5,0.5]),[observed_graph[x][y],observed_graph[x + 1][y]])

		if y != pixel_y:
			latent_factor_graph[x][y] = Node.Factor("F_latent" + str(x) + str(y),np.array([0.5,0.5]),[observed_graph[x][y],observed_graph[x][y + 1]])


for i in xrange(5):
	#First the observed variables
    for x in xrange(pixel_x):
		for y in xrange(pixel_y):
			node = observed_graph[x][y]

	        pending = set(node.pending)
	        for other in pending:
	           	node.send_ms_msg(other) 

	#Then the factors
	 for x in xrange(pixel_x):
		for y in xrange(pixel_y):
			node = observed_factor[x][y]

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

	#Then the latent factors
	 for x in xrange(pixel_x):
		for y in xrange(pixel_y):
			node = latent_factor[x][y]

	        pending = set(node.pending)
	        for other in pending:
	           	node.send_ms_msg(other) 






