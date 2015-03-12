To run the program:

The file main.py contains assignment 1 and 2. Just run "python main.py" to see the output of sum-product. 
For running max-sum it is necessary to comment line 85-92 and uncomment from 94-end. We added making influenza observed as an example.

The file assignment3.py contains the code to create the Markov Random Field for the image and run Max-Sum on it.
We initialize the condition probabilites of the factor between the observed and the latent variables to the amount of noise that is added. So if the probability that noise swaps a pixel is 0.1 (so 0.9 in the example code), we set the probability of two pixels remaining the same value to 0.9. As for the conditionals of the factors between latent variables, a good estimate for the conditional probability in the original image is the value in the noisy image.

The file Node contains all the parts of the algorithm, implemented as described in the assignment.

Answer to 1.8:
We now know the (marginal) a posteriori likelihood of each value of each node individually, 
but as the variables are not independent we do not know which combination of values is the MAP state.