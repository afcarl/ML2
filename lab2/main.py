import numpy as np
import matplotlib.pyplot as plt
import Node

def sum_product(node_list):
    for node in node_list:
        pending = set(node.pending)
        for other in pending:
           	node.send_sp_msg(other)        

# Instantiate Variable nodes
I = Node.Variable('Influenza',2)
SM= Node.Variable('Smokes',2)
ST= Node.Variable('SoreThroat',2)
F = Node.Variable('Fever',2)
B = Node.Variable('Bronchitis',2)
C = Node.Variable('Coughing',2)
W = Node.Variable('Wheezing',2)

# and Factor nodes
I_prior = Node.Factor('I_prior',np.array([0.05,0.95]),[I])
SM_prior = Node.Factor('SM_prior',np.array([0.2,0.8]),[SM])

I_ST  = Node.Factor('I_ST',np.empty((2,2)),[I, ST])
I_F   = Node.Factor('I_F',np.empty((2,2)),[I,F])
SMI_B  = Node.Factor('SMI_B',np.empty((2,2,2)),[SM,I,B])
B_C  = Node.Factor('B_C',np.empty((2,2)),[B,C])
B_W  = Node.Factor('B_W',np.empty((2,2)),[B,W])

#add priors
I_ST.f[1,1] = 0.3
I_ST.f[1,0] = 0.7
I_ST.f[0,1] = 0.001
I_ST.f[0,0] = 0.999

I_F.f[1,1] = 0.9
I_F.f[1,0] = 0.1
I_F.f[0,1] = 0.05
I_F.f[0,0] = 0.95

SMI_B.f[1,1,1] = 0.99
SMI_B.f[1,1,0] = 0.01
SMI_B.f[1,0,1] = 0.9
SMI_B.f[1,0,0] = 0.1
SMI_B.f[0,1,1] = 0.7
SMI_B.f[0,1,0] = 0.3
SMI_B.f[0,0,1] = 0.0001
SMI_B.f[0,0,0] = 0.9999

B_C.f[1,1] = 0.8
B_C.f[1,0] = 0.2
B_C.f[0,1] = 0.07
B_C.f[0,0] = 0.93
B_W.f[1,1] = 0.6
B_W.f[0,1] = 0.001
B_W.f[1,0] = 0.4
B_W.f[0,0] = 0.999

#Wheezing is root!
I_prior.pending.update([I])
SM_prior.pending.update([SM])
ST.pending.update([I_ST])
F.pending.update([I_F])
C.pending.update([B_C])

node_list = [I_prior, SM_prior, ST, F, C, I_ST, I_F, I, SM, SMI_B, B_C, B, B_W, W]
sum_product(node_list)

W.pending.update([B_W])

sum_product(reversed(node_list))

print "########################### Marginals ###########################"
for node in node_list:
    if node.__class__.__name__ == "Variable":
    	print node.name
        marginal, Z = node.marginal(None)
        print marginal
