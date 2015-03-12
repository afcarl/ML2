import numpy as np
import matplotlib.pyplot as plt

class Node(object):
    """
    Base-class for Nodes in a factor graph. Only instantiate sub-classes of Node.
    """
    def __init__(self, name):
        # A name for this Node, for printing purposes
        self.name = name
        
        # Neighbours in the graph, identified with their index in this list.
        # i.e. self.neighbours contains neighbour 0 through len(self.neighbours) - 1.
        self.neighbours = []
        
        # Reset the node-state (not the graph topology)
        self.reset()
        
    def reset(self):
        # Incoming messages; a dictionary mapping neighbours to messages.
        # That is, it maps  Node -> np.ndarray.
        self.in_msgs = {}
        
        # A set of neighbours for which this node has pending messages.
        # We use a python set object so we don't have to worry about duplicates.
        self.pending = set([])

    def add_neighbour(self, nb):
        self.neighbours.append(nb)

    def send_sp_msg(self, other):
        # To be implemented in subclass.
        raise Exception('Method send_sp_msg not implemented in base-class Node')
   
    def send_ms_msg(self, other):
        # To be implemented in subclass.
        raise Exception('Method send_ms_msg not implemented in base-class Node')
    
    def receive_msg(self, other, msg):
        # Store the incoming message, replacing previous messages from the same node
        self.in_msgs[other] = msg
        
        #Add pending neighbours (Note in our implementation there is no need to differentiate between loopy and non-loopy)
        for neighbour in self.neighbours:
            if neighbour == other:
                continue
            elif neighbour in self.in_msgs.keys() and len(self.in_msgs) == len(self.neighbours):
                self.pending.update([neighbour])
            elif neighbour not in self.in_msgs.keys() and len(self.in_msgs) == (len(self.neighbours) - 1):
                self.pending.update([neighbour])

    
    def __str__(self):
        # This is printed when using 'print node_instance'
        return self.name


class Variable(Node):
    def __init__(self, name, num_states):
        """
        Variable node constructor.
        Args:
            name: a name string for this node. Used for printing. 
            num_states: the number of states this variable can take.
            Allowable states run from 0 through (num_states - 1).
            For example, for a binary variable num_states=2,
            and the allowable states are 0, 1.
        """
        self.num_states = num_states
        
        # Call the base-class constructor
        super(Variable, self).__init__(name)
    
    def set_observed(self, observed_state):
        """
        Set this variable to an observed state.
        Args:
            observed_state: an integer value in [0, self.num_states - 1].
        """
        # Observed state is represented as a 1-of-N variable
        # Could be 0.0 for sum-product, but log(0.0) = -inf so a tiny value is preferable for max-sum
        self.observed_state[:] = 0.000001
        self.observed_state[observed_state] = 1.0
        
    def set_latent(self):
        """
        Erase an observed state for this variable and consider it latent again.
        """
        # No state is preferred, so set all entries of observed_state to 1.0
        # Using this representation we need not differentiate between observed and latent
        # variables when sending messages.
        self.observed_state[:] = 1.0
        
    def reset(self):
        super(Variable, self).reset()
        self.observed_state = np.ones(self.num_states)
        
    def marginal(self, Z=None):
        """
        Compute the marginal distribution of this Variable.
        It is assumed that message passing has completed when this function is called.
        Args:
            Z: an optional normalization constant can be passed in. If None is passed, Z is computed.
        Returns: marginal, Z. The first is a numpy array containing the normalized marginal distribution.
         Z is either equal to the input Z, or computed in this function (if Z=None was passed).
        """
        assert(len(self.in_msgs) == len(self.neighbours))
        
        marginal = np.multiply.reduce(self.in_msgs.values())

        marginal = marginal * self.observed_state

        #compute Z
        if Z == None:
            Z = np.sum(marginal)

        normalized_marginal = marginal / Z
        
        return normalized_marginal, Z

    def max(self):
        assert(len(self.in_msgs) == len(self.neighbours))
        
        marginal = np.add.reduce(self.in_msgs.values())

        marginal += np.log(self.observed_state)

        return np.argmax(marginal)

    
    def send_sp_msg(self, other):
        if len(self.neighbours) == 1:
            other.receive_msg(self,np.array([1.,1.]) * self.observed_state)
            self.pending.remove(other)
            return
            
        received = dict(self.in_msgs)
        if other in received:
            del received[other]

        assert(len(received) == (len(self.neighbours) - 1)), "Not all necessary messages have been received"

        new_msg = np.multiply.reduce(received.values())

        new_msg = new_msg * self.observed_state
        
        other.receive_msg(self,new_msg)
        self.pending.remove(other)
            
       
    def send_ms_msg(self, other):
        if len(self.neighbours) == 1:
            new_msg = np.array([0.,0.]) + np.log(self.observed_state)
            new_msg = new_msg / (np.sum(np.abs(new_msg)) + 1e-7)
            other.receive_msg(self,new_msg)
            self.pending.remove(other)
            return
            
        received = dict(self.in_msgs)
        if other in received:
            del received[other]

        assert(len(received) == (len(self.neighbours) - 1)), "Not all necessary messages have been received"

        new_msg = np.add.reduce(received.values())

        new_msg = new_msg + np.log(self.observed_state)

        new_msg = new_msg / np.sum(np.abs(new_msg), keepdims=True)
        
        other.receive_msg(self,new_msg)
        self.pending.remove(other)

class Factor(Node):
    def __init__(self, name, f, neighbours):
        """
        Factor node constructor.
        Args:
            name: a name string for this node. Used for printing
            f: a numpy.ndarray with N axes, where N is the number of neighbours.
               That is, the axes of f correspond to variables, and the index along that axes corresponds to a value of that variable.
               Each axis of the array should have as many entries as the corresponding neighbour variable has states.
            neighbours: a list of neighbouring Variables. Bi-directional connections are created.
        """
        # Call the base-class constructor
        super(Factor, self).__init__(name)

        assert len(neighbours) == f.ndim, 'Factor function f should accept as many arguments as this Factor node has neighbours'
        
        for nb_ind in range(len(neighbours)):
            nb = neighbours[nb_ind]
            assert f.shape[nb_ind] == nb.num_states, 'The range of the factor function f is invalid for input %i %s' % (nb_ind, nb.name)
            self.add_neighbour(nb)
            nb.add_neighbour(self) 
        self.f = f
        
    def send_sp_msg(self, other):
        # TODO: implement Factor -> Variable message for sum-product
        
        #Check if all messages from all neighbors except other have been received
        received = dict(self.in_msgs)
        if other in received:
            del received[other]
    
        assert(len(received) == (self.f.ndim - 1)), "Not all necessary messages have been received"
            
        msg_n = [n for n in self.neighbours if received.get(n) != None]
        received_ordered = [received.get(n) for n in msg_n]
        msg_product = np.multiply.reduce(np.ix_(*received_ordered))
        
        axes = [self.neighbours.index(n) for n in msg_n]
        axes = (axes,range(len(axes)))
        
        new_msg = np.tensordot(self.f,msg_product, axes=axes)

        other.receive_msg(self,new_msg)
        self.pending.remove(other)
           
    def send_ms_msg(self, other):
        #Check if all messages from all neighbors except other have been received
        received = dict(self.in_msgs)
        if other in received:
            del received[other]
    
        assert(len(received) == (self.f.ndim - 1)), "Not all necessary messages have been received"
            
        msg_n = [n for n in self.neighbours if received.get(n) != None]

        received_ordered = [received.get(n) for n in msg_n]

        msg_sum = np.add.reduce(np.ix_(*received_ordered))
        
        new_shape = list(msg_sum.shape)
        new_shape.insert(self.neighbours.index(other), 1)
        msg_sum_f = np.log(self.f) + msg_sum.reshape(new_shape)

        axes = [self.neighbours.index(n) for n in msg_n]
        
        max_msg = np.apply_over_axes(np.amax, msg_sum_f, axes).squeeze()

        other.receive_msg(self,max_msg)
        self.pending.remove(other)
