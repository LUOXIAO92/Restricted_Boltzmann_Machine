import numpy as np
import Functions as F
import copy

class RBM:
    def __init__(
            self, 
            visible_size    : int, 
            hidden_size     : int, 
            inverse_temp    : float = 1,
            k               : int   = 1,
            spin_low        : float = 0,
            spin_high       : float = 1,
            num_of_states   : float = 2,
            seed            : int   = None
            ):
        
        self.vs = visible_size
        self.hs = hidden_size 

        self.beta = inverse_temp

        self.k = k

        # v^T @ W @ h + v^T @ θv + θh @ h
        rs = np.random.RandomState(seed)
        self.Weight = rs.normal(size = (self.vs, self.hs))
        self.bias_v = rs.normal(size = self.vs)
        self.bias_h = rs.normal(size = self.hs)

        #All states of a spin
        self.spin_low         = spin_low
        self.spin_high        = spin_high
        self.num_of_states    = num_of_states
        self.__all_states = np.linspace(start = spin_low, stop = spin_high, num = num_of_states)
    
    def internal_energy(self, v : np.ndarray, h : np.ndarray):
        E = - v @ self.Weight @ h + v @ self.bias_v + h @ self.bias_h
        E = - np.einsum("bi,ij,bj->b", v, self.Weight, h)   \
            + np.einsum("bi,i->b", v, self.bias_v)          \
            + np.einsum("j,bj->b", self.bias_h, h)
        #Return average 
        return np.average(E)
    
    
    def free_energy(self, v : np.ndarray):
        batch_size, _ = v.shape

        external_field_term = np.einsum("bi,i->b", v, self.bias_v)

        _bias_h_padded = np.pad(
            array       = self.bias_h.reshape((1, self.hs)), 
            pad_width   = ((0, batch_size - 1), (0, 0)) 
            )
        _effective_field = np.log(1 + np.exp(np.einsum("bi,ij->bj", v, self.Weight) + _bias_h_padded))
        effective_field_term = np.sum(_effective_field, axis = 1)

        f = - self.beta * (external_field_term + effective_field_term)

        return np.average(f)
    
    def __get_most_likely_state(self, probabilities : np.ndarray):

        batch_size, size, _ = probabilities.shape
        
        states = np.concatenate(
            [self.__all_states for _ in range(batch_size * size)],
            axis = 0
            )
        states = states.reshape((batch_size, size, self.num_of_states))

        state = F.get_most_likely_state(states, probabilities)

        return state

    def __visible_to_hidden(self, v : np.ndarray):

        A = self.beta * (np.einsum("bi,ij->bj", v, self.Weight) - self.bias_h)

        p_h_under_given_v = F.softmax(
            x    = np.einsum("bj,k->bjk", A, self.__all_states), 
            axis = 2
            )

        h = self.__get_most_likely_state(p_h_under_given_v)

        return h
    
    def __hidden_to_visible(self, h : np.ndarray):
        A = self.beta * (np.einsum("ij,bj->bi", self.Weight, h) - self.bias_v)
        
        p_v_under_given_h = F.softmax(
            x    = np.einsum("bi,k->bik", A, self.__all_states), 
            axis = 2
            )
        
        v = self.__get_most_likely_state(p_v_under_given_h)

        return v
    
    def forward(self, v : np.ndarray):
        """
        Parameters
        ----------
        v : Input vectors with a size of (batch_size, input_size)
        k : Iteration of Contrastive-Divergence. Original RBM is k=1.

        Retrun
        ------
        `v_model`, `h_model`, `h_data`
        """

        h_data  = self.__visible_to_hidden(v)
        h_model = copy.deepcopy(h_data)
        for _ in range(self.k):
            v_model = self.__hidden_to_visible(h_model)
            h_model = self.__visible_to_hidden(v_model)
        
        return v_model, h_model, h_data
    
    def backward(
            self, 
            v_data          : np.ndarray, 
            h_data          : np.ndarray, 
            v_model         : np.ndarray, 
            h_model         : np.ndarray,
            learning_rate   : float = 0.001
        ):

        
        dw = np.einsum("bi,bj->bij", v_data, h_data) - np.einsum("bi,bj->bij", v_model, h_model)
        dbias_v = v_data - v_model
        dbias_h = h_data - h_model
        self.Weight += learning_rate * np.average(dw, axis = 0)
        self.bias_v += learning_rate * np.average(dbias_v, axis = 0)
        self.bias_h += learning_rate * np.average(dbias_h, axis = 0)

