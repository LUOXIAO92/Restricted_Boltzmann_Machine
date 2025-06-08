import time

import numpy as np
from functools import reduce
from operator import __mul__

from .Restricted_Boltzmann_Machine import RBM


class RBM_training:
    def __init__(
            self, 
            rbm                 : RBM, 
            epochs              : int,
            training_samples    : np.ndarray,
            validation_samples  : np.ndarray,
            testing_samples     : np.ndarray,
            batch_size          : int   = 1,
            learning_rate       : float = 0.001,
            algorithm           : str   = "PCD"
        ):
        
        self.rbm            = rbm
        self.epochs         = epochs
        self.batch_size     = batch_size
        self.learning_rate  = learning_rate
        self.algorithm      = algorithm

        self.training_samples   = training_samples     
        self.num_train_samples  = training_samples.shape[0]
        self.validation_samples = validation_samples
        self.num_validation     = validation_samples.shape[0]
        self.testing_samples    = testing_samples      
        self.num_test_samples   = testing_samples.shape[0]

        self.loss                = []
        self.training_accuracy   = []
        self.validation_accuracy = []
        self.testing_accuracy    = []
        self.free_energy         = []

        self.__flatten_shape  = reduce(__mul__, training_samples.shape[1:])
    
    def __accuracy(
            self, 
            samples   : np.ndarray, 
            predicted : np.ndarray
            ):
        
        acc = 1 - np.average(np.abs(samples - predicted))

        return acc

    def start_training(self, print_interval : int = 20):
        
        t_start = time.time()
        for i in range(self.epochs):
            loss                = []
            training_accuracy   = []
            validation_accuracy = []
            free_energy         = []

            for i_batch in range(self.num_train_samples // self.batch_size):
                
                #Slice training data
                train_data_slice = slice(i_batch * self.batch_size, (i_batch + 1) * self.batch_size)
                v_data = self.training_samples[train_data_slice].reshape((self.batch_size, self.__flatten_shape))
                #Forward
                v_model, h_model, h_data, p_hv_model, _, p_hv_data = self.rbm.forward(v_data)
                #Calculate loss
                free_energy_data  = self.rbm.free_energy(v_data )
                loss.append(self.rbm.pseudo_likelihood(v_data))
                free_energy.append(free_energy_data)
                #Calculate training accuracy
                training_accuracy.append(self.__accuracy(v_data, v_model))

                #Slice validation data
                start = i_batch * self.batch_size % self.num_validation
                stop  = (i_batch + 1) * self.batch_size % self.num_validation
                stop  = stop if stop > start else (i_batch + 1) * self.batch_size
                validation_data_slice = slice(start, stop)

                v_validation = self.validation_samples[validation_data_slice]
                v_validation = np.reshape(v_validation, (v_validation.shape[0], self.__flatten_shape))
                #Get prediction with validation data
                v_vali_model, _, _, _, _, _ = self.rbm.forward(v_validation)
                #Calculate validation accuracy
                validation_accuracy.append(self.__accuracy(v_validation, v_vali_model))

                #Update parameters
                self.rbm.backward(
                    v_data          = v_data, 
                    h_data          = h_data, 
                    v_model         = v_model, 
                    h_model         = h_model, 
                    p_hv_data       = p_hv_data,
                    p_hv_model      = p_hv_model,
                    learning_rate   = self.learning_rate,
                    algorithm       = self.algorithm
                    )
            
            training_accuracy.append(self.__accuracy(v_data, v_model))
            validation_accuracy.append(self.__accuracy(v_validation, v_vali_model))

            #Test
            testing_accuracy = []
            for i_batch in range(self.num_test_samples // min(self.num_test_samples, self.batch_size)):
                test_data_slice = slice(i_batch * self.batch_size, (i_batch + 1) * self.batch_size)
                v_test = self.testing_samples[test_data_slice]
                v_test = np.reshape(v_test, (v_test.shape[0], self.__flatten_shape))
                #Forward
                v_test_model, _, _, _, _, _ = self.rbm.forward(v_test)
                testing_accuracy.append(self.__accuracy(v_test, v_test_model))


            self.loss.append(np.average(loss))
            self.training_accuracy.append(np.average(training_accuracy))
            self.validation_accuracy.append(np.average(validation_accuracy))
            self.free_energy.append(np.average(free_energy_data))
            self.testing_accuracy.append(np.average(testing_accuracy))

            if i % print_interval == print_interval - 1:
                t_finish = time.time()
                print(f"{i+1}-th epoch finished. "
                      + f"Time= {t_finish - t_start:.2e} s. "
                      + f"Loss= {self.loss[i]:.4e} , "
                      + f"Train_acc= {self.training_accuracy[i]:.4e} , "
                      + f"Vali_acc= {self.validation_accuracy[i]:.4e} , "
                      + f"Test_acc= {self.testing_accuracy[i]:.4e} , "
                      + f"Free_en= {self.free_energy[i]:.4e}"
                      )
                t_start = time.time()
