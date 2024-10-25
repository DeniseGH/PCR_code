"""
This file contains the implementation of the Structural Causal Models used for modelling the effect of interventions
on the features of the individual seeking recourse, the code, especially the class SCM relies on the implementation of Dominguez et. al. (2022).
"""

import os
import numpy as np
import torch
import pandas as pd
import functions_estimation_SCM
from itertools import chain, combinations  # for the powerset of actionable combinations of interventions



class SCM:
    """
    Includes all the relevant methods required for generating counterfactuals. Classes inheriting this class must
    contain the following objects:
        self.f: list of functions, each representing a structural equation. Function self.f[i] must have i+1 arguments,
                corresponding to X_1, ..., X_{i-1}, U_{i+1} each being a torch.Tensor with shape (N, 1), and returns
                the endogenous variable X_{i+1} as a torch.Tensor with shape (N, 1)
        self.inv_f: list of functions, corresponding to the inverse mapping X -> U. Each function self.inv_f[i] takes
                    as argument the features X as a torch.Tensor with shape (N, D), and returns the corresponding
                    exogenous variable U_{i+1} as a torch.Tensor with shape (N, 1)
        self.actionable: list of int, indices of the actionable features
        self.soft_interv: list of bool with len = D, indicating whether the intervention on feature soft_interv[i] is
                          modeled as a soft intervention (True) or hard intervention (False)
        self.mean: expectation of the features, such that when generating data we can standarize it
        self.std: standard deviation of the features, such that when generating data we can standarize it
    """
    def sample_U(self, N):
        """
        Return N samples from the distribution over exogenous variables P_U.

        Inputs:     N: int, number of samples to draw

        Outputs:    U: np.array with shape (N, D)
        """
        raise NotImplementedError

    def label(self, X):
        """
        Label the input instances X

        Inputs:     X: np.array with shape (N, D)

        Outputs:    Y:  np.array with shape (N, )
        """
        raise NotImplementedError

    def generate(self, N):
        """
        Sample from the observational distribution implied by the SCM

        Inputs:     N: int, number of instances to sample

        Outputs:    X: np.array with shape (N, D), standarized (since we train the models on standarized data)
                    Y: np.array with shape (N, )
        """
        U = self.sample_U(N).astype(np.float32)
        X = self.U2X(torch.Tensor(U))
        Y = self.label(X.numpy())
        print("Y.shape", Y.shape)
        X = (X - self.mean) / self.std

        return X.numpy(), Y

    def U2X(self, U):
        """
        Map from the exogenous variables U to the endogenous variables X by using the structural equations self.f

        Inputs:     U: torch.Tensor with shape (N, D), exogenous variables

        Outputs:    X: torch.Tensor with shape (N, D), endogenous variables
        """
        X = []
        for i in range(U.shape[1]):
            X.append(self.f[i](*X[:i] + [U[:, [i]]]))
        return torch.cat(X, 1)

    def X2U(self, X):
        """
        Map from the endogenous variables to the exogenous variables by using the inverse mapping self.inv_f

        Inputs:     U: torch.Tensor with shape (N, D), exogenous variables

        Outputs:    X: torch.Tensor with shape (N, D), endogenous variables
        """
        if self.inv_f is None:
            return X + 0.

        U = torch.zeros_like(X)
        for i in range(X.shape[1]):
            U[:, [i]] = self.inv_f[i](X)
        return U

    def counterfactual(self, Xn, delta, actionable=None, soft_interv=None):
        """
        Computes the counterfactual of Xn under the intervention delta.

        Inputs:     Xn: torch.Tensor (N, D) factual
                    delta: torch.Tensor (N, D), intervention values
                    actionable: None or list of int, indices of the intervened upon variables
                    soft_interv: None or list of int, variables for which the interventions are soft (rather than hard)

        Outputs:
                    X_cf: torch.Tensor (N, D), counterfactual
        """
        actionable = self.actionable if actionable is None else actionable
        soft_interv = self.soft_interv if soft_interv is None else soft_interv

        # Abduction
        X = self.Xn2X(Xn)
        U = self.X2U(X)

        # Scale appropriately
        delta = delta * self.std

        X_cf = []
        for i in range(U.shape[1]):
            if i in actionable:
                if soft_interv[i]:
                    X_cf.append(self.f[i](*X_cf[:i] + [U[:, [i]]]) + delta[:, [i]])
                else:
                    X_cf.append(X[:, [i]] + delta[:, [i]])
            else:
                X_cf.append(self.f[i](*X_cf[:i] + [U[:, [i]]]))

        X_cf = torch.cat(X_cf, 1)
        return self.X2Xn(X_cf)

    def counterfactual_batch(self, Xn, delta, interv_mask):
        """
        Inputs:     Xn: torch.Tensor (N, D) factual
                    delta: torch.Tensor (N, D), intervention values
                    interv_sets: torch.Tensor (N, D)

        Outputs:
                    X_cf: torch.Tensor (N, D), counterfactual
        """
        N, D = Xn.shape
        soft_mask = torch.Tensor(self.soft_interv).repeat(N, 1)
        hard_mask = 1. - soft_mask

        mask_hard_actionable = hard_mask * interv_mask
        mask_soft_actionable = soft_mask * interv_mask

        return self.counterfactual_mask(Xn, delta, mask_hard_actionable, mask_soft_actionable)


    def counterfactual_mask(self, Xn, delta, mask_hard_actionable, mask_soft_actionable):
        """
        Different way of computing counterfactuals, which may be more computationally efficient in some cases, specially
        if different instances have different actionability constrains, or hard/soft intervention criteria.

        Inputs:     Xn: torch.Tensor (N, D) factual
                    delta: torch.Tensor (N, D), intervention values
                    mask_hard_actionable: torch.Tensor (N, D), 1 for actionable features under a hard intervention
                    mask_soft_actionable: torch.Tensor (N, D), 1 for actionable features under a soft intervention

        Outputs:
                    X_cf: torch.Tensor (N, D), counterfactual
        """
        # Abduction
        X = self.Xn2X(Xn)
        U = self.X2U(X)

        # Scale appropriately
        delta = delta * self.std

        X_cf = []
        for i in range(U.shape[1]):
            X_cf.append((X[:, [i]] + delta[:, [i]]) * mask_hard_actionable[:, [i]] + (1 - mask_hard_actionable[:, [i]])
                        * (self.f[i](*X_cf[:i] + [U[:, [i]]]) + delta[:, [i]] * mask_soft_actionable[:, [i]]))

        X_cf = torch.cat(X_cf, 1)
        return self.X2Xn(X_cf)

    def U2Xn(self, U):
        """
        Mapping from the exogenous variables U to the endogenous X variables, which are standarized

        Inputs:     U: torch.Tensor, shape (N, D)

        Outputs:    Xn: torch.Tensor, shape (N, D), is standarized
        """
        return self.X2Xn(self.U2X(U))

    def Xn2U(self, Xn):
        """
        Mapping from the endogenous variables X (standarized) to the exogenous variables U

        Inputs:     Xn: torch.Tensor, shape (N, D), endogenous variables (features) standarized

        Outputs:    U: torch.Tensor, shape (N, D)
        """
        return self.X2U(self.Xn2X(Xn))

    def Xn2X(self, Xn):
        """
        Transforms the endogenous features to their original form (no longer standarized)

        Inputs:     Xn: torch.Tensor, shape (N, D), features are standarized

        Outputs:    X: torch.Tensor, shape (N, D), features are not standarized
        """
        return Xn * self.std + self.mean

    def X2Xn(self, X):
        """
        Standarizes the endogenous variables X according to self.mean and self.std

        Inputs:     X: torch.Tensor, shape (N, D), features are not standarized

        Outputs:    Xn: torch.Tensor, shape (N, D), features are standarized
        """
        return (X - self.mean) / self.std

    def getActionable(self):
        """ Returns the indices of the actionable features, as a list of ints. """
        return self.actionable

    def getPowerset(self, actionable):
        """ Returns the power set of the set of actionable features, as a list of lists of ints. """
        s = actionable
        return list(chain.from_iterable(combinations(s, r) for r in range(len(s) + 1)))[1:]

    def build_mask(self, mylist, shape):
        """
        Builds a torch.Tensor mask according to the list of indices contained in mylist. Used to build the masks of
        actionable features, or those of variables which are intervened upon with soft interventions.

        Inputs:     mylist: list(D) of ints or list(N) of lists(D) of ints, corresponding to indices
                    shape: list of ints [N, D]

        Outputs:    mask: torch.Tensor with shape (N, D), where mask[i, j] = 1. if j in mylist (for list of ints) or
                          j in mylist[i] (for list of list of ints)
        """
        mask = torch.zeros(shape)
        if type(mylist[0]) == list: # nested list
            for i in range(len(mylist)):
                mask[i, mylist[i]] = 1.
        else:
            mask[:, mylist] = 1.
        return mask

    def get_masks(self, actionable, shape):
        """
        Returns the mask of actionable features, actionable features which are soft intervened, and actionable
        features which are hard intervened.

        Inputs:     actionable: list(D) of int, or list(N) of list(D) of int, containing the indices of actionable feats
                    shape: list of int [N, D]

        Outputs:    mask_actionable: torch.Tensor (N, D)
                    mask_soft_actionable: torch.Tensor (N, D)
                    mask_hard_actionable: torch.Tensor (N, D)
        """
        mask_actionable = self.build_mask(actionable, shape)
        mask_soft = self.build_mask(list(np.where(self.soft_interv)[0]), shape)
        mask_hard_actionable = (1 - mask_soft) * mask_actionable
        mask_soft_actionable = mask_soft * mask_actionable
        return mask_actionable, mask_soft_actionable, mask_hard_actionable

# ----------------------------------------------------------------------------------------------------------------------
# The following 5 classes are part of the thesis in Personalized Causal Reocurse
# ----------------------------------------------------------------------------------------------------------------------

class SCM_PCR_linear_Prior(SCM):
    """ Semi-synthetic SCM inspired by https://arxiv.org/pdf/2006.06831, introduced by Karimi et al. """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        thetas = [0, 0, 0]

        # Set the file path to save the results
        # current_dir = os.getcwd()  # Get the current working directory
        # result_folder = os.path.join(current_dir, "Results")
        # result_file_path = os.path.join(result_folder, f'results_log.txt')

        # Open the file in write mode
        # with open(result_file_path, 'a') as f:

        #     # Write the true thetas to the file
        #     f.write("######### TRUE #########\n")
        #     f.write(f"PRIOR THETAS: {thetas}\n")

        print("######### TRUE #########\n")
        print(f"PRIOR THETAS: {thetas}\n")

        # Define the functions f and inv_f
        self.f = [lambda U0: U0,
                lambda X0, U1: thetas[0] * X0 + U1,
                lambda X0, X1, U2: thetas[1] * X0 + thetas[2] * X1 + U2,
                ]

        self.inv_f = [lambda X: X[:, [0]],
                    lambda X: X[:, [1]] - thetas[0] * X[:, [0]],
                    lambda X: X[:, [2]] - thetas[1] * X[:, [0]] - thetas[2] * X[:, [1]],
                    ]

        # Generate SCM data
        scm_generated = functions_estimation_SCM.generate_scm(thetas)
        df = scm_generated.sample(n_samples=100)

        # Calculate mean and standard deviation
        self.mean = torch.Tensor([np.mean(df["x1"]), np.mean(df["x2"]), np.mean(df["x3"])])
        self.std = torch.Tensor([np.std(df["x1"]), np.std(df["x2"]), np.std(df["x3"])])

        # Write the calculated mean and std to the file
        # f.write(f"MEAN: {self.mean}\n")
        # f.write(f"STD: {self.std}\n")

        print(f"MEAN: {self.mean}\n")
        print(f"STD: {self.std}\n\n")

        # Set actionable and soft intervention values
        self.actionable = [0, 1, 2]
        self.soft_interv = [True, True, True]



    def sample_U(self, N):
       
        U1 = np.random.normal(0, 1, N)
        U2 = np.random.normal(0, 1, N)
        U3 = np.random.normal(0, 1, N)
        return np.c_[U1, U2, U3]

    def label(self, X):
        X_sum = X[:, 0]+ X[:, 1]+X[:, 2]  # Sum X1 + X2 + X3 for each sample
        rho = np.mean(X_sum)  # Compute the average of X1 + X2 + X3 across all samples
        
        # Compute the logistic function for each sample
        p = (1 + np.exp(-2.5 * (X_sum / rho)))**-1
        
        # Sample Y from a Bernoulli distribution based on these probabilities
        Y = np.random.binomial(1, p)
        
        return Y

    def get_Jacobian(self):
        """
        Computes the Jacobian matrix for the linear SCM.
        """

        thetas = [0, 0, 0]

        return np.array([[1, 0, 0],
                         [thetas[0], 1, 0],
                         [thetas[1], thetas[2], 1]])


    def get_Jacobian_interv(self, interv_set):
        """
        Computes the Jacobian matrix under interventions.
        Parameters:
        - interv_set: A list of indices where interventions occur
        """
        # Get the standard Jacobian
        J = self.get_Jacobian()

        # Iterate over the intervention set
        for i in interv_set:
            # If the intervention is hard (not soft), modify the Jacobian
            if not self.soft_interv[i]:
                # Set upstream effects to zero (all previous columns in the row)
                for j in range(i):
                    J[i][j] = 0.0
        return J


# Example from the paper thetas = [-0.99, 0.05, 0.25]

class SCM_PCR_linear(SCM):
    """ Semi-synthetic SCM inspired by https://arxiv.org/pdf/2006.06831, introduced by Karimi et al. """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        thetas = [-0.99, 0.05, 0.25]

        # Set the file path to save the results
        # current_dir = os.getcwd()  # Get the current working directory
        # result_folder = os.path.join(current_dir, "Results")
        # result_file_path = os.path.join(result_folder, f'results_log.txt')

        # Open the file in write mode
        # with open(result_file_path, 'a') as f:

        # # Write the true thetas to the file
        # f.write("######### TRUE #########\n")
        # f.write(f"TRUE THETAS: {thetas}\n")

        print("######### TRUE #########\n")
        #print(f"TRUE THETAS: {thetas}\n")

        # Define the functions f and inv_f
        self.f = [lambda U0: U0,
                    lambda X0, U1: thetas[0] * X0 + U1,
                    lambda X0, X1, U2: thetas[1] * X0 + thetas[2] * X1 + U2,
                    ]

        self.inv_f = [lambda X: X[:, [0]],
                        lambda X: X[:, [1]] - thetas[0] * X[:, [0]],
                        lambda X: X[:, [2]] - thetas[1] * X[:, [0]] - thetas[2] * X[:, [1]],
                        ]

        # Generate SCM data
        scm_generated = functions_estimation_SCM.generate_scm(thetas)
        df = scm_generated.sample(n_samples=100)

        # Calculate mean and standard deviation
        self.mean = torch.Tensor([np.mean(df["x1"]), np.mean(df["x2"]), np.mean(df["x3"])])
        self.std = torch.Tensor([np.std(df["x1"]), np.std(df["x2"]), np.std(df["x3"])])

        # Write the calculated mean and std to the file
        #f.write(f"MEAN: {self.mean}\n")
        #f.write(f"STD: {self.std}\n")

        print(f"MEAN: {self.mean}\n")
        print(f"STD: {self.std}\n\n")

        # Set actionable and soft intervention values
        self.actionable = [0, 1, 2]
        self.soft_interv = [True, True, True]



    def sample_U(self, N):
        # U1: Mixture of Gaussians
        # 50% from N(-2, 1.5^2) and 50% from N(1, 1^2)
        mixture_component = np.random.choice([0, 1], size=N, p=[0.5, 0.5])
        U1 = np.where(mixture_component == 0, 
                    np.random.normal(-2, 1.5, N),  # From N(-2, 1.5^2)
                    np.random.normal(1, 1, N))    # From N(1, 1^2)
    
        U2 = np.random.normal(0, 1, N)
        U3 = np.random.normal(0, 1, N)
        return np.c_[U1, U2, U3]

    def label(self, X):
        X_sum = X[:, 0]+ X[:, 1]+X[:, 2]  # Sum X1 + X2 + X3 for each sample
        rho = np.mean(X_sum)  # Compute the average of X1 + X2 + X3 across all samples
        
        # Compute the logistic function for each sample
        p = (1 + np.exp(-2.5 * (X_sum / rho)))**-1
        
        # Sample Y from a Bernoulli distribution based on these probabilities
        Y = np.random.binomial(1, p)
        
        return Y

    def get_Jacobian(self):
        """
        Computes the Jacobian matrix for the linear SCM.
        """

        thetas = [-0.99, 0.99, 0.99]

        return np.array([[1, 0, 0],
                         [thetas[0], 1, 0],
                         [thetas[1], thetas[2], 1]])


    def get_Jacobian_interv(self, interv_set):
        """
        Computes the Jacobian matrix under interventions.
        Parameters:
        - interv_set: A list of indices where interventions occur
        """
        # Get the standard Jacobian
        J = self.get_Jacobian()

        # Iterate over the intervention set
        for i in interv_set:
            # If the intervention is hard (not soft), modify the Jacobian
            if not self.soft_interv[i]:
                # Set upstream effects to zero (all previous columns in the row)
                for j in range(i):
                    J[i][j] = 0.0
        return J


class SCM_PCR_linear_estimated(SCM):
    """ Semi-synthetic ESTIMATED SCM inspired by https://arxiv.org/pdf/2006.06831, introduced by Karimi et al. """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Set the file path to save the results
        # current_dir = os.getcwd()  # Get the current working directory
        # result_folder = os.path.join(current_dir, "Results")
        # result_file_path = os.path.join(result_folder, f'results_log.txt')

        # with open(result_file_path, 'w') as f:

        # print("\n######### ESTIMATED #########")
        # f.write("######### ESTIMATED #########\n")
        folder_path = f"./SCM_Estimation_Interventions_Paper_Thesis/"
        file_path = os.path.join(folder_path, 'final_results_Interventions_10.csv')
        
        # Read the CSV file into a pandas DataFrame
        try:
            data = pd.read_csv(file_path)
        except FileNotFoundError:
            print(f"File not found at path: {file_path}")
            data = None
    
        # Calculate the mean for each column
        thetas = data.mean().tolist()
        
        print("ESTIMATED THETAS", thetas)
        # f.write(f"ESTIMATED THETAS: {thetas}\n")


        self.f = [lambda U0: U0 ,
                lambda X0, U1: (thetas[0])*X0+ U1, 
                lambda X0, X1, U2: thetas[1]*X0 +thetas[2]*X1+U2, 
                ]

        self.inv_f = [lambda X: X[:, [0]],
                    lambda X: X[:, [1]] - thetas[0]*X[:, [0]],
                    lambda X: X[:, [2]] - thetas[1]*X[:, [0]] - thetas[2]*X[:, [1]],
                    ]


        scm_generated = functions_estimation_SCM.generate_scm(thetas)
        df=scm_generated.sample(n_samples=100)

        

        # Compute the mean of x1, x2, and x3
        x1_mean = np.mean(df["x1"])
        x2_mean = np.mean(df["x2"])
        x3_mean = np.mean(df["x3"])

        self.mean = torch.Tensor([x1_mean, x2_mean, x3_mean]) #mean of the etimated
        
        self.std = torch.Tensor([np.std(df["x1"]), np.std(df["x2"]), np.std(df["x3"])])
        
        print("MEAN: ",self.mean, " STD: ", self.std)

        # f.write(f"MEAN: {self.mean}\n")
        # f.write(f"STD: {self.std}\n")
        print()
        self.actionable = [0, 1, 2]
        self.soft_interv = [True, True, True]

    def sample_U(self, N):

        U1 = np.random.normal(0, 1, N)
        U2 = np.random.normal(0, 1, N)
        U3 = np.random.normal(0, 1, N)
        return np.c_[U1, U2, U3]

    def label(self, X):
        X_sum = X[:, 0]+ X[:, 1]+X[:, 2]  # Sum X1 + X2 + X3 for each sample
        rho = np.mean(X_sum)  # Compute the average of X1 + X2 + X3 across all samples
        
        # Compute the logistic function for each sample
        p = (1 + np.exp(-2.5 * (X_sum / rho)))**-1
        
        # Sample Y from a Bernoulli distribution based on these probabilities
        Y = np.random.binomial(1, p)
        
        return Y

    def get_Jacobian(self):
        """
        Computes the Jacobian matrix for the linear SCM.
        """

        folder_path = f"/home/tampieri/Results_Interventions_SD/IterationsResults_0927_2300/"

        file_path = os.path.join(folder_path, 'final_results_Interventions_10.csv')
        
        # Read the CSV file into a pandas DataFrame
        try:
            data = pd.read_csv(file_path)
        except FileNotFoundError:
            print(f"File not found at path: {file_path}")
            data = None
    
        # Calculate the mean for each column
        thetas = data.mean().tolist()
        return np.array([[1, 0, 0],
                         [thetas[0], 1, 0],
                         [thetas[1], thetas[2], 1]])


    def get_Jacobian_interv(self, interv_set):
        """
        Computes the Jacobian matrix under interventions.
        Parameters:
        - interv_set: A list of indices where interventions occur
        """
        # Get the standard Jacobian
        J = self.get_Jacobian()

        # Iterate over the intervention set
        for i in interv_set:
            # If the intervention is hard (not soft), modify the Jacobian
            if not self.soft_interv[i]:
                # Set upstream effects to zero (all previous columns in the row)
                for j in range(i):
                    J[i][j] = 0.0

        return J


# Example from our tests thetas = [-0.99, 0.99, 0.99]

class SCM_PCR_linear_Test(SCM):
    """ Semi-synthetic SCM inspired by https://arxiv.org/pdf/2006.06831, introduced by Karimi et al. """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        thetas = [-0.99, 0.99, 0.99]

        # Set the file path to save the results
        current_dir = os.getcwd()  # Get the current working directory
        result_folder = os.path.join(current_dir, "Results")
        result_file_path = os.path.join(result_folder, f'results_log.txt')

        # Open the file in write mode
        with open(result_file_path, 'a') as f:

            # Write the true thetas to the file
            f.write("######### TRUE #########\n")
            f.write(f"TRUE THETAS: {thetas}\n")

            print("######### TRUE #########\n")
            print(f"TRUE THETAS: {thetas}\n")

            # Define the functions f and inv_f
            self.f = [lambda U0: U0,
                      lambda X0, U1: thetas[0] * X0 + U1,
                      lambda X0, X1, U2: thetas[1] * X0 + thetas[2] * X1 + U2,
                      ]

            self.inv_f = [lambda X: X[:, [0]],
                          lambda X: X[:, [1]] - thetas[0] * X[:, [0]],
                          lambda X: X[:, [2]] - thetas[1] * X[:, [0]] - thetas[2] * X[:, [1]],
                          ]

            # Generate SCM data
            scm_generated = functions_estimation_SCM.generate_scm(thetas)
            df = scm_generated.sample(n_samples=100)

            # Calculate mean and standard deviation
            self.mean = torch.Tensor([np.mean(df["x1"]), np.mean(df["x2"]), np.mean(df["x3"])])
            self.std = torch.Tensor([np.std(df["x1"]), np.std(df["x2"]), np.std(df["x3"])])

            # Write the calculated mean and std to the file
            f.write(f"MEAN: {self.mean}\n")
            f.write(f"STD: {self.std}\n")

            print(f"MEAN: {self.mean}\n")
            print(f"STD: {self.std}\n\n")

            # Set actionable and soft intervention values
            self.actionable = [0, 1, 2]
            self.soft_interv = [True, True, True]



    def sample_U(self, N):
        thetas = [-0.99, 0.99, 0.99]
    
        U1 = np.random.normal(0, 1, N)
        U2 = np.random.normal(0, 1, N)
        U3 = np.random.normal(0, 1, N)
        return np.c_[U1, U2, U3]

    def label(self, X):
        X_sum = X[:, 0]+ X[:, 1]+X[:, 2]  # Sum X1 + X2 + X3 for each sample
        rho = np.mean(X_sum)  # Compute the average of X1 + X2 + X3 across all samples
        
        # Compute the logistic function for each sample
        p = (1 + np.exp(-2.5 * (X_sum / rho)))**-1
        
        # Sample Y from a Bernoulli distribution based on these probabilities
        Y = np.random.binomial(1, p)
        
        return Y

    def get_Jacobian(self):
        """
        Computes the Jacobian matrix for the linear SCM.
        """

        thetas = [-0.99, 0.99, 0.99]

        return np.array([[1, 0, 0],
                         [thetas[0], 1, 0],
                         [thetas[1], thetas[2], 1]])


    def get_Jacobian_interv(self, interv_set):
        """
        Computes the Jacobian matrix under interventions.
        Parameters:
        - interv_set: A list of indices where interventions occur
        """
        # Get the standard Jacobian
        J = self.get_Jacobian()

        # Iterate over the intervention set
        for i in interv_set:
            # If the intervention is hard (not soft), modify the Jacobian
            if not self.soft_interv[i]:
                # Set upstream effects to zero (all previous columns in the row)
                for j in range(i):
                    J[i][j] = 0.0
        return J

class SCM_PCR_linear_estimated_Test(SCM):
    """ Semi-synthetic ESTIMATED SCM inspired by https://arxiv.org/pdf/2006.06831, introduced by Karimi et al. """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Set the file path to save the results
        current_dir = os.getcwd()  # Get the current working directory
        result_folder = os.path.join(current_dir, "Results")
        result_file_path = os.path.join(result_folder, f'results_log.txt')

        with open(result_file_path, 'w') as f:

            print("\n######### ESTIMATED #########")
            f.write("######### ESTIMATED #########\n")
            folder_path = f"/home/tampieri/SCM_Estimation_Interventions_Test_Thesis/"

            file_path = os.path.join(folder_path, 'final_results_Interventions_10.csv')
            
            # Read the CSV file into a pandas DataFrame
            try:
                data = pd.read_csv(file_path)
            except FileNotFoundError:
                print(f"File not found at path: {file_path}")
                data = None
        
            # Calculate the mean for each column
            thetas = data.mean().tolist()
            
            print("ESTIMATED THETAS", thetas)
            f.write(f"ESTIMATED THETAS: {thetas}\n")


            self.f = [lambda U0: U0 ,
                    lambda X0, U1: (thetas[0])*X0+ U1, 
                    lambda X0, X1, U2: thetas[1]*X0 +thetas[2]*X1+U2, 
                    ]

            self.inv_f = [lambda X: X[:, [0]],
                        lambda X: X[:, [1]] - thetas[0]*X[:, [0]],
                        lambda X: X[:, [2]] - thetas[1]*X[:, [0]] - thetas[2]*X[:, [1]],
                        ]


            scm_generated = functions_estimation_SCM.generate_scm(thetas)
            df=scm_generated.sample(n_samples=100)

            

            # Compute the mean of x1, x2, and x3
            x1_mean = np.mean(df["x1"])
            x2_mean = np.mean(df["x2"])
            x3_mean = np.mean(df["x3"])

            self.mean = torch.Tensor([x1_mean, x2_mean, x3_mean]) #mean of the etimated
            #self.mean =  torch.Tensor([ 0.0790, -0.2219,  0.0546]) # mean of the true thetas
            

            self.std = torch.Tensor([np.std(df["x1"]), np.std(df["x2"]), np.std(df["x3"])])
            #self.std = torch.Tensor([0.8780, 1.4748, 1.0167]) # std of the true thetas
            print("MEAN: ",self.mean, " STD: ", self.std)

            f.write(f"MEAN: {self.mean}\n")
            f.write(f"STD: {self.std}\n")
            print()
            self.actionable = [0, 1, 2]
            self.soft_interv = [True, True, True]

    def sample_U(self, N):

        U1 = np.random.normal(0, 1, N)
        U2 = np.random.normal(0, 1, N)
        U3 = np.random.normal(0, 1, N)
        return np.c_[U1, U2, U3]

    def label(self, X):
        X_sum = X[:, 0]+ X[:, 1]+X[:, 2]  # Sum X1 + X2 + X3 for each sample
        rho = np.mean(X_sum)  # Compute the average of X1 + X2 + X3 across all samples
        
        # Compute the logistic function for each sample
        p = (1 + np.exp(-2.5 * (X_sum / rho)))**-1
        
        # Sample Y from a Bernoulli distribution based on these probabilities
        Y = np.random.binomial(1, p)
        
        return Y

    def get_Jacobian(self):
        """
        Computes the Jacobian matrix for the linear SCM.
        """

        folder_path = f"/home/tampieri/Results_Interventions_SD/IterationsResults_0927_2300/"

        file_path = os.path.join(folder_path, 'final_results_Interventions_10.csv')
        
        # Read the CSV file into a pandas DataFrame
        try:
            data = pd.read_csv(file_path)
        except FileNotFoundError:
            print(f"File not found at path: {file_path}")
            data = None
    
        # Calculate the mean for each column
        thetas = data.mean().tolist()
        return np.array([[1, 0, 0],
                         [thetas[0], 1, 0],
                         [thetas[1], thetas[2], 1]])


    def get_Jacobian_interv(self, interv_set):
        """
        Computes the Jacobian matrix under interventions.
        Parameters:
        - interv_set: A list of indices where interventions occur
        """
        # Get the standard Jacobian
        J = self.get_Jacobian()

        # Iterate over the intervention set
        for i in interv_set:
            # If the intervention is hard (not soft), modify the Jacobian
            if not self.soft_interv[i]:
                # Set upstream effects to zero (all previous columns in the row)
                for j in range(i):
                    J[i][j] = 0.0

        return J

# Example from our non linear SCM 

class SCM_PCR_not_linear(SCM):
    """ Semi-synthetic SCM inspired by https://arxiv.org/pdf/2006.06831, introduced by Karimi et al. """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        thetas = [0, 0, 0]

        # Set the file path to save the results
        current_dir = os.getcwd()  # Get the current working directory
        result_folder = os.path.join(current_dir, "Results")
        result_file_path = os.path.join(result_folder, f'results_log.txt')

        # Open the file in write mode
        with open(result_file_path, 'a') as f:

            # Write the true thetas to the file
            f.write("######### TRUE #########\n")
            #f.write(f"TRUE THETAS: {thetas}\n")

            print("######### TRUE #########\n")
            #print(f"TRUE THETAS: {thetas}\n")

            # Define the functions f and inv_f
            self.f = [lambda U0: U0,
                      lambda X0, U1: -1 +3/(1+ torch.exp(-2*X0)) + U1,
                      lambda X0, X1, U2: -0.05 * X0 + 0.25 * X1*X1  + U2,
                      ]

            self.inv_f = [lambda X: X[:, [0]],
                          lambda X: X[:, [1]] + 1 - 3/(1+ torch.exp(-2*X[:, [0]])),
                          lambda X: X[:, [2]] + 0.05 * X[:, [0]] - 0.25 * X[:, [1]] * X[:, [1]]  ,
                          ]

            # Generate SCM data
            scm_generated = functions_estimation_SCM.generate_scm(thetas)
            df = scm_generated.sample(n_samples=100)

            # Calculate mean and standard deviation
            self.mean = torch.Tensor([np.mean(df["x1"]), np.mean(df["x2"]), np.mean(df["x3"])])
            self.std = torch.Tensor([np.std(df["x1"]), np.std(df["x2"]), np.std(df["x3"])])

            # Write the calculated mean and std to the file
            f.write(f"MEAN: {self.mean}\n")
            f.write(f"STD: {self.std}\n")

            print(f"MEAN: {self.mean}\n")
            print(f"STD: {self.std}\n\n")

            # Set actionable and soft intervention values
            self.actionable = [0, 1, 2]
            self.soft_interv = [True, True, True]



    def sample_U(self, N):

        mixture_component = np.random.choice([0, 1], size=N, p=[0.5, 0.5])
        U1 = np.where(mixture_component == 0, 
                    np.random.normal(-2, 1.5, N),  # From N(-2, 1.5^2)
                    np.random.normal(1, 1, N))    # From N(1, 1^2)
        U2 = np.random.normal(0, 0.1, N)
        U3 = np.random.normal(0, 1, N)
        return np.c_[U1, U2, U3]

    def label(self, X):
        X_sum = X[:, 0]+ X[:, 1]+X[:, 2]  # Sum X1 + X2 + X3 for each sample
        rho = np.mean(X_sum)  # Compute the average of X1 + X2 + X3 across all samples
        
        # Compute the logistic function for each sample
        p = (1 + np.exp(-2.5 * (X_sum / rho)))**-1
        
        # Sample Y from a Bernoulli distribution based on these probabilities
        Y = np.random.binomial(1, p)
        
        return Y

    def get_Jacobian(self):
        """
        Computes the Jacobian matrix for the linear SCM.
        """
        print("ERROR!!!!")

        thetas = [-0.99, 0.99, 0.99]

        return np.array([[1, 0, 0],
                         [thetas[0], 1, 0],
                         [thetas[1], thetas[2], 1]])


    def get_Jacobian_interv(self, interv_set):
        """
        Computes the Jacobian matrix under interventions.
        Parameters:
        - interv_set: A list of indices where interventions occur
        """
        # Get the standard Jacobian
        J = self.get_Jacobian()

        # Iterate over the intervention set
        for i in interv_set:
            # If the intervention is hard (not soft), modify the Jacobian
            if not self.soft_interv[i]:
                # Set upstream effects to zero (all previous columns in the row)
                for j in range(i):
                    J[i][j] = 0.0
        return J

class SCM_PCR_not_linear_estimated(SCM):
    """ Semi-synthetic ESTIMATED SCM inspired by https://arxiv.org/pdf/2006.06831, introduced by Karimi et al. """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Set the file path to save the results
        current_dir = os.getcwd()  # Get the current working directory
        result_folder = os.path.join(current_dir, "Results")
        result_file_path = os.path.join(result_folder, f'results_log.txt')

        with open(result_file_path, 'w') as f:

            print("\n######### ESTIMATED #########")
            f.write("######### ESTIMATED #########\n")
            folder_path = f"/home/tampieri/Results_SCM_Estimation_Trial_Non_Linear/"

            file_path = os.path.join(folder_path, 'final_results_Trial_Non_Linear.csv')
            
            # Read the CSV file into a pandas DataFrame
            try:
                data = pd.read_csv(file_path)
            except FileNotFoundError:
                print(f"File not found at path: {file_path}")
                data = None
        
            # Calculate the mean for each column
            thetas = data.mean().tolist()
            
            print("ESTIMATED THETAS", thetas)
            f.write(f"ESTIMATED THETAS: {thetas}\n")


            self.f = [lambda U0: U0 ,
                    lambda X0, U1: (thetas[0])*X0+ U1, 
                    lambda X0, X1, U2: thetas[1]*X0 +thetas[2]*X1+U2, 
                    ]

            self.inv_f = [lambda X: X[:, [0]],
                        lambda X: X[:, [1]] - thetas[0]*X[:, [0]],
                        lambda X: X[:, [2]] - thetas[1]*X[:, [0]] - thetas[2]*X[:, [1]],
                        ]


            scm_generated = functions_estimation_SCM.generate_scm(thetas)
            df=scm_generated.sample(n_samples=100)

            

            # Compute the mean of x1, x2, and x3
            x1_mean = np.mean(df["x1"])
            x2_mean = np.mean(df["x2"])
            x3_mean = np.mean(df["x3"])

            self.mean = torch.Tensor([x1_mean, x2_mean, x3_mean]) #mean of the etimated
            self.std = torch.Tensor([np.std(df["x1"]), np.std(df["x2"]), np.std(df["x3"])])

            print("MEAN: ",self.mean, " STD: ", self.std)

            f.write(f"MEAN: {self.mean}\n")
            f.write(f"STD: {self.std}\n")
            print()
            self.actionable = [0, 1, 2]
            self.soft_interv = [True, True, True]

    def sample_U(self, N):

        U1 = np.random.normal(0, 1, N)
        U2 = np.random.normal(0, 1, N)
        U3 = np.random.normal(0, 1, N)
        return np.c_[U1, U2, U3]

    def label(self, X):
        X_sum = X[:, 0]+ X[:, 1]+X[:, 2]  # Sum X1 + X2 + X3 for each sample
        rho = np.mean(X_sum)  # Compute the average of X1 + X2 + X3 across all samples
        
        # Compute the logistic function for each sample
        p = (1 + np.exp(-2.5 * (X_sum / rho)))**-1
        
        # Sample Y from a Bernoulli distribution based on these probabilities
        Y = np.random.binomial(1, p)
        
        return Y

    def get_Jacobian(self):
        """
        Computes the Jacobian matrix for the linear SCM.
        """

        folder_path = f"/home/tampieri/Results_Interventions_SD/IterationsResults_0927_2300/"

        file_path = os.path.join(folder_path, 'final_results_Interventions_10.csv')
        
        # Read the CSV file into a pandas DataFrame
        try:
            data = pd.read_csv(file_path)
        except FileNotFoundError:
            print(f"File not found at path: {file_path}")
            data = None
    
        # Calculate the mean for each column
        thetas = data.mean().tolist()
        return np.array([[1, 0, 0],
                         [thetas[0], 1, 0],
                         [thetas[1], thetas[2], 1]])


    def get_Jacobian_interv(self, interv_set):
        """
        Computes the Jacobian matrix under interventions.
        Parameters:
        - interv_set: A list of indices where interventions occur
        """
        # Get the standard Jacobian
        J = self.get_Jacobian()

        # Iterate over the intervention set
        for i in interv_set:
            # If the intervention is hard (not soft), modify the Jacobian
            if not self.soft_interv[i]:
                # Set upstream effects to zero (all previous columns in the row)
                for j in range(i):
                    J[i][j] = 0.0

        return J


def generate_SCM_data(id, N):
    """
    Return samples of the SCM (if synthetic), as well as information pertaining to the features (which ones are
    actionable, increasing, decreasing, and categorical)
    
    Inputs:     id: str, dataset id. One of 'German', 'Adult'.
                N: int, number of samples to draw (if the data set is synthetic).
                
    Outputs:    myscm: type SCM
                X: np.array (N, D) or None
                Y: np.array (N, ) or None
                actionable: list of ints, indices of the actionable features
                increasing: list of ints, indices of the features which can only be increased (actionability constrain)
                decreasing: list of ints, indices of the features which can only be decreased (actionability constrain)
                categorical: list of ints, indices of the features which are categorical (and thus not real-valued)
    """
    if id == 'German':
        myscm = SCM_Loan()
    elif id == 'Adult':
        myscm = Learned_Adult_SCM()
        myscm.load('scms/adult_scm')
    else:
        raise NotImplemented

    if id == 'German': # synthetic, generate the data
        X, Y = myscm.generate(N)
    else: # real world data set, no data returned
        X, Y = None, None

    actionable = myscm.getActionable()
    if id == 'German':
        increasing = [2]
        decreasing = []
        categorical = [0]
    elif id == 'Adult':
        increasing = [4, 5]
        decreasing = []
        categorical = [0, 1, 2]

    return myscm, X, Y, actionable, increasing, decreasing, categorical