import numpy as np
import pandas as pd
import networkx as nx
import seaborn as sns
import torch
from csm import StructuralCausalModel




def mixture_of_gaussians(generator, n_samples):
            """
            Generates samples from a mixture of two Gaussian distributions:
            - 50% of the samples are from N(-2, 1.5^2)
            - 50% of the samples are from N(1, 1^2)
            """
            # Randomly select the mixture component (0 or 1) for each sample
            mixture_component = generator.choice([0, 1], size=n_samples, p=[0.5, 0.5])
            
            # Generate samples from the two different Gaussian distributions
            samples = np.where(
                mixture_component == 0,
                generator.normal(loc=-2, scale=1.5, size=n_samples),  # From N(-2, 1.5^2)
                generator.normal(loc=1, scale=1, size=n_samples)      # From N(1, 1^2)
            )
            
            return samples

# Here I changed the theta definition to be a vector instead of a matrix. Basically, since most of the entries
# of the matrix are zero, we can just consider a "flatter" representation.
def generate_scm(thetas):
    # print("generate_scm")
    generator = np.random.default_rng(2024)

    if np.all(thetas == np.array([-0.99, 0.05, 0.25])):
        print("\n You picked thetas: ", thetas, " and a mixure of Gaussians")
            
        scm = StructuralCausalModel({
        "x1": lambda n_samples, thetas=thetas: mixture_of_gaussians(generator, n_samples),
        "x2": lambda x1, n_samples, thetas=thetas: thetas[0] * x1 + generator.normal(loc=0, scale=1, size=n_samples),
        "x3": lambda x2, x1, n_samples, thetas=thetas: thetas[1] * x1 + thetas[2] * x2 + generator.normal(loc=0,
                                                                                                        scale=1,
                                                                                                    size=n_samples),
    })
    elif np.all(thetas == np.array([0, 0, 0])):
        print("\n You picked non linear ANM: ", thetas, " and a mixure of Gaussians")
            
        scm = StructuralCausalModel({
        "x1": lambda n_samples, thetas=thetas: mixture_of_gaussians(generator, n_samples),
        "x2": lambda x1, n_samples, thetas=thetas:-1 +3/(1+ np.exp(-2*x1)) + generator.normal(loc=0, scale=0.1, size=n_samples),
        "x3": lambda x2, x1, n_samples, thetas=thetas: -0.05 * x1 + 0.25 * x2*x2 + generator.normal(loc=0,
                                                                                                        scale=1,
                                                                                                    size=n_samples),
    })
    else:

        scm = StructuralCausalModel({
            "x1": lambda n_samples, thetas=thetas: generator.normal(loc=0, scale=1, size=n_samples),
            "x2": lambda x1, n_samples, thetas=thetas: thetas[0] * x1 + generator.normal(loc=0, scale=1, size=n_samples),
            "x3": lambda x2, x1, n_samples, thetas=thetas: thetas[1] * x1 + thetas[2] * x2 + generator.normal(loc=0,
                                                                                                            scale=1,
                                                                                                        size=n_samples),
        })

    return scm

# def generate_scm_2(thetas):
#     # print("generate_scm")
#     generator = np.random.default_rng(2024)
#     np.random.seed(42)
#     scm = StructuralCausalModel({
#         "x1": lambda n_samples, thetas=thetas: torch.tensor(np.random.binomial(1, 0.5, size=n_samples), dtype=torch.float32),
#         "x2": lambda x1, n_samples, thetas=thetas: -35 + thetas[0]*x1 + torch.tensor(np.random.gamma(shape=10, scale=3.5, size=n_samples), dtype=torch.float32),
#         "x3": lambda x2, x1, n_samples, thetas=thetas: -0.5 + thetas[1] * x1 + thetas[2] * x2 + torch.normal(mean=0, std=0.25, size=(n_samples,)),
#         "x4": lambda x3, x2, x1, n_samples, thetas=thetas: 1  + x1 - 0.01 * (x2) + torch.normal(mean=0, std=4, size=(n_samples,)),
#         "x5": lambda x4, x3, x2, x1, n_samples, thetas=thetas: -1 + 0.1 * x2 + 2 * x1 + x4 + torch.normal(mean=0, std=9, size=(n_samples,)),
#         "x6": lambda x5, x4, x3, x2, x1, n_samples: -4 +0.1*35 + 0.1 * (x2) + 2 * x1 + 0.05 * x1 +  0.05 * x3 + torch.normal(mean=0, std=4, size=(n_samples,)),
#         "x7": lambda x6, x5, x4, x3, x2, x1, n_samples: -4 + 1.5 * x6 + torch.normal(mean=0, std=25, size=(n_samples,))
#     })


#     return scm

# %%
# I've changed this function to perform a "soft intervention" instead.
# Basically, given the intervention value v, we just add it to the variable (e.g., X = X + v).
# Previously, we were using a "hard intervention", which basically overwrite the value of the variable (e.g, X = v)
# by removing the corresponding parents edges.
# I had to change the StructuralCausalModel class (the sample() function) to enable soft interventions.
def intervened_data(scm, intervention, intervention_type):
    node, value = intervention
    # scm_do = scm.do(node)
    # We perform here the sampling of 100 instances to compute the expected value of the intervention E[X | do(X=X+v]
    ds_do = scm.sample(n_samples=100, set_values={node: np.full(100, value)}, type_of_intervention=intervention_type)
    return ds_do



# %%
# The prior is taken from zeus example. Basically, a particle is valid if and only
# if it lies between -5 and 5. If a particle k lies there, then P(\theta=k) = 1.
# Thus, the logprobability is 0, since np.log(1) = 0. However, if the particle
# is not in the correct range, the logprob is -infinity, since np.log(0) is undefined.
def log_prior(theta, TRUE_THETA_MIN, TRUE_THETA_MAX):
    if np.all(theta > TRUE_THETA_MIN) and np.all(theta < TRUE_THETA_MAX):
        return 0.0
    else:
        return -np.inf


# We consider a "hard" version of the likelihood function. Basically, given the user preferences
# we assign a positive probability only to those particles which matches **all** the ground truth choices.
# Basically, in the code below, estimated_result and ground_truth_choice must match always.
def likelihood(interventions, values_real_scm, estimated_thetas, epsilon, alpha_value, version, info_descendents):
    counter = []
    for intervention, scm_do_real in zip(interventions, values_real_scm):

        # generate the scm with the estimated thetas
        scm_temp = generate_scm(estimated_thetas)
        scm_do = intervened_data(scm_temp, intervention, "soft")
        
        node, value = intervention
        result = True

        if info_descendents == "partial":
            if np.isnan(scm_do_real[0]):
                delta = np.abs(np.mean(scm_do["x3"].values) - scm_do_real[1])
            else:
                delta = np.abs(np.mean(scm_do["x2"].values) - scm_do_real[0])
            
            if not (delta <= epsilon):
                result = False

        else:
            delta_2 = np.abs(np.mean(scm_do["x3"].values) - scm_do_real[1])

            # If the node is "x1", also calculate the difference for x2
            if node == "x1":
                delta_1 = np.abs(np.mean(scm_do["x2"].values) - scm_do_real[0])

                # print("np.mean(delta_1)", np.mean(delta_1))
                # Check if all values in both delta_1 and delta_2 are less than 0.1
                if not ((delta_1 <= epsilon) and (delta_2 <= epsilon)): # eventualmente si può mettere AND
                    result = False

            else:
                # Check if all values in delta_2 are less than 0.1
                if not (delta_2 <= epsilon):
                    result = False

        counter.append(result)

        
    # Calculate the likelihood based on the consistency count
    if version == "soft": 
        likelihood_value = np.sum(counter) / len(counter) >= (1 - alpha_value)
    else:
        likelihood_value = np.all(counter)
        
    #print("likelihood_value", likelihood_value)
    return likelihood_value



def log_posterior(thetas, interventions, values_real_scm, TRUE_THETA_MIN, TRUE_THETA_MAX, epsilon, alpha_value, version, info_descendents):
    log_lk = likelihood(interventions, values_real_scm, thetas, epsilon, alpha_value, version, info_descendents)
    log_lk = np.log(log_lk) if log_lk > 0 else -np.inf
    return log_lk + log_prior(thetas, TRUE_THETA_MIN, TRUE_THETA_MAX)



def generate_user_info(simulation_scm, TRUE_THETAS, interventions, ALPHA, info_descendents, noisy_thetas):
        
        noisy_scm = generate_scm(noisy_thetas)
        print("\n ################# True thetas", TRUE_THETAS, " and Noisy thetas ", noisy_thetas, " with alpha = ", ALPHA, " #################\n")



        values_real_scm = []

        for intervention in interventions:
            # Generate a random value between 0 and 1
            random_value = np.random.rand()

            # Choose the appropriate SCM based on the random value
            if random_value >= ALPHA:
                scm_do_real = intervened_data(simulation_scm, intervention, "soft")
            else:
                scm_do_real = intervened_data(noisy_scm, intervention, "soft")
            
            node, value = intervention

            mean_1 = np.nan  # Initialize with NaN
            mean_2 = np.mean(scm_do_real["x3"].values)
            # if (len(TRUE_THETAS) > 3):
            #     mean_3 = np.mean(scm_do_real["x4"].values)
            #     mean_4 = np.mean(scm_do_real["x5"].values)
            #     mean_5 = np.mean(scm_do_real["x6"].values)
            #     mean_6 = np.mean(scm_do_real["x7"].values)


            if node == "x1":
                mean_1 = np.mean(scm_do_real["x2"].values)
                if info_descendents == "partial":
                    if np.random.choice([True, False]):
                        mean_1 = np.nan
                    else:
                        mean_2 = np.nan

        
            # if (len(TRUE_THETAS) > 3):
            #     values_real_scm.append([mean_1, mean_2, mean_3, mean_4, mean_5, mean_6])
            # else:
            values_real_scm.append([mean_1, mean_2])


        return values_real_scm