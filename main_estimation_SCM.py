import time
import numpy as np
import zeus
from multiprocessing import Pool
import os
import csv
import functions_estimation_SCM


example_type = input("\nWhich set of thetas do you want to use? (test/paper): ").strip().lower()

# Check user input and initialize variables accordingly
if example_type == "test":
    TRUE_THETAS = [-0.99, 0.99, 0.99]
else:
    TRUE_THETAS = [-0.99, 0.05, 0.25]



focus_of_analysis = input("\nWhat's the focus of your analysis? (Trial/Interventions/Alpha/Epsilon): ")
TRUE_THETA_MAX = 1
TRUE_THETA_MIN = -1
noisy_thetas = [0.5, -0.5, -0.5]
INFO_REQUESTED = "full"
likelihood_version = "soft"
timeout = 400
nwalkers = 30 # Number of particles (they call them walkers)
nsteps = 50 # Number of steps/iterations.
ITERATIONS_NUMBER = 10



# Set some seed for reproducibility
np.random.seed(2024)

list_ = []

################## ANALYSIS ON: ALPHA, EPSILON, INTERVENTIONS


csv_file_path_yes = input(f'\nIs this file path for the results ok? "/home/tampieri/Results_SCM_Estimation_{focus_of_analysis}/" (yes/no): ')
if csv_file_path_yes == "yes":
    csv_file_path = f'/home/tampieri/Results_SCM_Estimation_{focus_of_analysis}/hyperparameters.csv'
else:
    csv_file_path = input(f'write your preferred file path:')


# Extract the directory from the file path
directory = os.path.dirname(csv_file_path)

# Check if the directory exists, if not, create it
os.makedirs(directory, exist_ok=True)

# Create the CSV file and write the parameters
with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    
    # Write headers
    writer.writerow(['Parameter', 'Value'])
    
    # Write the parameter values
    writer.writerow(['TRUE_THETAS', TRUE_THETAS])
    writer.writerow(['TRUE_THETA_MAX', TRUE_THETA_MAX])
    writer.writerow(['TRUE_THETA_MIN', TRUE_THETA_MIN])
    writer.writerow(['INFO_REQUESTED', INFO_REQUESTED])
    writer.writerow(['likelihood_version', likelihood_version])
    writer.writerow(['timeout', timeout])
    writer.writerow(['nwalkers', nwalkers])
    writer.writerow(['nsteps', nsteps])
    writer.writerow(['noisy_thetas', noisy_thetas])
    writer.writerow(['ITERATIONS_NUMBER', ITERATIONS_NUMBER])

    if focus_of_analysis == "Alpha":
        interventions_num = 10
        epsilon = 0.4
        list_ = np.arange(0.0, 1.0 + 0.1, 0.1).round(1).tolist()
        
        writer.writerow(['interventions_num', interventions_num])
        writer.writerow(['epsilon', epsilon])
        writer.writerow(['alpha_list', list_])


    if focus_of_analysis == "Epsilon":
        interventions_num = 10
        ALPHA = 0.1
        list_ = np.arange(0.3, 0.3 + 0.1, 0.1).round(1).tolist()

        writer.writerow(['interventions_num', interventions_num])
        writer.writerow(['epsilon_list', list_])
        writer.writerow(['alpha', ALPHA])

    if focus_of_analysis == "Interventions":
        ALPHA = 0.0
        epsilon = 0.6
        list_ = np.arange(7, 16, 1).round(1).tolist()

        writer.writerow(['interventions_list', list_])
        writer.writerow(['epsilon', epsilon])
        writer.writerow(['alpha', ALPHA])
    
    if focus_of_analysis == "Trial":
        ALPHA = 0.1
        epsilon = 0.5
        list_ = [10]

        writer.writerow(['interventions_list', list_])
        writer.writerow(['epsilon', epsilon])
        writer.writerow(['alpha', ALPHA])




# Final results list
final_results = []

# Simulate SCM and generate user info
simulation_scm = functions_estimation_SCM.generate_scm(TRUE_THETAS)

# Loop over interventions
for element in list_:


    if focus_of_analysis == "Alpha":
        ALPHA = element
    if focus_of_analysis == "Epsilon":
        epsilon = element
    if focus_of_analysis == "Interventions":
        interventions_num = element
    if focus_of_analysis == "Trial":
        interventions_num = element


    # Number of iterations
    for iteration_num in range(ITERATIONS_NUMBER):  # Loop over n iterations
        
        interventions = []

        for i in range(interventions_num):
            node = np.random.choice(["x1", "x2"], 1)[0]
            value = np.random.rand()
            interventions.append([node, value])



        values_real_scm = functions_estimation_SCM.generate_user_info(simulation_scm, TRUE_THETAS, interventions, ALPHA, INFO_REQUESTED, noisy_thetas)


        ndim = 3
        start = []
        start_time = time.time()

        while len(start) < nwalkers:
            current_time = time.time()

            # Timeout handling
            if current_time - start_time > timeout:
                print("Timeout reached, exiting the loops.")
                break

            tmp = np.random.uniform(TRUE_THETA_MIN, TRUE_THETA_MAX, ndim)

            if np.isfinite(functions_estimation_SCM.log_posterior(tmp, interventions, values_real_scm, TRUE_THETA_MIN, TRUE_THETA_MAX, epsilon, ALPHA, "soft", INFO_REQUESTED)):
                # print(f"FOUND {len(start)+1}/{nwalkers}")
                start.append(tmp)

        if current_time - start_time > timeout:
            break

        # Run the MCMC sampling
        with Pool() as pool:
            sampler = zeus.EnsembleSampler(nwalkers, ndim, functions_estimation_SCM.log_posterior, 
                                           args=[interventions, values_real_scm, TRUE_THETA_MIN, TRUE_THETA_MAX, epsilon, ALPHA, "soft", INFO_REQUESTED], 
                                           verbose=True, light_mode=True, pool=pool)
            sampler.run_mcmc(start, nsteps)
            sampler.summary


        chain_ = sampler.get_chain(flat=True, discard=0.7)
        estimated_theta = np.mean(chain_, axis=0)

        # Append the results to final_results
        final_results.append(estimated_theta)


    # Write the final results to a CSV file for this epsilon
    csv_filename = f"/home/tampieri/Results_SCM_Estimation_{focus_of_analysis}/final_results_{focus_of_analysis}.csv"
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["theta_1", "theta_2", "theta_3"])  # CSV header
        for result in final_results:
            writer.writerow(result)
    final_results = []
    print(f"\n --------------------------- \n\n Results for {focus_of_analysis} saved to {csv_filename}")



