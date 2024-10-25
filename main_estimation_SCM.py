import time
import numpy as np
import zeus
from multiprocessing import Pool
import os
import csv
import functions_estimation_SCM
import matplotlib.pyplot as plt
import seaborn as sns

# Set the seaborn style
sns.set(
    style="ticks",
    rc={
        "font.family": "serif",
    }
)

def plot_convergence_estimation(sampler, estimated_theta, true_data, TRUE_THETA_MIN, TRUE_THETA_MAX, epsilon, interventions_num, ALPHA_, iteration_num, focus_of_analysis):
    scm_estimated = functions_estimation_SCM.generate_scm(estimated_theta)
    estm_data = scm_estimated.sample(1000)
    samples = sampler.get_chain()
    
    # Define the path where the image will be saved
    save_path = f"Results_SCM_Estimation_{focus_of_analysis}"
    os.makedirs(save_path, exist_ok=True)  # Create the directory if it doesn't exist

    # Plot settings
    fig, ax = plt.subplots(3, 2, figsize=(12, 10))
    colors = sns.color_palette("husl", 3)  # Use a color palette for distinct colors

    # For each node, plot the convergence of the particles
    for id in range(3):
        if id == 2:
            # Plot convergence for θ_{32}
            ax[id, 0].plot(samples[:, :, id], alpha=0.6, linewidth=1.5, color=colors[id])
            ax[id, 0].set_ylabel(r"$\theta_{32}$", fontsize=14)        
            ax[id, 0].set_title(r"Convergence for $θ_{32}$", fontsize=16)
        else:
            # Plot convergence for θ_{21} and θ_{31}
            ax[id, 0].plot(samples[:, :, id], alpha=0.6, linewidth=1.5, color=colors[id + 1])
            ax[id, 0].set_ylabel(r"$\theta_{" + str(id + 2) + "1}$", fontsize=14)
            ax[id, 0].set_title(r"Convergence for $\theta_{" + str(id + 2) + "1}$", fontsize=16)


        ax[id, 0].set_ylim(TRUE_THETA_MIN - 0.5, TRUE_THETA_MAX + 0.5)
        ax[id, 0].set_xlabel("Iterations", fontsize=14)
        
        ax[id, 0].grid(True, linestyle='--', alpha=0.7)
        
        # Plot KDE for estimated vs. true data
        sns.kdeplot(estm_data[f"x{id+1}"], ax=ax[id, 1], color=colors[id], label=f'$X_{id+1}$', linewidth=2)
        sns.kdeplot(true_data[f"x{id+1}"], ls='--', ax=ax[id, 1], color='black', label='Truth', linewidth=2)
        ax[id, 1].set_title(f"KDE for $X_{id+1}$", fontsize=16)
        ax[id, 1].set_xlabel(f"$X_{id+1}$ Values", fontsize=14)
        ax[id, 1].set_ylabel("Density", fontsize=14)
        ax[id, 1].grid(True, linestyle='--', alpha=0.7)
        ax[id, 1].legend(fontsize=14)


    # Show the plot with estimated_theta and TRUE_THETAS
    fig.suptitle(fr"Results of MCMC (Estimated: {list(np.round(estimated_theta,2))}, $\sigma$ = {epsilon}, α = {ALPHA_})", fontsize=16)

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # Adjust title to fit

    # Save the plot as a PNG file
    file_name = f"convergence_{TRUE_THETA_MIN}_and_{TRUE_THETA_MAX}_epsilon_{epsilon}_inter_{interventions_num}_alpha_{ALPHA_}_TRUE_THETAS_{TRUE_THETAS}_{iteration_num}.png"
    full_path = os.path.join(save_path, file_name)
    plt.savefig(full_path, dpi=300)  # Save with high resolution
    file_name = f"convergence_{TRUE_THETA_MIN}_and_{TRUE_THETA_MAX}_epsilon_{epsilon}_inter_{interventions_num}_alpha_{ALPHA_}_TRUE_THETAS_{TRUE_THETAS}_{iteration_num}.pdf"
    full_path = os.path.join(save_path, file_name)
    plt.savefig(full_path, dpi=300)  # Save with high resolution

    print(f"Plot saved successfully at {full_path}")




example_type = input("\nWhich set of thetas do you want to use? (test/paper): ").strip().lower()

# Check user input and initialize variables accordingly
if example_type == "test":
    TRUE_THETAS = [-0.99, 0.99, 0.99]
elif example_type == "paper":
    TRUE_THETAS = [-0.99, 0.05, 0.25]
else:
    TRUE_THETAS = [0, 0, 0]





focus_of_analysis = input("\nWhat's the focus of your analysis? (Trial/Trial_Non_Linear/Interventions/Alpha/Epsilon): ")
TRUE_THETA_MAX = 1
TRUE_THETA_MIN = -1
noisy_thetas = [0.5, -0.5, -0.5]
INFO_REQUESTED = "full"
likelihood_version = "soft"
timeout = 400
nwalkers = 50 # Number of particles (they call them walkers)
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
        epsilon = 0.7
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
        epsilon = 0.4
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

    if focus_of_analysis == "Trial_Non_Linear":
        ALPHA = 0.1
        epsilon = 0.6
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
    if focus_of_analysis == "Trial_Non_Linear":
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


        
        true_data = simulation_scm.sample(1000)
        plot_convergence_estimation(sampler, estimated_theta, true_data, TRUE_THETA_MIN, TRUE_THETA_MAX,epsilon, interventions_num, ALPHA, iteration_num, focus_of_analysis)

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



