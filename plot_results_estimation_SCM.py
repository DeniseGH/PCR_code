import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_results_general(TRUE_THETAS, focus_of_analysis, folder_path, error_type="SE"):
    """
    Plots the mean and error (standard deviation or standard error) for different interventions.

    Args:
    - TRUE_THETAS: list of true theta values to compare against.
    - focus_of_analysis: The focus of analysis (e.g., "Interventions").
    - error_type: "SD" for standard deviation, "SE" for standard error.
    - folder_path: Path where the CSV files are stored and where the plot will be saved.
    """

    # Dynamically find all files that match the pattern
    files = os.listdir(folder_path)
    pattern = rf"final_results_{focus_of_analysis}_(\d+)\.csv"
    list_ = sorted([int(re.search(pattern, f).group(1)) for f in files if re.search(pattern, f)])

    # Initialize lists to store the results for plotting
    means_theta_1 = []
    means_theta_2 = []
    means_theta_3 = []
    means_theta_1.append(np.abs(0 - TRUE_THETAS[0]))
    means_theta_2.append(np.abs(0 - TRUE_THETAS[1]))
    means_theta_3.append(np.abs(0 - TRUE_THETAS[2]))

    errors_theta_1 = []
    errors_theta_2 = []
    errors_theta_3 = []
    errors_theta_1.append(0)
    errors_theta_2.append(0)
    errors_theta_3.append(0)

    # Loop over all elements from the dynamically generated list
    for element in list_:
        # Construct the file path
        file_path = os.path.join(folder_path, f"final_results_{focus_of_analysis}_{element}.csv")
        
        # Read the CSV file
        data = pd.read_csv(file_path)
        
        # Z-score threshold for outlier detection
        z_threshold = 2

        # Calculate Z-scores for each column
        z_scores = np.abs((data - data.mean()) / data.std())

        # Identify rows where any of the Z-scores exceed the threshold
        outliers = (z_scores > z_threshold).any(axis=1)

        # Filter out the outliers
        data = data[~outliers]

        # Compute the mean for each column
        mean_theta_1 = np.abs(data["theta_1"].mean() - TRUE_THETAS[0])
        mean_theta_2 = np.abs(data["theta_2"].mean() - TRUE_THETAS[1])
        mean_theta_3 = np.abs(data["theta_3"].mean() - TRUE_THETAS[2])
        
        # Calculate the sample size (used for standard error calculation)
        n = len(data)

        # Depending on the error_type, compute the standard deviation or standard error
        if error_type == "SD":
            error_theta_1 = data["theta_1"].std()
            error_theta_2 = data["theta_2"].std()
            error_theta_3 = data["theta_3"].std()
        elif error_type == "SE":
            error_theta_1 = data["theta_1"].std() / np.sqrt(n)
            error_theta_2 = data["theta_2"].std() / np.sqrt(n)
            error_theta_3 = data["theta_3"].std() / np.sqrt(n)
        else:
            raise ValueError("error_type must be either 'SD' or 'SE'.")

        # Store the computed values in the corresponding lists
        means_theta_1.append(mean_theta_1)
        means_theta_2.append(mean_theta_2)
        means_theta_3.append(mean_theta_3)
        
        errors_theta_1.append(error_theta_1)
        errors_theta_2.append(error_theta_2)
        errors_theta_3.append(error_theta_3)
    
    # Plotting the results
    plt.figure(figsize=(12, 8))
    
    # X-axis is the intervention list
    x = [0.0] + list_

    # Plot for theta_1 with shaded area for SD or SE
    plt.plot(x, means_theta_1, '-o', label=r'$\theta_{21}$')
    plt.fill_between(x, 
                     np.array(means_theta_1) - np.array(errors_theta_1), 
                     np.array(means_theta_1) + np.array(errors_theta_1), 
                     color='blue', alpha=0.1)

    # Plot for theta_2 with shaded area for SD or SE
    plt.plot(x, means_theta_2, '-o', label=r'$\theta_{31}$')
    plt.fill_between(x, 
                     np.array(means_theta_2) - np.array(errors_theta_2), 
                     np.array(means_theta_2) + np.array(errors_theta_2), 
                     color='orange', alpha=0.1)

    # Plot for theta_3 with shaded area for SD or SE
    plt.plot(x, means_theta_3, '-o', label=r'$\theta_{32}$')
    plt.fill_between(x, 
                     np.array(means_theta_3) - np.array(errors_theta_3), 
                     np.array(means_theta_3) + np.array(errors_theta_3), 
                     color='green', alpha=0.1)
    
    # Adding labels and title
    font_size = 18
    plt.xticks(x, fontsize=font_size)
    plt.xlabel(f'{focus_of_analysis}', fontsize=font_size)
    plt.ylabel('Mean and Standard Error' if error_type == "SE" else 'Mean and SD', fontsize=font_size)
    plt.title(f'Mean and {error_type} of ' 
          r'$\theta_{21}$, $\theta_{31}$, $\theta_{32}$ for Different Values of '
          f'{focus_of_analysis}', fontsize=20)

    plt.legend(fontsize=font_size)

    # Save the plot to the folder
    plot_filename = os.path.join(folder_path, f"theta_mean_and_{error_type.lower()}_shaded_plot_{focus_of_analysis}.png")
    plt.savefig(plot_filename)
    plot_filename = os.path.join(folder_path, f"theta_mean_and_{error_type.lower()}_shaded_plot_{focus_of_analysis}.pdf")
    plt.savefig(plot_filename)
    print(f"Plot saved to {plot_filename}")
    
    # Show the plot
    plt.grid(True)
    plt.show()




# Example usage
folder_path=f"/home/tampieri/SCM_Estimation_Interventions_Test_Thesis/"
TRUE_THETAS = [-0.99, 0.99, 0.99]
focus_of_analysis = "Interventions"
plot_results_general(TRUE_THETAS, focus_of_analysis, folder_path)
