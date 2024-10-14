import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
import recourse_interventions_plots_function

# Set up Seaborn and Matplotlib with LaTeX-style formatting

sns.set(
    style="ticks",
    rc={
        "font.family": "serif",
    }
)

def initial_setting():
    error_type = "Standard Error"
    name_of_the_folder = input("Write the name of the folder: ")

    # Step 1: Read the CSV file into a pandas DataFrame
    df = pd.read_csv(f'{name_of_the_folder}/results_summary.csv')

    # Get the values for epsilon and true_thetas from the DataFrame
    # Assuming you want to use the first row's values. Adjust this if necessary.
    epsilon = df['epsilon'].iloc[0]
    true_thetas = df['true_thetas'].iloc[0]
    return df, name_of_the_folder, error_type, epsilon, true_thetas



def recourse_costs_plot(df, name_of_the_folder, error_type, epsilon, true_thetas):
    # Step 2: Group the data by 'learning_rate' and calculate the mean and standard deviation for cost_mean, cost_mean_estimated, cost_mean_baseline
    mean_values = df.groupby('learning_rate')[['cost_mean', 'cost_mean_estimated', 'cost_mean_baseline']].mean()
    std_values = df.groupby('learning_rate')[['cost_mean', 'cost_mean_estimated', 'cost_mean_baseline']].std()

    if error_type == "Standard Error":
        # Calculate the sample size (n) for each group
        sample_size = df.groupby('learning_rate')[['cost_mean', 'cost_mean_estimated', 'cost_mean_baseline']].count()
        # Calculate Standard Error (SE) by dividing the standard deviation by the square root of the sample size
        std_values = std_values / np.sqrt(sample_size)

    # Extract the relevant columns (mean and standard deviation values)
    learning_rates = mean_values.index
    cost_mean = mean_values['cost_mean']
    cost_mean_estimated = mean_values['cost_mean_estimated']
    cost_mean_baseline = mean_values['cost_mean_baseline']

    cost_mean_std = std_values['cost_mean']
    cost_mean_estimated_std = std_values['cost_mean_estimated']
    cost_mean_baseline_std = std_values['cost_mean_baseline']

    x = np.arange(len(learning_rates))  # The positions of the learning rates
    width = 0.2  # Width of the bars, reduced to accommodate 3 bars

    # Step 4: Create the plot with 3 bars per learning rate (mean values) with transparency
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plotting the three bars for each learning rate (mean values)
    bar1 = ax.bar(x - width, cost_mean, width, label='True SCM Cost', color='royalblue', alpha=0.7)
    bar2 = ax.bar(x, cost_mean_estimated, width, label='Estimated SCM Cost', color='tomato', alpha=0.7)
    bar3 = ax.bar(x + width, cost_mean_baseline, width, label='Prior SCM Cost', color='seagreen', alpha=0.7)

    # Step 5: Customize the plot with enhanced error bars
    errorbar_color = 'darkgray'  # Set error bar color

    # Adding error bars separately to customize their appearance
    ax.errorbar(x - width, cost_mean, yerr=cost_mean_std, fmt='none', ecolor=errorbar_color, 
                elinewidth=2, capsize=8, capthick=2, linestyle='--', alpha=0.9)  # Error bars for True SCM

    ax.errorbar(x, cost_mean_estimated, yerr=cost_mean_estimated_std, fmt='none', ecolor=errorbar_color, 
                elinewidth=2, capsize=8, capthick=2, linestyle='--', alpha=0.9)  # Error bars for Estimated SCM

    ax.errorbar(x + width, cost_mean_baseline, yerr=cost_mean_baseline_std, fmt='none', ecolor=errorbar_color, 
                elinewidth=2, capsize=8, capthick=2, linestyle='--', alpha=0.9)  # Error bars for Prior SCM

    # Step 6: Customize the plot labels, title, and appearance
    ax.set_xlabel(f'Learning Rate', fontsize=16)
    ax.set_ylabel(f'Mean Cost', fontsize=16)
    ax.set_title(f'Recourse Cost Comparison: True, Estimated, and Prior SCMs', fontsize=18)

    ax.set_xticks(x)
    ax.set_xticklabels(learning_rates, fontsize=14)  # Set the learning rates as x-tick labels
    ax.legend(fontsize=12)

    # Add a grid for better readability
    ax.grid(True, linestyle='--', alpha=0.7)

    # Set y-axis limit to make the graph more readable
    ax.set_ylim(0, max(cost_mean.max(), cost_mean_estimated.max(), cost_mean_baseline.max()) * 1.1)

    # Increase the font size of the y-ticks and x-ticks
    ax.tick_params(axis='both', which='major', labelsize=14)

    # Step 7: Add data labels on the bars
    def add_labels(bars):
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, yval + 0.06, round(yval, 2), ha='center', va='bottom', fontsize=10)

    add_labels(bar1)
    add_labels(bar2)
    add_labels(bar3)

    # Step 8: Save the plot as a .pdf file with LaTeX formatting
    fig.tight_layout()

    # Create a filename based on epsilon and true_thetas
    output_filename = f"{name_of_the_folder}/cost_comparison_plot_eps_{epsilon}_thetas_{true_thetas}.pdf"
    fig.savefig(output_filename, format="pdf", bbox_inches='tight')
    output_filename = f"{name_of_the_folder}/cost_comparison_plot_eps_{epsilon}_thetas_{true_thetas}.png"
    fig.savefig(output_filename, format="png", bbox_inches='tight')

    # Optional: Inform the user where the image is saved
    print(f"Plot saved as {output_filename}")

    # Show the plot (optional, can be commented out if you just want to save the plot)
    plt.show()

def recourse_effectiveness_plot(df, name_of_the_folder, error_type, epsilon, true_thetas):
    # Step 2: Group the data by 'learning_rate' and calculate the mean and standard deviation for accuracy_true and accuracy_preds
    mean_values = df.groupby('learning_rate')[['accuracy_true', 'accuracy_preds', 'accuracy_preds_prior']].mean()*100
    std_values = df.groupby('learning_rate')[['accuracy_true', 'accuracy_preds', 'accuracy_preds_prior']].std()*100

    if error_type == "Standard Error":
        # Calculate the sample size (n) for each group
        sample_size = df.groupby('learning_rate')[['accuracy_true', 'accuracy_preds',  'accuracy_preds_prior']].count()
        # Calculate Standard Error (SE) by dividing the standard deviation by the square root of the sample size
        std_values = std_values / np.sqrt(sample_size)

    # Extract the relevant columns (mean and standard deviation values)
    learning_rates = mean_values.index
    accuracy_true_mean = mean_values['accuracy_true']
    accuracy_preds_mean = mean_values['accuracy_preds']
    accuracy_preds_mean_prior = mean_values['accuracy_preds_prior']

    accuracy_true_std = std_values['accuracy_true']
    accuracy_preds_std = std_values['accuracy_preds']
    accuracy_preds_prior_std = std_values['accuracy_preds_prior']

    x = np.arange(len(learning_rates))  # The positions of the learning rates
    width = 0.2  # Width of the bars, reduced to accommodate 3 bars

    # Step 4: Create the plot with 3 bars per learning rate (mean values) with transparency
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plotting the three bars for each learning rate (mean values)
    bar1 = ax.bar(x - width, accuracy_true_mean, width, label='Ground Truth SCM', color='royalblue', alpha=0.7)
    bar2 = ax.bar(x, accuracy_preds_mean, width, label='Estimated SCM', color='tomato', alpha=0.7)
    bar3 = ax.bar(x + width, accuracy_preds_mean_prior, width, label='Prior SCM', color='seagreen', alpha=0.7)

    # Step 5: Customize the plot with enhanced error bars
    errorbar_color = 'darkgray'  # Set error bar color

    # Adding error bars separately to customize their appearance
    ax.errorbar(x - width, accuracy_true_mean, yerr=accuracy_true_std, fmt='none', ecolor=errorbar_color, 
                elinewidth=2, capsize=8, capthick=2, linestyle='--', alpha=0.9)  # Error bars for Ground Truth SCM

    ax.errorbar(x, accuracy_preds_mean, yerr=accuracy_preds_std, fmt='none', ecolor=errorbar_color, 
                elinewidth=2, capsize=8, capthick=2, linestyle='--', alpha=0.9)  # Error bars for Estimated SCM

    ax.errorbar(x + width, accuracy_preds_mean_prior, yerr=accuracy_preds_prior_std, fmt='none', ecolor=errorbar_color, 
                elinewidth=2, capsize=8, capthick=2, linestyle='--', alpha=0.9)  # Error bars for Prior SCM

    # Step 6: Customize the plot labels, title, and appearance
    ax.set_xlabel('Learning Rate', fontsize=16)
    ax.set_ylabel('Average Recourse Effectiveness', fontsize=16)
    ax.set_title(f'Recourse Effectiveness: GT, Estimated and Prior SCMs for Different LR ($\epsilon ={epsilon}$)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(learning_rates, fontsize=14)  # Set the learning rates as x-tick labels
    ax.legend(fontsize=10)

    # Add a grid for better readability
    ax.grid(True, linestyle='--', alpha=0.7)

    # Set y-axis limit from 0 to 1.0
    ax.set_ylim(0, 100)

    # Increase the font size of the y-ticks and x-ticks
    ax.tick_params(axis='both', which='major', labelsize=14)

    # Step 7: Add data labels on the bars
    def add_labels(bars):
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, yval + 0.1, f'{round(yval, 1)}%', ha='center', va='bottom', fontsize = 10)

    add_labels(bar1)
    add_labels(bar2)
    add_labels(bar3)

    # Step 8: Save the plot as a .pdf file with LaTeX formatting
    fig.tight_layout()
    output_path_pdf = f"{name_of_the_folder}/validity_plot_eps_{epsilon}_thetas_{true_thetas}.png"
    fig.savefig(output_path_pdf)
    output_path_pdf = f"{name_of_the_folder}/validity_plot_eps_{epsilon}_thetas_{true_thetas}.pdf"
    fig.savefig(output_path_pdf, format="pdf", bbox_inches='tight')


    # Optional: Inform the user where the image is saved
    print(f"Plot saved as {output_path_pdf}")

    # Show the plot (optional, can be commented out if you just want to save the plot)
    plt.show()


# Example

df, name_of_the_folder, error_type, epsilon, true_thetas = initial_setting()
recourse_costs_plot(df, name_of_the_folder, error_type, epsilon, true_thetas)
recourse_effectiveness_plot(df, name_of_the_folder, error_type, epsilon, true_thetas)
recourse_interventions_plots_function.recourse_interventions_plot(name_of_the_folder, epsilon, true_thetas)




