import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
import recourse_interventions_plots_function

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


def recourse_costs_plot(ax, df, epsilon, true_thetas):
    # Step 2: Group the data by 'learning_rate' and calculate the mean and standard deviation
    mean_values = df.groupby('learning_rate')[['cost_mean', 'cost_mean_estimated', 'cost_mean_baseline']].mean()
    std_values = df.groupby('learning_rate')[['cost_mean', 'cost_mean_estimated', 'cost_mean_baseline']].std()

    learning_rates = mean_values.index
    cost_mean = mean_values['cost_mean']
    cost_mean_estimated = mean_values['cost_mean_estimated']
    cost_mean_baseline = mean_values['cost_mean_baseline']

    cost_mean_std = std_values['cost_mean']
    cost_mean_estimated_std = std_values['cost_mean_estimated']
    cost_mean_baseline_std = std_values['cost_mean_baseline']

    x = np.arange(len(learning_rates))  # The positions of the learning rates
    width = 0.2  # Width of the bars

    # Plotting the three bars for each learning rate (mean values)
    ax.bar(x - width, cost_mean, width, label='True SCM Cost', color='royalblue', alpha=0.7)
    ax.bar(x, cost_mean_estimated, width, label='Estimated SCM Cost', color='tomato', alpha=0.7)
    ax.bar(x + width, cost_mean_baseline, width, label='Prior SCM Cost', color='seagreen', alpha=0.7)

    # Adding error bars
    errorbar_color = 'darkgray'
    ax.errorbar(x - width, cost_mean, yerr=cost_mean_std, fmt='none', ecolor=errorbar_color, capsize=5)
    ax.errorbar(x, cost_mean_estimated, yerr=cost_mean_estimated_std, fmt='none', ecolor=errorbar_color, capsize=5)
    ax.errorbar(x + width, cost_mean_baseline, yerr=cost_mean_baseline_std, fmt='none', ecolor=errorbar_color, capsize=5)

    # Customize labels, title, and appearance
    ax.set_xlabel('Learning Rate', fontsize=22)
    ax.set_ylabel('Mean Cost', fontsize=22)
    ax.set_xticks(x)
    ax.set_xticklabels(learning_rates, fontsize=18)
    ax.tick_params(axis='y', labelsize=18)
    ax.legend(fontsize=18)
    ax.grid(True, linestyle='--', alpha=0.7)

    #ax.set_title(f'Recourse Cost Comparison\n($\epsilon = {epsilon}$)', fontsize=20)
    if true_thetas[2] == 0.99:
        ax.set_title(f'Recourse Cost of $\mathcal{{M}}_1$\nfor Different LR ($\epsilon ={epsilon}$)', fontsize=24)
    elif true_thetas[2] == 0.25:
        ax.set_title(f'Recourse Cost of $\mathcal{{M}}_2$\nfor Different LR ($\epsilon ={epsilon}$)', fontsize=24)
    else:
        ax.set_title(f'Recourse Cost of $\mathcal{{M}}_3$\nfor Different LR ($\epsilon ={epsilon}$)', fontsize=24)


def recourse_effectiveness_plot(ax, df, epsilon, true_thetas):
    # Step 2: Group the data by 'learning_rate' and calculate the mean and standard deviation
    mean_values = df.groupby('learning_rate')[['accuracy_true', 'accuracy_preds', 'accuracy_preds_prior']].mean() * 100
    std_values = df.groupby('learning_rate')[['accuracy_true', 'accuracy_preds', 'accuracy_preds_prior']].std() * 100

    learning_rates = mean_values.index
    accuracy_true_mean = mean_values['accuracy_true']
    accuracy_preds_mean = mean_values['accuracy_preds']
    accuracy_preds_mean_prior = mean_values['accuracy_preds_prior']

    accuracy_true_std = std_values['accuracy_true']
    accuracy_preds_std = std_values['accuracy_preds']
    accuracy_preds_prior_std = std_values['accuracy_preds_prior']

    x = np.arange(len(learning_rates))  # The positions of the learning rates
    width = 0.2  # Width of the bars

    # Plotting the three bars for each learning rate (mean values)
    ax.bar(x - width, accuracy_true_mean, width, label='Ground Truth SCM', color='royalblue', alpha=0.7)
    ax.bar(x, accuracy_preds_mean, width, label='Estimated SCM', color='tomato', alpha=0.7)
    ax.bar(x + width, accuracy_preds_mean_prior, width, label='Prior SCM', color='seagreen', alpha=0.7)

    # Adding error bars
    errorbar_color = 'darkgray'
    ax.errorbar(x - width, accuracy_true_mean, yerr=accuracy_true_std, fmt='none', ecolor=errorbar_color, capsize=5)
    ax.errorbar(x, accuracy_preds_mean, yerr=accuracy_preds_std, fmt='none', ecolor=errorbar_color, capsize=5)
    ax.errorbar(x + width, accuracy_preds_mean_prior, yerr=accuracy_preds_prior_std, fmt='none', ecolor=errorbar_color, capsize=5)

    # Customize labels, title, and appearance
    ax.set_xlabel('Learning Rate', fontsize=22)
    ax.set_ylabel('Recourse Effectiveness (%)', fontsize=22)
    ax.set_xticks(x)
    ax.set_xticklabels(learning_rates, fontsize=18)
    ax.legend(fontsize=18)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.tick_params(axis='y', labelsize=18)

    #ax.set_title(f'Recourse Effectiveness Comparison\n($\epsilon = {epsilon}$)', fontsize=20)
    if true_thetas[2] == 0.99:
        ax.set_title(f'Recourse Effectiveness of $\mathcal{{M}}_1$\nfor Different LR ($\epsilon ={epsilon}$)', fontsize=24)
    if true_thetas[2] == 0.25:
        ax.set_title(f'Recourse Effectiveness of $\mathcal{{M}}_2$\nfor Different LR ($\epsilon ={epsilon}$)', fontsize=24)
    else:
        ax.set_title(f'Recourse Effectiveness of $\mathcal{{M}}_3$ \nfor Different ($\epsilon ={epsilon}$)', fontsize=24)
        
    


def create_combined_plots(df, name_of_the_folder, epsilon, true_thetas):
    fig, axs = plt.subplots(1, 2, figsize=(20, 10))  # Create a side-by-side plot with two subplots

    # Generate the plots
    
    recourse_effectiveness_plot(axs[0], df, epsilon, true_thetas)
    recourse_costs_plot(axs[1], df, epsilon, true_thetas)

    # Save the combined plot as a high-resolution PDF
    output_filename_pdf = f"{name_of_the_folder}/combined_plots_eps_{epsilon}_thetas_{true_thetas}.pdf"
    fig.tight_layout()
    fig.savefig(output_filename_pdf, format="pdf", bbox_inches='tight')
    output_filename_pdf = f"{name_of_the_folder}/combined_plots_eps_{epsilon}_thetas_{true_thetas}.png"
    fig.tight_layout()
    fig.savefig(output_filename_pdf, format="png", bbox_inches='tight')

    # Show the combined plot
    plt.show()

    print(f"Combined plots saved as {output_filename_pdf}")

# Example usage
df, name_of_the_folder, error_type, epsilon, true_thetas = initial_setting()
create_combined_plots(df, name_of_the_folder, epsilon, true_thetas)
recourse_interventions_plots_function.recourse_interventions_plot_v1(name_of_the_folder, epsilon, true_thetas)
