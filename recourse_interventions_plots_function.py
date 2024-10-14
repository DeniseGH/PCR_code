import ast
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set the seaborn style
sns.set(
    style="ticks",
    rc={
        "font.family": "serif",
    }
)

# Helper function to clean tensor strings
def clean_tensor_string(tensor_str):
    return tensor_str.replace('tensor(', '').replace(')', '')

def interventions_non_zero(df):
    # Initialize an empty DataFrame to store the non-zero counts for each iteration
    non_zero_counts_df = pd.DataFrame(columns=[
        'Iteration', 'lr', 'x_1_estimated', 'x_2_estimated', 'x_3_estimated', 
        'x_1_prior', 'x_2_prior', 'x_3_prior', 'x_1_true', 'x_2_true', 'x_3_true'
    ])

    for i in range(0, 30):
        ######### ESTIMATED
        # Clean up the tensor representation and safely evaluate the string for estimated interventions
        intervention_estimated = clean_tensor_string(df.iloc[i, 1])
        intervention_estimated_list = ast.literal_eval(intervention_estimated)
        
        # Convert the list into a DataFrame for estimated interventions
        df_intervention_estimated = pd.DataFrame(intervention_estimated_list, columns=['x_1_estimated', 'x_2_estimated', 'x_3_estimated'])

        # Count non-zero values per column for estimated interventions
        non_zero_counts_estimated = (df_intervention_estimated != 0).sum()/len(df_intervention_estimated)
        
        ######### PRIOR
        # Clean up the tensor representation and safely evaluate the string for prior interventions
        intervention_prior = clean_tensor_string(df.iloc[i, 2])
        intervention_prior_list = ast.literal_eval(intervention_prior)
        
        # Convert the list into a DataFrame for prior interventions
        df_intervention_prior = pd.DataFrame(intervention_prior_list, columns=['x_1_prior', 'x_2_prior', 'x_3_prior'])

        # Count non-zero values per column for prior interventions
        non_zero_counts_prior = (df_intervention_prior != 0).sum()/len(df_intervention_prior)

        ######### TRUE
        # Clean up the tensor representation and safely evaluate the string for true interventions
        intervention_true = clean_tensor_string(df.iloc[i, 3])
        intervention_true_list = ast.literal_eval(intervention_true)
        
        # Convert the list into a DataFrame for true interventions
        df_intervention_true = pd.DataFrame(intervention_true_list, columns=['x_1_true', 'x_2_true', 'x_3_true'])
        
        # Count non-zero values per column for true interventions
        non_zero_counts_true = (df_intervention_true != 0).sum()/len(df_intervention_true)
        
        # Create a new DataFrame row for the current iteration's non-zero counts
        new_row = pd.DataFrame({
            'Iteration': [i],
            'lr': [df.iloc[i, 0]],  # Making 'lr' consistent with the others by placing it in a list
            'x_1_estimated': [non_zero_counts_estimated['x_1_estimated']],
            'x_2_estimated': [non_zero_counts_estimated['x_2_estimated']],
            'x_3_estimated': [non_zero_counts_estimated['x_3_estimated']],
            'x_1_prior': [non_zero_counts_prior['x_1_prior']],
            'x_2_prior': [non_zero_counts_prior['x_2_prior']],
            'x_3_prior': [non_zero_counts_prior['x_3_prior']],
            'x_1_true': [non_zero_counts_true['x_1_true']],
            'x_2_true': [non_zero_counts_true['x_2_true']],
            'x_3_true': [non_zero_counts_true['x_3_true']]
        })
        
        # Concatenate the new row to the non_zero_counts_df
        non_zero_counts_df = pd.concat([non_zero_counts_df, new_row], ignore_index=True)

    # Display the final DataFrame with non-zero counts per iteration
    # print(non_zero_counts_df)

    return non_zero_counts_df


def recourse_interventions_plot(name_of_the_folder, epsilon, true_thetas):

    df = pd.read_csv(f'{name_of_the_folder}/interventions_analysis.csv')



    non_zero_counts_df = interventions_non_zero(df)
    new_column_order = ['lr', 
                        'x_1_true', 'x_1_prior', 'x_1_estimated', 
                        'x_2_true', 'x_2_prior', 'x_2_estimated', 
                        'x_3_true', 'x_3_prior', 'x_3_estimated']

    # Reordering the DataFrame
    non_zero_counts_df = non_zero_counts_df[new_column_order]


    # Initialize the directory for results
    current_dir = os.getcwd()  # Get the current working directory
    result_folder = os.path.join(current_dir, f'{name_of_the_folder}')
    os.makedirs(result_folder, exist_ok=True)  # Create the folder if it doesn't exist

    # Group by learning rate and calculate mean counts
    mean_counts_df = non_zero_counts_df.groupby('lr')[['x_1_true', 'x_1_prior', 'x_1_estimated', 
                        'x_2_true', 'x_2_prior', 'x_2_estimated', 
                        'x_3_true', 'x_3_prior', 'x_3_estimated']].mean()
    print(mean_counts_df)
    # Calculate standard error
    std_counts_df = non_zero_counts_df.groupby('lr')[['x_1_true', 'x_1_prior', 'x_1_estimated', 
                        'x_2_true', 'x_2_prior', 'x_2_estimated', 
                        'x_3_true', 'x_3_prior', 'x_3_estimated']].std()
    sample_size = non_zero_counts_df.groupby('lr')[['x_1_true', 'x_1_prior', 'x_1_estimated', 
                        'x_2_true', 'x_2_prior', 'x_2_estimated', 
                        'x_3_true', 'x_3_prior', 'x_3_estimated']].count()
    std_error_df = std_counts_df / np.sqrt(sample_size)

    # Create a new DataFrame to hold the reshaped data for plotting
    mean_counts_reshaped = mean_counts_df.stack().reset_index()
    mean_counts_reshaped.columns = ['lr', 'Variable', 'MeanCount']

    std_error_reshaped = std_error_df.stack().reset_index()
    std_error_reshaped.columns = ['lr', 'Variable', 'StdError']

    # Merge mean and std error data
    plot_data = pd.merge(mean_counts_reshaped, std_error_reshaped, on=['lr', 'Variable'])

    # Define the order for the x-axis
    variable_order = ['x_1_true', 'x_1_prior', 'x_1_estimated', 
                        'x_2_true', 'x_2_prior', 'x_2_estimated', 
                        'x_3_true', 'x_3_prior', 'x_3_estimated']

    # Extract learning rates directly from the DataFrame
    learning_rates = plot_data['lr'].unique()  # Get unique learning rates from the DataFrame

    # Create a figure with subplots
    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(18, 12))  # 2 rows, 3 columns
    axs = axs.flatten()  # Flatten to easily index into

    # Create bar plots for each learning rate
    for idx, lr in enumerate(learning_rates):
        # Filter the reshaped DataFrame for the current learning rate
        lr_data = plot_data[plot_data['lr'] == lr].copy()  # Use copy to avoid SettingWithCopyWarning

        # Ensure the variables are in the specified order using .loc
        lr_data.loc[:, 'Variable'] = pd.Categorical(lr_data['Variable'], categories=variable_order, ordered=True)
        lr_data = lr_data.sort_values('Variable')

        # Convert mean counts to percentages
        lr_data['MeanCount'] *= 100  # Convert to percentage

        # Create a bar plot
        bar_width = 0.25  # Adjusted width of the bars
        x = np.arange(len(lr_data) // 3)  # Positions of the variable groups (3 per variable)

        # Create bars for ground truth, prior, and estimated interventions
        bar1 = axs[idx].bar(x - bar_width, lr_data[lr_data['Variable'].str.contains('true')]['MeanCount'], width=bar_width, label='Ground Truth SCM', color='royalblue', alpha=0.7)
        bar2 = axs[idx].bar(x, lr_data[lr_data['Variable'].str.contains('prior')]['MeanCount'], width=bar_width, label='Prior SCM', color='seagreen', alpha=0.7)
        bar3 = axs[idx].bar(x + bar_width, lr_data[lr_data['Variable'].str.contains('estimated')]['MeanCount'], width=bar_width, label='Estimated SCM', color='tomato', alpha=0.7)

        # Add error bars for ground truth, prior, and estimated interventions
        axs[idx].errorbar(x - bar_width, lr_data[lr_data['Variable'].str.contains('true')]['MeanCount'], 
                        yerr=lr_data[lr_data['Variable'].str.contains('true')]['StdError'] * 100, fmt='none', ecolor='darkgray', 
                        elinewidth=2, capsize=5, capthick=2, linestyle='--', alpha=0.8)
        axs[idx].errorbar(x, lr_data[lr_data['Variable'].str.contains('prior')]['MeanCount'], 
                        yerr=lr_data[lr_data['Variable'].str.contains('prior')]['StdError'] * 100, fmt='none', ecolor='darkgray', 
                        elinewidth=2, capsize=5, capthick=2, linestyle='--', alpha=0.8)
        axs[idx].errorbar(x + bar_width, lr_data[lr_data['Variable'].str.contains('estimated')]['MeanCount'], 
                        yerr=lr_data[lr_data['Variable'].str.contains('estimated')]['StdError'] * 100, fmt='none', ecolor='darkgray', 
                        elinewidth=2, capsize=5, capthick=2, linestyle='--', alpha=0.8)
                        


        # Customize the plot
        axs[idx].set_title(f'Interventions (%) for Learning Rate {lr}', fontsize=18)
        axs[idx].set_ylabel(f'% of intervention on the node', fontsize=16)
        axs[idx].set_xticks(x)
        axs[idx].set_xticklabels([r'$X_1$', r'$X_2$', r'$X_3$'], fontsize=14)
        axs[idx].legend()
        axs[idx].grid(axis='y', linestyle='--', alpha=0.7)

        # Set y-axis limits
        axs[idx].set_ylim(0, 100) 

        # Add data labels on the bars
        for bar in [bar1, bar2, bar3]:
            for b in bar:
                yval = b.get_height()
                axs[idx].text(b.get_x() + b.get_width() / 2, yval + 1.5, f'{round(yval, 1)}%', ha='center', va='bottom', fontsize=10)



    # Adjust layout
    plt.tight_layout()

    fig.suptitle('Interventions Percentage for Different Learning Rates', fontsize=24, y=1.05)

    # Save the combined plot as a single image
    combined_plot_file_path = os.path.join(result_folder, f'focus_interventions_eps_{epsilon}_thetas_{true_thetas}.png')
    plt.savefig(combined_plot_file_path, format='png', bbox_inches='tight')
    combined_plot_file_path = os.path.join(result_folder, f'focus_interventions_eps_{epsilon}_thetas_{true_thetas}.pdf')
    plt.savefig(combined_plot_file_path, format='pdf', bbox_inches='tight')
    plt.close()  # Close the figure after saving to avoid display

    print(f'Combined bar plots saved in {result_folder}')
