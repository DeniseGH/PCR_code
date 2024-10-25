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

def filter_non_zero_rows(est_str, prior_str, true_str):

    # Convert the string representations to actual lists
    est = ast.literal_eval(est_str)
    prior = ast.literal_eval(prior_str)
    true = ast.literal_eval(true_str)
    #print(f"Length of est: {len(est)}, Length of prior: {len(prior)}, Length of true: {len(true)}")


    filtered_estimated = []
    filtered_prior = []
    filtered_true = []

    # Loop over the rows in the tensors
    for i in range(len(est)):
        # Check if there is at least one non-zero value in all three tensors
        if any(x != 0 for x in est[i]) and any(x != 0 for x in prior[i]) and any(x != 0 for x in true[i]):
            filtered_estimated.append(est[i])
            filtered_prior.append(prior[i])
            filtered_true.append(true[i])

    #print(f"Length of est: {len(filtered_estimated)}, Length of prior: {len(filtered_prior)}, Length of true: {len(filtered_true)}")

    # Convert filtered lists back to string representations
    filtered_estimated_str = str(filtered_estimated).replace('], [', '],[')  # To maintain the format
    filtered_prior_str = str(filtered_prior).replace('], [', '],[')  # To maintain the format
    filtered_true_str = str(filtered_true).replace('], [', '],[')  # To maintain the format

    # Remove any extra spaces for consistency
    filtered_estimated_str = filtered_estimated_str.replace(' ', '')
    filtered_prior_str = filtered_prior_str.replace(' ', '')
    filtered_true_str = filtered_true_str.replace(' ', '')


    return filtered_estimated_str, filtered_prior_str, filtered_true_str


def interventions_non_zero(df):
    # Initialize an empty DataFrame to store the non-zero counts for each iteration
    non_zero_counts_df = pd.DataFrame(columns=[
        'Iteration', 'lr', 'x_1_estimated', 'x_2_estimated', 'x_3_estimated', 
        'x_1_prior', 'x_2_prior', 'x_3_prior', 'x_1_true', 'x_2_true', 'x_3_true'
    ])

    for i in range(0, 30):
        intervention_estimated = clean_tensor_string(df.iloc[i, 1])
        intervention_prior = clean_tensor_string(df.iloc[i, 2])
        intervention_true = clean_tensor_string(df.iloc[i, 3])
        intervention_estimated, intervention_prior, intervention_true = filter_non_zero_rows(intervention_estimated, intervention_prior, intervention_true)

        ######### ESTIMATED
        # Clean up the tensor representation and safely evaluate the string for estimated interventions
        intervention_estimated_list = ast.literal_eval(intervention_estimated)

        
        # Convert the list into a DataFrame for estimated interventions
        df_intervention_estimated = pd.DataFrame(intervention_estimated_list, columns=['x_1_estimated', 'x_2_estimated', 'x_3_estimated'])

        # Count non-zero values per column for estimated interventions
        non_zero_counts_estimated = (df_intervention_estimated != 0).sum()/len(df_intervention_estimated)
        
        ######### PRIOR
        # Clean up the tensor representation and safely evaluate the string for prior interventions
        intervention_prior_list = ast.literal_eval(intervention_prior)
        
        # Convert the list into a DataFrame for prior interventions
        df_intervention_prior = pd.DataFrame(intervention_prior_list, columns=['x_1_prior', 'x_2_prior', 'x_3_prior'])

        # Count non-zero values per column for prior interventions
        non_zero_counts_prior = (df_intervention_prior != 0).sum()/len(df_intervention_prior)

        ######### TRUE
        # Clean up the tensor representation and safely evaluate the string for true interventions
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


def recourse_interventions_plot_v2(name_of_the_folder, epsilon, true_thetas):

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
    # print(mean_counts_df)
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
    fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(12, 18))  # 2 rows, 3 columns
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
        axs[idx].set_xticklabels([r'$X_1$', r'$X_2$', r'$X_3$'], fontsize=16)
        axs[idx].legend()
        axs[idx].grid(axis='y', linestyle='--', alpha=0.7)

        # Set y-axis limits
        axs[idx].set_ylim(0, 109) 

        # # Add data labels on the bars
        # for bar in [bar1, bar2, bar3]:
        #     for b in bar:
        #         yval = b.get_height()
        #         axs[idx].text(b.get_x() + b.get_width() / 2, yval + 1.5, f'{round(yval, 1)}%', ha='center', va='bottom', fontsize=12)



    # Adjust layout
    plt.tight_layout()

    fig.suptitle('Interventions Percentage for Different Learning Rates', fontsize=24, y=1.05)

    # Save the combined plot as a single image
    combined_plot_file_path = os.path.join(result_folder, f'focus_interventions_eps_{epsilon}_thetas_{true_thetas}_nonzero_32.png')
    plt.savefig(combined_plot_file_path, format='png', bbox_inches='tight')
    combined_plot_file_path = os.path.join(result_folder, f'focus_interventions_eps_{epsilon}_thetas_{true_thetas}_nonzero_32.pdf')
    plt.savefig(combined_plot_file_path, format='pdf', bbox_inches='tight')
    plt.close()  # Close the figure after saving to avoid display

    print(f'Combined bar plots saved in {result_folder}')


def count_non_zero_combinations(df, prefix):
    # Define combinations and their conditions
    conditions = {
            f'x_1_{prefix}': [f'x_1_{prefix} != 0', f'x_2_{prefix} == 0', f'x_3_{prefix} == 0'],
            f'x_2_{prefix}': [f'x_1_{prefix} == 0', f'x_2_{prefix} != 0', f'x_3_{prefix} == 0'],
            f'x_3_{prefix}': [f'x_1_{prefix} == 0', f'x_2_{prefix} == 0', f'x_3_{prefix} != 0'],
            f'x_1_&_x_2_{prefix}': [f'x_1_{prefix} != 0', f'x_2_{prefix} != 0', f'x_3_{prefix} == 0'],
            f'x_1_&_x_3_{prefix}': [f'x_1_{prefix} != 0', f'x_2_{prefix} == 0', f'x_3_{prefix} != 0'],
            f'x_2_&_x_3_{prefix}': [f'x_1_{prefix} == 0', f'x_2_{prefix} != 0', f'x_3_{prefix} != 0'],
            f'x_1_&_x_2_&_x_3_{prefix}': [f'x_1_{prefix} != 0', f'x_2_{prefix} != 0', f'x_3_{prefix} != 0']
        }

    # Count occurrences of each condition
    counts = {
        key: len(df.query(" and ".join(conds)))
        for key, conds in conditions.items()
    }

    return counts


def combination_count(df):
    # Initialize an empty DataFrame to store the non-zero counts for each iteration
    non_zero_counts_df = pd.DataFrame(columns=[
        'Iteration', 'lr', 
        'x_1_true', 'x_2_true', 'x_3_true', 'x_1_&_x_2_true', 'x_1_&_x_3_true', 'x_2_&_x_3_true', 'x_1_&_x_2_&_x_3_true', 
        'x_1_prior', 'x_2_prior', 'x_3_prior', 'x_1_&_x_2_prior', 'x_1_&_x_3_prior', 'x_2_&_x_3_prior', 'x_1_&_x_2_&_x_3_prior', 
        'x_1_estimated', 'x_2_estimated', 'x_3_estimated', 'x_1_&_x_2_estimated', 'x_1_&_x_3_estimated', 'x_2_&_x_3_estimated', 'x_1_&_x_2_&_x_3_estimated',
    ])

    for i in range(0, 30):
        intervention_estimated = clean_tensor_string(df.iloc[i, 1])
        intervention_prior = clean_tensor_string(df.iloc[i, 2])
        intervention_true = clean_tensor_string(df.iloc[i, 3])
        
        intervention_estimated_, intervention_prior_, intervention_true_ = filter_non_zero_rows(intervention_estimated, intervention_prior, intervention_true)

        ######### ESTIMATED
        # Clean up the tensor representation and safely evaluate the string for estimated interventions
        intervention_estimated_list = ast.literal_eval(intervention_estimated_)
        

        
        # Convert the list into a DataFrame for estimated interventions
        df_intervention_estimated = pd.DataFrame(intervention_estimated_list, columns=['x_1_estimated', 'x_2_estimated', 'x_3_estimated'])
        estimates_counts = count_non_zero_combinations(df_intervention_estimated, 'estimated')


        ######### PRIOR
        # Clean up the tensor representation and safely evaluate the string for prior interventions
        intervention_prior_list = ast.literal_eval(intervention_prior_)
        
        # Convert the list into a DataFrame for prior interventions
        df_intervention_prior = pd.DataFrame(intervention_prior_list, columns=['x_1_prior', 'x_2_prior', 'x_3_prior'])
        prior_counts = count_non_zero_combinations(df_intervention_prior, 'prior')



        ######### TRUE
        # Clean up the tensor representation and safely evaluate the string for true interventions
        intervention_true_list = ast.literal_eval(intervention_true_)
        
        # Convert the list into a DataFrame for true interventions
        df_intervention_true = pd.DataFrame(intervention_true_list, columns=['x_1_true', 'x_2_true', 'x_3_true'])
        true_counts = count_non_zero_combinations(df_intervention_true, 'true')

        
        new_row = pd.DataFrame({
                'Iteration': [i],
                'lr': [df.iloc[i, 0]],  # Assuming the learning rate is in column 0
                **{key: [val] for key, val in true_counts.items()},
                **{key: [val] for key, val in prior_counts.items()},
                **{key: [val] for key, val in estimates_counts.items()}
            })
        
        # Concatenate the new row to the non_zero_counts_df
        non_zero_counts_df = pd.concat([non_zero_counts_df, new_row], ignore_index=True)

    # Display the final DataFrame with non-zero counts per iteration
    print(non_zero_counts_df.head())


    return non_zero_counts_df


def recourse_interventions_plot_v3(name_of_the_folder, epsilon, true_thetas):

    df = pd.read_csv(f'{name_of_the_folder}/interventions_analysis.csv')



    non_zero_counts_df = combination_count(df)
    new_column_order = ['Iteration', 'lr', 
        'x_1_true', 'x_2_true', 'x_3_true', 'x_1_&_x_2_true', 'x_1_&_x_3_true', 'x_2_&_x_3_true', 'x_1_&_x_2_&_x_3_true', 
        'x_1_prior', 'x_2_prior', 'x_3_prior', 'x_1_&_x_2_prior', 'x_1_&_x_3_prior', 'x_2_&_x_3_prior', 'x_1_&_x_2_&_x_3_prior', 
        'x_1_estimated', 'x_2_estimated', 'x_3_estimated', 'x_1_&_x_2_estimated', 'x_1_&_x_3_estimated', 'x_2_&_x_3_estimated', 'x_1_&_x_2_&_x_3_estimated',]

    # Reordering the DataFrame
    non_zero_counts_df = non_zero_counts_df[new_column_order]
    print("len(non_zero_counts_df)", len(non_zero_counts_df))
    print(non_zero_counts_df.head())


    # Initialize the directory for results
    current_dir = os.getcwd()  # Get the current working directory
    result_folder = os.path.join(current_dir, f'{name_of_the_folder}')
    os.makedirs(result_folder, exist_ok=True)  # Create the folder if it doesn't exist

    # Group by learning rate and calculate mean counts
    mean_counts_df = non_zero_counts_df.groupby('lr')[new_column_order].mean()
    print("len(mean_counts_df)", len(mean_counts_df))
    print(mean_counts_df.head())

    # print(mean_counts_df)
    # Calculate standard error
    std_counts_df = non_zero_counts_df.groupby('lr')[new_column_order].std()
    sample_size = non_zero_counts_df.groupby('lr')[new_column_order].count()
    std_error_df = std_counts_df / np.sqrt(sample_size)
    print("len(std_error_df)", len(std_error_df))
    print(std_error_df.head())

    # Create a new DataFrame to hold the reshaped data for plotting
    mean_counts_reshaped = mean_counts_df.stack().reset_index()
    mean_counts_reshaped.columns = ['lr', 'Variable', 'MeanCount']

    std_error_reshaped = std_error_df.stack().reset_index()
    std_error_reshaped.columns = ['lr', 'Variable', 'StdError']
    print(mean_counts_reshaped.head())
    print(std_error_reshaped.head())

    # Merge mean and std error data
    plot_data = pd.merge(mean_counts_reshaped, std_error_reshaped, on=['lr', 'Variable'])
    print("len(plot_data)", len(plot_data))
    
    print(plot_data.head())

    plot_data_filtered = plot_data[plot_data['lr'] == 0.75]
    print((plot_data_filtered))



    # Define the groups and their corresponding variable names
    groups = {
        'X_1': ['x_1_true', 'x_1_prior', 'x_1_estimated'],
        'X_2': ['x_2_true', 'x_2_prior', 'x_2_estimated'],
        'X_3': ['x_3_true', 'x_3_prior', 'x_3_estimated'],
        'X_1_&_X_2': ['x_1_&_x_2_true', 'x_1_&_x_2_prior', 'x_1_&_x_2_estimated'],
        'X_1_&_X_3': ['x_1_&_x_3_true', 'x_1_&_x_3_prior', 'x_1_&_x_3_estimated'],
        'X_2_&_X_3': ['x_2_&_x_3_true', 'x_2_&_x_3_prior', 'x_2_&_x_3_estimated'],
        'X_1_&_X_2_&_X_3': ['x_1_&_x_2_&_x_3_true', 'x_1_&_x_2_&_x_3_prior', 'x_1_&_x_2_&_x_3_estimated']
    }

        # Create a figure and axes
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))  # 2 rows, 4 columns to fit 7 plots (we'll leave 1 empty)

    # Flatten the axes array to easily iterate over it
    axes = axes.flatten()

    # Set common x-axis labels for Ground Truth, Prior, and Estimated
    x_labels = ['Ground Truth', 'Prior', 'Estimated']

    # Loop through each group and plot the corresponding values
    for i, (group_name, variables) in enumerate(groups.items()):
        # Filter the data for the relevant variables
        group_data = plot_data_filtered[plot_data_filtered['Variable'].isin(variables)]
        
        # Extract the MeanCount and StdError for the 3 variables (True, Prior, Estimated)
        mean_counts = group_data['MeanCount'].values
        std_errors = group_data['StdError'].values

        # Plot the bar chart for the current group
        axes[i].bar(x_labels, mean_counts, yerr=std_errors, capsize=5, color=['royalblue', 'seagreen', 'tomato'], alpha=0.7)
        

        # Set the title and labels
        axes[i].set_title(f'{group_name}', fontsize=16)
        axes[i].set_ylabel('Mean Count', fontsize=12)

    # Hide the 8th subplot (which will be empty since we only need 7 plots)
    axes[-1].axis('off')
    # fig.suptitle('Interventions Percentage for Different Learning Rates', fontsize=24, y=1.05)

    # # Save the combined plot as a single image
    combined_plot_file_path = os.path.join(result_folder, f'interventions_analysis_eps_{epsilon}_thetas_{true_thetas}_TRIAL.png')
    plt.savefig(combined_plot_file_path, format='png', bbox_inches='tight')
    combined_plot_file_path = os.path.join(result_folder, f'focus_interventions_eps_{epsilon}_thetas_{true_thetas}_nonzero_32.pdf')
    plt.savefig(combined_plot_file_path, format='pdf', bbox_inches='tight')
    plt.close()  # Close the figure after saving to avoid display

    print(f'Combined bar plots saved in {result_folder}')


def recourse_interventions_plot_v1(name_of_the_folder, epsilon, true_thetas):
    df = pd.read_csv(f'{name_of_the_folder}/interventions_analysis.csv')

    non_zero_counts_df = combination_count(df)
    new_column_order = [
        'Iteration', 'lr', 
        'x_1_true', 'x_2_true', 'x_3_true', 
        'x_1_&_x_2_true', 'x_1_&_x_3_true', 
        'x_2_&_x_3_true', 'x_1_&_x_2_&_x_3_true', 
        'x_1_prior', 'x_2_prior', 'x_3_prior', 
        'x_1_&_x_2_prior', 'x_1_&_x_3_prior', 
        'x_2_&_x_3_prior', 'x_1_&_x_2_&_x_3_prior', 
        'x_1_estimated', 'x_2_estimated', 
        'x_3_estimated', 'x_1_&_x_2_estimated', 
        'x_1_&_x_3_estimated', 'x_2_&_x_3_estimated', 
        'x_1_&_x_2_&_x_3_estimated'
    ]

    # Reordering the DataFrame
    non_zero_counts_df = non_zero_counts_df[new_column_order]

    # Initialize the directory for results
    current_dir = os.getcwd()  # Get the current working directory
    result_folder = os.path.join(current_dir, f'{name_of_the_folder}')
    os.makedirs(result_folder, exist_ok=True)  # Create the folder if it doesn't exist

    # Group by learning rate and calculate mean counts
    mean_counts_df = non_zero_counts_df.groupby('lr')[new_column_order].mean()

    # Calculate standard error
    std_counts_df = non_zero_counts_df.groupby('lr')[new_column_order].std()
    sample_size = non_zero_counts_df.groupby('lr')[new_column_order].count()
    std_error_df = std_counts_df / np.sqrt(sample_size)
    print("len(std_error_df)", len(std_error_df))
    print(std_error_df.head())

    # Create a new DataFrame to hold the reshaped data for plotting
    mean_counts_reshaped = mean_counts_df.stack().reset_index()
    mean_counts_reshaped.columns = ['lr', 'Variable', 'MeanCount']

    std_error_reshaped = std_error_df.stack().reset_index()
    std_error_reshaped.columns = ['lr', 'Variable', 'StdError']

    # Merge mean and std error data
    plot_data = pd.merge(mean_counts_reshaped, std_error_reshaped, on=['lr', 'Variable'])


    choosen_learning_rate = 0.75

    plot_data_filtered = plot_data[plot_data['lr'] == choosen_learning_rate]
    # print("#############", (plot_data_filtered))

    # Define the groups and their corresponding variable names with new labels
    groups = {
        'Interventions on $X_1$': ['x_1_true', 'x_1_prior', 'x_1_estimated'],
        'Interventions on $X_2$': ['x_2_true', 'x_2_prior', 'x_2_estimated'],
        'Interventions on $X_3$': ['x_3_true', 'x_3_prior', 'x_3_estimated'],
        'Interventions on $X_1$ and $X_2$': ['x_1_&_x_2_true', 'x_1_&_x_2_prior', 'x_1_&_x_2_estimated'],
        'Interventions on $X_1$ and $X_3$': ['x_1_&_x_3_true', 'x_1_&_x_3_prior', 'x_1_&_x_3_estimated'],
        'Interventions on $X_2$ and $X_3$': ['x_2_&_x_3_true', 'x_2_&_x_3_prior', 'x_2_&_x_3_estimated'],
        'Interventions on $X_1$, $X_2$, and $X_3$': ['x_1_&_x_2_&_x_3_true', 'x_1_&_x_2_&_x_3_prior', 'x_1_&_x_2_&_x_3_estimated']
    }

    # Create a figure and axes
    fig, axes = plt.subplots(3, 3, figsize=(20, 20))  # 2 rows, 4 columns to fit 7 plots (we'll leave 1 empty)
    print(true_thetas[2])
    if true_thetas[2] == 0.99:
        fig.suptitle(
            fr'Recourse Distributions of Interventions of $\mathcal{{M}}_1$ for Learning Rate = {choosen_learning_rate} (ε = {epsilon})', 
            fontsize=22, 
            y=0.93
        )
    else:
        fig.suptitle(
            fr'Recourse Distributions of Interventions of $\mathcal{{M}}_2$ for Learning Rate = {choosen_learning_rate} (ε = {epsilon})', 
            fontsize=22, 
            y=0.93
        )


    # Flatten the axes array to easily iterate over it
    axes = axes.flatten()

    # Set common x-axis labels for Ground Truth, Prior, and Estimated
    x_labels = ['Ground Truth', 'Prior', 'Estimated']

    # Loop through each group and plot the corresponding values
    for i, (group_name, variables) in enumerate(groups.items()):
        # Filter the data for the relevant variables
        group_data = plot_data_filtered[plot_data_filtered['Variable'].isin(variables)]
        
        # Extract the MeanCount and StdError for the 3 variables (True, Prior, Estimated)
        mean_counts = group_data['MeanCount'].values
        std_errors = group_data['StdError'].values

        # Plot the bar chart for the current group
        axes[i].bar(x_labels, mean_counts, width=0.5, capsize=5, color=['royalblue', 'seagreen', 'tomato'], alpha = 0.7)
        
        # Add the error bars with custom styling
        axes[i].errorbar(x_labels, mean_counts, yerr=std_errors, fmt='none',
                        ecolor="darkgray", elinewidth=2, capsize=8, 
                        capthick=2, linestyle='--', alpha=0.9)

        
        # Set y-limits to 20
        axes[i].set_ylim(0, 31)

        # Set the title and labels
        axes[i].set_title(f'{group_name}', fontsize=18)
        axes[i].set_ylabel('Mean of the # Interventions', fontsize=18, labelpad=2.5)
        axes[i].set_xticklabels(x_labels, fontsize=16)
        # Add a grid for better readability
        axes[i].grid(True, linestyle='--', alpha=0.7)


    # Hide the 8th subplot (which will be empty since we only need 7 plots)
    axes[-2].axis('off')
    axes[-1].axis('off')
    
    # Save the combined plot as a single image
    combined_plot_file_path = os.path.join(result_folder, f'interventions_analysis_eps_{epsilon}_thetas_{true_thetas}.png')
    plt.savefig(combined_plot_file_path, format='png', bbox_inches='tight')
    
    combined_plot_file_path = os.path.join(result_folder, f'interventions_analysis_eps_{epsilon}_thetas_{true_thetas}.pdf')
    plt.savefig(combined_plot_file_path, format='pdf', bbox_inches='tight')
    
    plt.close()  # Close the figure after saving to avoid display

    print(f'Combined bar plots saved in {result_folder}')





# name_of_the_folder= "Results_Recourse_Test_Thesis_eps_0.1"
# # df = pd.read_csv(f'{name_of_the_folder}/interventions_analysis.csv')
# recourse_interventions_plot_v1(name_of_the_folder, 0.1, [-0.99, 0.99, 0.99])
# # combination_count(df)
