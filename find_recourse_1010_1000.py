import torch
import recourse
import data_utils
import trainers
import os
import numpy as np
from scm import SCM_PCR_linear
from scm import SCM_PCR_linear_estimated
from scm import SCM_PCR_linear_Test
from scm import SCM_PCR_linear_estimated_Test
from scm import SCM_PCR_linear_Prior
import csv
import uuid  # Import the UUID module to create unique keys



# Function to print and log to a file simultaneously
def print_and_log(message, log_file):
    print(message)  # Print to console
    with open(log_file, 'a') as f:
        f.write(message + '\n')  # Write to file

def checks(X_explain, interv_estimated, interv_prior):
        if not isinstance(X_explain, torch.Tensor):
            X_explain = torch.tensor(X_explain)

        # Ensure interv_estimated is a torch.Tensor of shape (N, D)
        if not isinstance(interv_estimated, torch.Tensor):
            interv_estimated = torch.tensor(interv_estimated)
        
        # Ensure interv_prior is a torch.Tensor of shape (N, D)
        if not isinstance(interv_prior, torch.Tensor):
            interv_prior = torch.tensor(interv_prior)

        # Check if X_explain and interv_estimated have the same shape (N, D)
        if X_explain.shape != interv_estimated.shape:
            raise ValueError("X_explain and interv_estimated must have the same shape (N, D).")
        
        # Check if X_explain and interv_estimated have the same shape (N, D)
        if X_explain.shape != interv_prior.shape:
            raise ValueError("X_explain and interv_prior must have the same shape (N, D).")

        return X_explain, interv_estimated, interv_prior

def find_recourse_mlp_with_comparison(model, scmm_prior, scmm_estimated, scmm_true, X_explain, constraints, epsilon, learning_rate):
    print_and_log("\n ##################### Entering find_recourse_MLP with comparison", log_file)
    hyperparams = {'lr': learning_rate, 'lambd_init': 1.0, 'decay_rate': 0.02, 'outer_iters': 50, 'inner_iters': 10, 'recourse_lr': 0.1}
    explainer = recourse.DifferentiableRecourse(model, hyperparams)

    print_and_log("Starting causal recourse computation with SCM_PCR_linear_estimated", log_file)
    interv_estimated, _, cost_recourse_estimated, _, _ = recourse.causal_recourse(X_explain, explainer, constraints, scm=scmm_estimated, epsilon=epsilon, robust=epsilon>0)
    print_and_log("Causal recourse computation with SCM_PCR_linear_estimated finished", log_file)


    print_and_log("Starting causal recourse computation with SCM_PCR_linear_prior", log_file)
    interv_prior, _, cost_recourse_prior, _, _ = recourse.causal_recourse(X_explain, explainer, constraints, scm=scmm_prior, epsilon=epsilon, robust=epsilon>0)
    print_and_log("Causal recourse computation with SCM_PCR_linear_prior finished", log_file)

    
    X_explain, interv_estimated, interv_prior = checks(X_explain, interv_estimated, interv_prior)

    X_results = scmm_true.counterfactual(X_explain, interv_estimated)
    preds = model.predict(X_results)
    # Print and log the counts
    print_and_log(f"Number of ones: {(preds == 1).sum()}, Number of zeros: {(preds == 0).sum()} out of the total {len(preds)}", log_file)


    X_results_prior = scmm_true.counterfactual(X_explain, interv_prior)
    preds_prior = model.predict(X_results_prior)
    # Print and log the counts
    print_and_log(f"Number of ones: {(preds_prior == 1).sum()}, Number of zeros: {(preds_prior == 0).sum()} out of the total {len(preds_prior)}", log_file)



    # Now check validity with the true SCM (SCM_PCR_linear)
    print_and_log("Starting causal recourse validation with SCM_PCR_linear (true SCM)", log_file)
    interv_true, recourse_valid_true, cost_recourse_true, _, _ = recourse.causal_recourse(
         X_explain, explainer, constraints, scm=scmm_true, epsilon=epsilon, robust=epsilon>0)
    print_and_log(f"Number of ones: {(recourse_valid_true == True).sum()}, Number of zeros: {(recourse_valid_true == False).sum()} out of the total {len(recourse_valid_true)}", log_file)
    print_and_log("Causal recourse validation with SCM_PCR_linear finished", log_file)

    if not isinstance(interv_true, torch.Tensor):
            interv_true = torch.tensor(interv_true)

    # Comparison logic
    recourse_comparison_valid = (preds.astype(np.bool_) & recourse_valid_true.astype(np.bool_))  & (preds_prior.astype(np.bool_) )
    
    true_greater_estimated = 0 
    estimated_greater_true = 0 
    true_greater_prior = 0 
    prior_greater_true = 0 

    cost_dict = {}

    # Check cost where both recourse are valid under the true SCM
    for i, valid in enumerate(recourse_comparison_valid):
        if i not in cost_dict:
            cost_dict[i] = []
        if valid: 
            
            if cost_recourse_true[i] <= cost_recourse_estimated[i]:
                estimated_greater_true = estimated_greater_true + 1

            elif cost_recourse_true[i] <= cost_recourse_prior[i]:
                prior_greater_true = prior_greater_true + 1

            else:
                true_greater_estimated = true_greater_estimated + 1
                true_greater_prior = true_greater_prior + 1
            
            cost_entry = {
                "cost_recourse_true": cost_recourse_true[i],
                "cost_recourse_estimated": cost_recourse_estimated[i],
                "cost_recourse_prior": cost_recourse_prior[i]
            }

            # Append the cost data to the current iteration's list
            cost_dict[i].append(cost_entry)
        


    print_and_log(f"\nCost was lower or equal for true SCM compared to the estimated {estimated_greater_true} times ", log_file)
    print_and_log(f"\nCost was greater for true SCM compared to the estimated {true_greater_estimated} times ", log_file)
    print_and_log(f"\nCost was lower or equal for true SCM compared to the prior {prior_greater_true} times ", log_file)
    print_and_log(f"\nCost was greater for true SCM compared to the prior {true_greater_prior} times ", log_file)



    return (preds.astype(np.bool_), preds_prior.astype(np.bool_), recourse_valid_true, recourse_comparison_valid, cost_dict,
    interv_estimated, interv_prior, interv_true) 

def compute_cost_means(cost_dict):
    # Initialize lists to collect costs
    true_costs = []
    estimated_costs = []
    prior_costs = []

    # Iterate through the dictionary to gather costs
    for iteration, cost_entries in cost_dict.items():
        for cost_entry in cost_entries:
            true_costs.append(cost_entry['cost_recourse_true'])
            estimated_costs.append(cost_entry['cost_recourse_estimated'])
            prior_costs.append(cost_entry['cost_recourse_prior'])

    # Compute the mean of each list of costs
    mean_true = np.mean(true_costs) if true_costs else 0
    mean_estimated = np.mean(estimated_costs) if estimated_costs else 0
    mean_prior = np.mean(prior_costs) if prior_costs else 0

    # Print or return the means
    print(f"Mean cost of recourse (True): {mean_true}")
    print(f"Mean cost of recourse (Estimated): {mean_estimated}")
    print(f"Mean cost of recourse (Prior): {mean_prior}")
    
    return mean_true, mean_estimated, mean_prior

def set_the_csv_files(current_dir, name_of_the_folder): 
    result_folder = os.path.join(current_dir, f"{name_of_the_folder}")
    os.makedirs(result_folder, exist_ok=True)  # Create the folder if it doesn't exist
    log_file = os.path.join(result_folder, f'results_log.txt')

    csv_file_path = os.path.join(result_folder, f'results_summary.csv')
    interv_file_path = os.path.join(result_folder, f'interventions_analysis.csv')

    # Initialize CSV files and write the headers
    with open(csv_file_path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        
        # Write header for results summary
        csvwriter.writerow([
            'key', 'true_thetas', 'learning_rate', 'epsilon', 'N_explain', 'num_ones_train', 
            'num_zeros_train', 'num_ones_test', 'num_zeros_test', 'model_type', 'num_epochs', 
            'correct_preds', 'MLP accuracy', "X_explain_shape", 'Recourse_estimated_in_True_SCM', 
            'Recourse_in_True_SCM', 'accuracy_preds', 'accuracy_true', 'accuracy_comparison_tot', 
            'cost_mean', 'cost_mean_estimated', 'cost_mean_baseline', 'accuracy_preds_prior'
        ])

    with open(interv_file_path, 'w', newline='') as intervfile:
        intervwriter = csv.writer(intervfile)

        # Write header for interventions analysis
        intervwriter.writerow([
            'learning_rate', 'interv_estimated', 'interv_prior', 'interv_true'
        ])

    return log_file, csv_file_path, interv_file_path

# Function to check if the input is a float
def get_valid_float(prompt):
    while True:
        try:
            epsilon = float(input(prompt))
            return epsilon
        except ValueError:
            print("Invalid input! Please enter a valid float (e.g., 0.0, 0.1, 0.2).")

# Function to validate the example type
def get_valid_example_type(prompt):
    while True:
        example_type = input(prompt).strip().lower()
        if example_type in ['test', 'paper']:
            return example_type
        else:
            print("Invalid input! Please choose either 'test' or 'paper'.")

# Function to validate the path input
def get_valid_directory(prompt):
    while True:
        path_input = input(prompt).strip().lower()
        if path_input == "yes":
            current_dir = os.getcwd()  # Use current directory
            print(f"Results will be stored in the current directory: {current_dir}")
            return current_dir
        elif path_input == "no":
            while True:
                custom_dir = input("Write here the path to the directory you would like to use: ").strip()
                if os.path.exists(custom_dir) and os.path.isdir(custom_dir):
                    print(f"Results will be stored in the directory: {custom_dir}")
                    return custom_dir
                else:
                    print("Invalid directory path! Please enter a valid directory.")
        else:
            print("Invalid input! Please enter 'yes' or 'no'.")


name_of_the_folder = input("Please write the name of the folder: ")
current_dir = get_valid_directory("Do you want to use the current directory to store the results? (yes/no): ")

log_file, csv_file_path, interv_file_path = set_the_csv_files(current_dir, name_of_the_folder)


example_type = get_valid_example_type("Which example do you want to use? (test/paper): ")
epsilon = get_valid_float("Which value of epsilon would you like to use? (Suggested 0.0/0.1/0.2): ")


# Check user input and initialize variables accordingly
if example_type == "test":
    # Test example
    true_thetas = [-0.99, 0.99, 0.99]  # From true SCM
    scmm_estimated = SCM_PCR_linear_estimated_Test()  # Estimated SCM
    scmm_true = SCM_PCR_linear_Test()  # True SCM
else:
    # Paper example (default if user doesn't input 'test')
    true_thetas = [-0.99, 0.05, 0.25]  # From true SCM
    scmm_true = SCM_PCR_linear()  # True SCM
    scmm_estimated = SCM_PCR_linear_estimated()  # Estimated SCM

scmm_prior = SCM_PCR_linear_Prior()

model_type = 'MLP'

N_explain = 500
print_and_log(f"\nEpsilon: {epsilon}", log_file)

number_of_iterations = 5

print_and_log(f"\nNumber of iterations: {number_of_iterations}", log_file)

for iteration in range(0, number_of_iterations):
    print("########################## Iteration: ", iteration + 1)
    X, Y, constraints = data_utils.process_data('linear', scmm_true)
    X_train, Y_train, X_test, Y_test = data_utils.train_test_split(X, Y)

    # Count the number of 1s and 0s in Y_train
    num_ones_train = np.sum(Y_train == 1)
    num_zeros_train = np.sum(Y_train == 0)

    # Count the number of 1s and 0s in Y_test
    num_ones_test = np.sum(Y_test == 1)
    num_zeros_test = np.sum(Y_test == 0)

    # Print the results
    print_and_log(f"Y_train: {num_ones_train} ones, {num_zeros_train} zeros", log_file)
    print_and_log(f"Y_test: {num_ones_test} ones, {num_zeros_test} zeros \n", log_file)

    model = trainers.MLP
    model = model(X_train.shape[-1], actionable_features=constraints['actionable'], actionable_mask=False)

    X_train_tensor = torch.Tensor(X_train)
    Y_train_tensor = torch.Tensor(Y_train).reshape(-1)  # Make it (80,)

    # Define loss function and optimizer
    criterion = torch.nn.BCEWithLogitsLoss()  # Binary Cross Entropy with logits
    optimizer = torch.optim.Adam(model.parameters(), lr=0.5)  # Use Adam optimizer

    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        logits = model(X_train_tensor)  # This is model.forward()

        # Compute loss
        loss = criterion(logits, Y_train_tensor)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f'Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.4f}')

    model.set_threshold(0.5)

    # Now check predictions
    preds = model.predict(X_test)

    correct_preds = ((preds == Y_test).sum())
    print_and_log(f"Number of correct predictions: {correct_preds}", log_file)
    print_and_log(f"MLP accuracy: {correct_preds / len(Y_test)}", log_file)

    # Sample N_explain number of misclassified points
    id_neg = model.predict(X_test) == 0

    # Ensure id_neg is a PyTorch tensor (if it's still a NumPy array)
    if isinstance(id_neg, np.ndarray):
        id_neg = torch.tensor(id_neg)

    # Assuming X_test is a PyTorch tensor
    X_neg = X_test[id_neg.bool()]  # Get the negative examples
    N_Explain = min(N_explain, X_neg.shape[0])  # Number of examples to explain

    # Randomly select indices
    id_explain = torch.randperm(X_neg.shape[0])[:N_Explain]  # Randomly permute and take the first N_Explain indices

    # Get the corresponding indices from the original id_neg
    id_neg_explain = torch.nonzero(id_neg)[id_explain].squeeze()  # Get the actual negative indices

    # Extract the explanations
    X_explain = X_neg[id_explain]

    # Print and log
    print_and_log(f"Shape of the samples that need explanation: {X_explain.shape}\n", log_file)

    learning_rate_list = [1, 0.75, 0.5, 0.25, 0.1, 0.01]
    print_and_log(f"learning rates we will examine{learning_rate_list}", log_file)

    # Loop through learning rates and log results
    for lr in learning_rate_list:
        print_and_log(f"Learning Rate: {lr}\n", log_file)
        
        # Generate a unique key for this row
        unique_key = str(uuid.uuid4())

        # Run the recourse computation with comparison
        preds, preds_prior, recourse_valid_true, recourse_comparison_valid, cost_dict, interv_estimated, interv_prior, interv_true = find_recourse_mlp_with_comparison(
            model, scmm_prior, scmm_estimated, scmm_true, X_explain, constraints, epsilon, lr
        )

        # Calculate accuracies
        accuracy_preds = sum(preds) / len(preds)
        accuracy_true = sum(recourse_valid_true) / len(recourse_valid_true)


        cost_recourse_true, cost_recourse_estimated, cost_recourse_prior = compute_cost_means(cost_dict)


        # Calculate accuracies
        accuracy_preds_prior = sum(preds_prior) / len(preds_prior)
        accuracy_true = sum(recourse_valid_true) / len(recourse_valid_true)
        accuracy_comparison = sum(recourse_comparison_valid) / len(recourse_comparison_valid)

        

        # Log the information to the results summary CSV file
        with open(csv_file_path, 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow([
                unique_key, true_thetas, lr, epsilon, N_explain, num_ones_train, 
                num_zeros_train, num_ones_test, num_zeros_test, model_type, num_epochs, 
                correct_preds, correct_preds / len(Y_test), X_explain.shape, sum(preds), sum(recourse_valid_true), 
                accuracy_preds, accuracy_true, accuracy_comparison, cost_recourse_true, cost_recourse_estimated, cost_recourse_prior, accuracy_preds_prior
            ])

        # Log interventions to the interventions analysis CSV file
        with open(interv_file_path, 'a', newline='') as intervfile:
            intervwriter = csv.writer(intervfile)
            intervwriter.writerow([lr, interv_estimated, interv_prior, interv_true])


         

        # Also log to the regular text file for backward compatibility (optional)
        print_and_log(f"Accuracy of Recourse valid under estimated SCM in the True SCM: {accuracy_preds}", log_file)
        print_and_log(f"Accuracy of Recourse valid under true SCM: {accuracy_true}", log_file)
        print_and_log(f"Accuracy of Recourse valid under all three SCMs: {accuracy_comparison}", log_file)

        print_and_log(f"Accuracy of Recourse valid under prior SCM in the True SCM: {accuracy_preds_prior}", log_file)
        print_and_log(f"Accuracy of Recourse valid under true SCM: {accuracy_true}", log_file)
        print_and_log(f"Accuracy of Recourse valid under all three SCMs: {accuracy_comparison}", log_file)


        print_and_log("\n\n-----------------------------------------------------------\n\n", log_file)