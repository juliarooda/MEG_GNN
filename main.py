# %%
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import ray
from ray import train, tune
from dataset import MEGGraphs
from train import train_func, test_best_model
from visualize_graphs import plot_PSD, visualize_graph

def create_dataset():
    '''
    Calls the MEGGraphs class (see dataset.py) to create a dataset of graphs out of raw MEG data. The inputs needed for the MEGGraphs class are defined here. 
    INPUT: N/A
    OUTPUT: 
        - dataset    : Dataset of graphs

    Note: if you want to load multiple files at once, add more filenames to the list. 
    '''

    # Define filenames of (raw) MEG data you want to create graphs out of
    filenames = ["sub-PT06ses04_ses-20181204_task-somatosensory_run-06_meg.ds"]

    # Define the duration of the subepochs you want to create out of the 60-second epochs and the amount of overlap between them
    duration = 30
    overlap = 25

    # Define variables for edge features 
    conn_method = 'pli'
    freqs = {'fmin': 1, 
            'fmax': 40, 
            'freqs': np.linspace(1, 40, (40 - 1) * 4 + 1)}
    
    # Call the MEGGraphs class (see dataset.py)
    dataset = MEGGraphs(root="data\\", 
                        filenames=filenames, 
                        duration=duration,
                        overlap=overlap,
                        conn_method=conn_method,
                        freqs=freqs)
    
    return dataset

def split_train_test(dataset):
    '''
    Splits the Dataset object into a train and test set. 
    INPUT:
        - dataset           : Dataset of graphs
    
    OUTPUTS: 
        - dataset_train     : Dataset of graphs for training
        - dataset_test      : Dataset of graphs for testing 
        - y_train           : list of labels of train set 
        - y_test            : list of labels of test set

    '''
    # Retrieve the labels from all the graphs in the dataset object
    labels = []
    for data in dataset:
        labels.append(data.y)

    # Perform the stratified train test split
    dataset_train, dataset_test, y_train, y_test = train_test_split(dataset, labels, test_size=0.2, random_state=42, stratify=labels)

    return dataset_train, dataset_test, y_train, y_test

def train_hyperparameters(dataset, dataset_train, y_train):
    '''
    Trains the GNN model (see model.py) using the Ray trainable train_func (see train.py). The search space used for the hyperparameter tuning is defined here.
    INPUTS: 
        - dataset           : Dataset of graphs
        - dataset_train     : Dataset of graphs for training 
        - y_train           : list of labels of train set 
    
    OUTPUTS: 
        - results           : ResultGrid object of results of all       hyperparameter configurations
        - best_result       : Result object of results of best hyperparameter configuration
        - best_params       : dictionary of the best hyperparameter configuration
    '''

    # Terminate processes started by ray.init(), so you can define a local _temp_dir to store the Ray process files
    ray.shutdown()
    ray.init(_temp_dir=r"D:\GNN\ray_temp")

    # Make sure Ray doesn't change the working directory to the trial directory, so you can define your own (relative) path to store results 
    os.environ["RAY_CHDIR_TO_TRIAL_DIR"] = "0"

    # Make sure Ray can handle reporting more than one metric 
    os.environ["TUNE_DISABLE_STRICT_METRIC_CHECKING"] = "1"

    # Define hyperparameter search space 
    search_space = {
        'hidden_channels': tune.choice([16, 32, 64, 128]),
        'lr': tune.loguniform(1e-5, 1e-1),
        'batch_size': tune.choice([2, 4, 6, 8])
    }

    # Define path where results need to be stored
    run_config = train.RunConfig(
        storage_path=r"D:\GNN\ray_results"
    )               

    # Define which metric you want Ray to base 'best_results' on, whether that metric needs to be 'max' or 'min', and how many configurations you want it to try
    tune_config = tune.TuneConfig(
        metric='val_accuracy',
        mode='max',
        num_samples=5
    )

    # Perform the training and hyperparameter tuning
    tuner = tune.Tuner(
        tune.with_parameters(train_func, dataset=dataset, dataset_train=dataset_train, y_train=y_train),
        param_space=search_space,
        tune_config=tune_config,
        run_config=run_config
    )

    results = tuner.fit()

    # Retrieve best result
    best_result = results.get_best_result()

    # Retrieve hyperparameter configuration of best result
    best_params = best_result.config
    return results, best_result, best_params

def plot_train_results(best_result):
    '''
    Plots the training results by plotting the losses and accuracies of both the train and validation set. 
    INPUT: N/A
    OUTPUT: N/A
    '''

    # Plot losses
    plt.figure(figsize=(12, 5))

    plt.subplot(121)
    plt.plot(best_result.metrics_dataframe['training_iteration'], 
             best_result.metrics_dataframe['train_loss'], 
             label='Train loss')
    plt.plot(best_result.metrics_dataframe['training_iteration'], 
             best_result.metrics_dataframe['val_loss'], 
             label='Validation loss')
    plt.legend()
    plt.ylabel('Loss')
    plt.xlabel('Training iteration')
    plt.title('Training and Validation Loss')
    plt.grid(True)

    # Plot accuracies 
    plt.subplot(122)
    plt.plot(best_result.metrics_dataframe['training_iteration'], 
             best_result.metrics_dataframe['train_accuracy'], 
             label='Train accuracy')
    plt.plot(best_result.metrics_dataframe['training_iteration'], 
             best_result.metrics_dataframe['val_accuracy'], 
             label='Validation accuracy')
    plt.legend()
    plt.ylabel('Accuracy')
    plt.xlabel('Training iteration')
    plt.title('Training and Validation Accuracy')
    plt.grid(True)

def main():
    '''
    Calls all functions in this script. 
    INPUT: N/A
    OUTPUT: N/A
    '''
    dataset = create_dataset()
    plot_PSD(dataset)
    visualize_graph(dataset)
    dataset_train, dataset_test, y_train, y_test = split_train_test(dataset)
    results, best_result, best_params = train_hyperparameters(dataset, dataset_train, y_train)
    plot_train_results(best_result)
    acc_test = test_best_model(best_result, dataset, dataset_test)
    print(f'Test accuracy: {acc_test}')

if __name__ == "__main__":
    '''
    Runs the main function. 
    '''
    main()
