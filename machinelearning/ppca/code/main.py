import numpy as np
import matplotlib.pyplot as plt
import pickle
import random
from models import ppca
from pathlib import Path


def load_dataset(filename, *keys):
    """ 
    
        Loads dataset mnsit35

        Reference: code from CPSC 540 assignment 2 2023 winter session
    
    """

    if not filename.endswith(".pkl"):
        filename = filename + ".pkl"
    with open(Path("..", "data", filename), "rb") as (f):
        data = pickle.load(f)
        if not keys:
            return data
        return [data[k] for k in keys]


def get_relative_error(
        
        X,
        model, 
        num_points,
        y=None, 
        plot=True, 
        class_labels=['3', '5']
        
    ):
    """ 

        evaluate the model

    """

    fitted_data = np.reshape(X, (X.shape[0], -1))

    reduced_data = model.transform(fitted_data)
    created_data = model.inverse_transform(reduced_data)
    err = []
    
    for i in range(int(num_points) - 1):
       
        error_per_point = np.linalg.norm(created_data[i] - X[i], ord=2) / (
            np.linalg.norm(X[i], ord=2))
        error = error_per_point * 100
        err.append(error)
    
    reconstruction_error = np.array(err)

    if plot:

        # visualize a sample of reconstructed data images
        created_data = np.reshape(created_data, (created_data.shape[0], 28, 28))


        # randomly select 5 images from 1 to num_pics_to_load
        rand_Images_idx = random.sample(range(num_points), 6)

        fig, axs = plt.subplots(2, 3, figsize=(12, 6))
        col = 0
        row = 0

        for i, idx in enumerate(rand_Images_idx):
            
            axs[row, col].imshow(created_data[i].real)
            axs[row, col].set_xlabel("Actual Number {}".format(class_labels[y[i]]))

            if col == 2:

                row += 1
                col = 0

            else:

                col += 1

        plt.show()

    return reconstruction_error


def main():
    """
    
        Script that runs A1 of the assignment using PPCA on the mnist35
    
    """  
    
    f = "mnist35.pkl"

    X, y, Xtest, ytest = load_dataset(f, "X", "y", "Xtest", "ytest")

    # determine samples
    num_samples = X.shape[0]

    # y = y[:10000]
    Xtest = np.reshape(Xtest, (Xtest.shape[0], -1))
    # ytest = ytest[:10000]
    model = ppca(X=X, num_components=200)

    print("=======>Training Phase<=======")
    error_train = get_relative_error(X, model, num_samples, y=y, plot=False)
    print("The avg error of the dataset is: {0}\n\n".format(np.mean(error_train)))

    print("=======>Testing Phase<=======")
    error_test = get_relative_error(Xtest, model, Xtest.shape[0], y=ytest)
    print("The testing avg error of the dataset is: {0}".format(np.mean(error_test)))

if __name__ == "__main__":
    main()
