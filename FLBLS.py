import numpy as np
from construct_L import construct_L
import matplotlib.pyplot as plt

def LDMBLS(A, Y, A_test=None, Y_test=None, lambda_para1=1e-3, lambda_para2=1e-3, epochs=500, epsilon=1e-1):
    """
    Local Discrimination and Manifold Regularization Broad Learning System (LDMBLS)

    Parameters:
    A: Input feature matrix (n_samples x n_features)
    Y: Label matrix (n_samples x n_classes)
    A_test: Test feature matrix (optional, for calculating test accuracy)
    Y_test: Test label matrix (optional, for calculating test accuracy)
    lambda_para1: Regularization parameter for manifold term
    lambda_para2: Regularization parameter for L2 term
    epochs: Number of epochs
    epsilon: Convergence threshold

    Returns:
    W: Output weight matrix
    """
    n, c = Y.shape
    B = (Y - 1) * 2 + 1  # Transform labels to -1/+1
    gnd = np.argmax(Y, axis=1)  # Get ground truth labels

    # Construct Laplacian matrix L
    L = construct_L(A, gnd)

    # Initialize W0
    t0 = np.dot(A.T, A) + lambda_para1 * np.dot(A.T, L).dot(A) + lambda_para2 * np.eye(A.shape[1])
    W0 = np.linalg.solve(t0, np.dot(A.T, Y))

    losses = []
    train_accuracies = []
    test_accuracies = []

    for i in range(epochs):
        # Optimize M
        P = np.dot(A, W0) - Y
        M = np.maximum(B * P, np.zeros_like(B))
        R = Y + B * M

        # Optimize W
        W = np.linalg.solve(t0, np.dot(A.T, R))

        # Calculate current loss
        t1 = np.dot(A, W) - R
        t2 = t1 * t1
        t3 = np.dot(np.dot(np.dot(np.dot(W.T, A.T), L), A), W)
        t4 = np.trace(t3)
        t5 = W * W
        current_loss = np.sum(t2) + lambda_para1 * t4 + lambda_para2 * np.sum(t5)

        # Save current loss value
        losses.append(current_loss)

        # Calculate training accuracy
        train_accuracy = np.mean(result(np.dot(A, W)) == np.argmax(Y, axis=1))
        train_accuracies.append(train_accuracy)

        # Calculate test accuracy (if test set is provided)
        if A_test is not None and Y_test is not None:
            test_accuracy = np.mean(result(np.dot(A_test, W)) == np.argmax(Y_test, axis=1))
            test_accuracies.append(test_accuracy)
        else:
            test_accuracy = None

        # Print loss and accuracies (every 10 epochs)
        if i % 10 == 0:
            print(f"Epoch {i}, Loss: {current_loss}, Training Accuracy: {train_accuracy * 100:.2f}%",
                  f"Test Accuracy: {test_accuracy * 100:.2f}%" if test_accuracy is not None else "")

        # Check convergence
        if np.trace(np.dot((W - W0).T, (W - W0))) < epsilon:
            print(f"Converged at epoch {i}, Loss: {current_loss}")
            break

        # Check if loss increased (optional)
        if i > 0 and current_loss > losses[-2]:  # Compare with previous loss
            W = W0  # Revert to previous W if loss increased
            print(f"Loss increased at epoch {i}, reverting to previous W")
            break

        W0 = W

    # Plot loss curve
    plt.plot(losses)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('LDMBLS Training Loss')
    plt.show()

    # Plot training accuracy curve
    plt.plot(train_accuracies)
    plt.xlabel('Epochs')
    plt.ylabel('Training Accuracy')
    plt.title('LDMBLS Training Accuracy')
    plt.show()

    # Plot test accuracy curve (if test set is provided)
    if A_test is not None and Y_test is not None:
        plt.plot(test_accuracies)
        plt.xlabel('Epochs')
        plt.ylabel('Test Accuracy')
        plt.title('LDMBLS Test Accuracy')
        plt.show()

    return W

def REMBLS(A, Y, A_test=None, Y_test=None, lambda_para1=1e-3, lambda_para2=1e-3, epochs=500, epsilon=1e-1):
    N,_ = A.shape
    gnd = np.argmax(Y, axis=1)  # Get ground truth labels

    # Construct Laplacian matrix L
    L = construct_L(A, gnd)

    # Initialize W0
    t0 = np.dot(A.T, A) + lambda_para1 * np.dot(A.T, L).dot(A) + lambda_para2 * np.eye(A.shape[1])
    W0 = np.linalg.solve(t0, np.dot(A.T, Y))

    losses = []
    train_accuracies = []
    test_accuracies = []

    for i in range(epochs):
        # Optimize R
        P = np.dot(A, W0)

        R = np.zeros_like(Y)
        for ind in range(N):
            R[ind,:] = optimize_R(P[ind,:], gnd[ind])

        # Optimize W
        W = np.linalg.solve(t0, np.dot(A.T, R))

        # Calculate current loss
        t1 = np.dot(A, W) - R
        t2 = t1 * t1
        t3 = np.dot(np.dot(np.dot(np.dot(W.T, A.T), L), A), W)
        t4 = np.trace(t3)
        t5 = W * W
        current_loss = np.sum(t2) + lambda_para1 * t4 + lambda_para2 * np.sum(t5)

        # Save current loss value
        losses.append(current_loss)

        # Calculate training accuracy
        train_accuracy = np.mean(result(np.dot(A, W)) == np.argmax(Y, axis=1))
        train_accuracies.append(train_accuracy)

        # Calculate test accuracy (if test set is provided)
        if A_test is not None and Y_test is not None:
            test_accuracy = np.mean(result(np.dot(A_test, W)) == np.argmax(Y_test, axis=1))
            test_accuracies.append(test_accuracy)
        else:
            test_accuracy = None

        # Print loss and accuracies (every 10 epochs)
        if i % 10 == 0:
            print(f"Epoch {i}, Loss: {current_loss}, Training Accuracy: {train_accuracy * 100:.2f}%",
                  f"Test Accuracy: {test_accuracy * 100:.2f}%" if test_accuracy is not None else "")

        # Check convergence
        if np.trace(np.dot((W - W0).T, (W - W0))) < epsilon:
            print(f"Converged at epoch {i}, Loss: {current_loss}")
            break

        if i > 0 and current_loss > losses[-2]:  # Compare with previous loss
            W = W0  # Revert to previous W if loss increased
            print(f"Loss increased at epoch {i}, reverting to previous W")
            break

        W0 = W

    # Plot loss curve
    plt.plot(losses)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('REMBLS Training Loss')
    plt.show()

    # Plot training accuracy curve
    plt.plot(train_accuracies)
    plt.xlabel('Epochs')
    plt.ylabel('Training Accuracy')
    plt.title('REMBLS Training Accuracy')
    plt.show()

    # Plot test accuracy curve (if test set is provided)
    if A_test is not None and Y_test is not None:
        plt.plot(test_accuracies)
        plt.xlabel('Epochs')
        plt.ylabel('Test Accuracy')
        plt.title('REMBLS Test Accuracy')
        plt.show()

    return W

def optimize_R(R, label):
    classNum = len(R)
    T = np.zeros(classNum)
    V = R + 1 - np.tile(R[label], classNum)
    step = 0
    num = 0
    for i in range(classNum):
        if i != label:
            dg = V[i]
            for j in range(classNum):
                if j != label:
                    if V[i] < V[j]:
                        dg = dg + V[i] - V[j]
            if dg > 0:
                step = step + V[i]
                num = num + 1
    step = step / (1 + num)
    for i in range(classNum):
        if i == label:
            T[i] = R[i] + step
        else:
            T[i] = R[i] + min(step - V[i], 0)
    return T

def result(x):
    """Helper function to get predicted class labels"""
    return np.argmax(x, axis=1)