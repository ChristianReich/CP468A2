import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from preceptron import Preceptron

df = pd.read_csv('WeatherData_Q4.csv')

def plot_data(df):
    # Split the data for plotting
    rain = df[df['rain'] == 1]
    no_rain = df[df['rain'] == 0]

    # Plot both sets of points and add labels
    plt.scatter(rain['temp'], rain['humid'], color='red', marker='o')
    plt.scatter(no_rain['temp'], no_rain['humid'], color='blue', marker='s')
    plt.xlabel("Temperature (°C)")
    plt.ylabel("Humidity (%)")
    plt.title("Rainy Days: Temperature vs Humidity")
    plt.show()
    return

def plot_data_db(df, p):
    # Split the data for plotting
    rain = df[df['rain'] == 1]
    no_rain = df[df['rain'] == 0]

    x = df[['temp', 'humid']].values

    # Create decision boundary based on weights from the preceptron
    x_vals = np.linspace(min(x[:,0]) - 1, max(x[:,0]) + 1, 100)
    y_vals = -(p.weights[0] + p.weights[1] * x_vals) / p.weights[2]
    # Plot decision boundary
    plt.plot(x_vals, y_vals, 'k--', label='Decision Boundary')
    # Plot both sets of points and add labels
    plt.scatter(rain['temp'], rain['humid'], color='red', marker='o', label="rain")
    plt.scatter(no_rain['temp'], no_rain['humid'], color='blue', marker='s', label="no rain")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("Temperature (°C)")
    plt.ylabel("Humidity (%)")
    plt.title("Rainy Days: Temperature vs Humidity")
    plt.legend()
    plt.show()

def main():
    plot_data(df)
    
    # Split data for x/y test and train sets 
    x = df[['temp', 'humid']].values
    y = df['rain'].values
    X_train = x[:15]
    y_train = y[:15]
    X_test = x[15:]
    y_test = y[15:]

    # Create preceptron model
    p = Preceptron()
    # Train model
    p.train(X_train, y_train)

    # Loop through test sets and record results 
    results = []
    for i in range(len(X_test)):
        result = p.predict(X_test[i])
        results.append(result)
    acc_score = np.mean(results == y_test) # Compare predictions to actual answers
    print("Final Accuracy Score: {}%".format(acc_score * 100))

    # Plot decision boundary
    plot_data_db(df, p)

    return

if __name__ == "__main__":
    main()