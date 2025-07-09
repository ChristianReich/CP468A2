import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('WeatherData_Q4.csv')

def plot_data(df):
    rain = df[df['rain'] == 1]
    no_rain = df[df['rain'] == 0]

    plt.scatter(rain['temp'], rain['humid'], color='red', marker='o')
    plt.scatter(no_rain['temp'], no_rain['humid'], color='blue', marker='s')
    plt.xlabel("Temperature (°C)")
    plt.ylabel("Humidity (%)")
    plt.title("Rainy Days: Temperature vs Humidity")
    plt.show()
    return

def main():
    plot_data(df)
    return

if __name__ == "__main__":
    main()