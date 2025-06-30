import matplotlib.pyplot as plt

def plot_predictions(dates, actual, predicted):
    plt.figure(figsize=(10,5))
    plt.plot(dates, actual, label='Actual')
    plt.plot(dates, predicted, label='Predicted')
    plt.xlabel('Date')
    plt.ylabel('Temperature')
    plt.title('Test Set Predictions')
    plt.legend()
    plt.tight_layout()
    plt.show()
