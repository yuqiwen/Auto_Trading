import math
import numpy as np
import pandas as pd

class DataLoader():

    def __init__(self, filename, split, cols):
        dataframe = pd.read_csv(filename)
        i_split = int(len(dataframe) * split)
        self.data_train = dataframe.get(cols).values[:i_split]
        self.data_test  = dataframe.get(cols).values[i_split:]
        self.len_train  = len(self.data_train)
        self.len_test   = len(self.data_test)
        self.len_train_windows = None

    def get_test_data(self, seq_len, normalise):

        data_windows = []
        for i in range(self.len_test - seq_len):
            data_windows.append(self.data_test[i:i+seq_len])

        data_windows = np.array(data_windows).astype(float)

        data_windows = self.normalise_windows(data_windows, single_window=False) if normalise else data_windows
        x = data_windows[:, :-1] #This line is selecting all windows, but removing the last timestep from each window. So, if your window length is 50 , for each window, you are selecting the first 49 timesteps as your input features.
        y = data_windows[:, -1, [0]] #This line is selecting the last timestep from each window as the target output. Here, the -1 selects the last timestep, and [0] likely means you're selecting the first feature or column of that timestep. If your data only has one feature per timestep (like closing prices), this will just select that value.
        return x, y,data_windows

    def get_train_data(self, seq_len, normalise):

        data_x = []
        data_y = []
        for i in range(self.len_train - seq_len):
            x, y = self._next_window(i, seq_len, normalise)
            data_x.append(x)
            data_y.append(y)
        return np.array(data_x), np.array(data_y)

    def generate_train_batch(self, seq_len, batch_size, normalise):
        
        i = 0
        while i < (self.len_train - seq_len):
            x_batch = []
            y_batch = []
            for b in range(batch_size):
                if i >= (self.len_train - seq_len):
                    # stop-condition for a smaller final batch if data doesn't divide evenly
                    yield np.array(x_batch), np.array(y_batch)
                    i = 0
                x, y = self._next_window(i, seq_len, normalise)
                x_batch.append(x)
                y_batch.append(y)
                i += 1
            yield np.array(x_batch), np.array(y_batch)

    def _next_window(self, i, seq_len, normalise):
        window = self.data_train[i:i+seq_len]
        window = self.normalise_windows(window, single_window=True)[0] if normalise else window
        x = window[:-1]
        y = window[-1, [0]]
        return x, y


    def normalise_windows(self, window_data, single_window=False):
        normalised_data = []
        window_data = [window_data] if single_window else window_data
        prev_window_last_value = None

        for window in window_data:
            normalised_window = []
            for col_i in range(window.shape[1]):
                if prev_window_last_value is not None:
                    # Using the last value of the previous window as the reference
                    ref_value = prev_window_last_value
                else:
                    # If there's no previous window, use the first value of the current window
                    ref_value = window[0, col_i]
                
                # Calculate log returns for the column
                log_returns = np.log(window[:, col_i] / ref_value)
                normalised_window.append(log_returns)
            
            # Save the last value of the current window for the next iteration
            prev_window_last_value = window[-1, col_i]
            normalised_window = np.array(normalised_window).T
            normalised_data.append(normalised_window)

        return np.array(normalised_data)


    def denormalise_windows(self, window_data, normalised_data):
        denormalised_data = []
        for idx in range(len(normalised_data)):
            starting_price = window_data[idx]
            price_ratios = np.exp(np.cumsum(normalised_data[idx]))
            denormalised_prices = starting_price * price_ratios
            denormalised_data.append(denormalised_prices)
        return np.array(denormalised_data)
