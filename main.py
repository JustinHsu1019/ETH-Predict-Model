""" Original MLP """
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.math import argmax

# 函數：從股票代碼和歷史年數取得數據
def get_stock_data(stock_code, history_period):
    stock_data = yf.Ticker(stock_code).history(period=history_period, interval="1d")
    stock_data = stock_data.drop(columns=["Dividends", "Stock Splits"])
    stock_data = stock_data[:-1]
    return stock_data

# 函數：創建模型
def create_model(timesteps, features, num_classes):
    input_layer = Input(shape=(timesteps, features))
    x = Flatten()(input_layer)
    x = Dense(256, activation="relu")(x)
    x = Dense(256, activation="relu")(x)
    output_layer = Dense(num_classes, activation="softmax")(x)
    model = Model(inputs=input_layer, outputs=output_layer)
    return model

# 函數：處理數據，創建特徵集和標籤集
def process_data(stock_data, past_window_length):
    categories = ["Bull", "Bear"]
    label_bull = categories.index("Bull")
    label_bear = categories.index("Bear")

    features, labels = [], []
    for today_index in range(len(stock_data)):
        past_data = stock_data[:today_index + 1]
        future_data = stock_data[today_index + 1:]

        if (len(past_data) < past_window_length) or (len(future_data) < 1):
            continue

        past_window = past_data[-past_window_length:]
        future_window = future_data[:1]

        today_close_price = past_window.iloc[-1]["Close"]
        next_day_close_price = future_window.iloc[0]["Close"]

        label = label_bull if next_day_close_price > today_close_price else label_bear

        features.append(past_window.values)
        labels.append(label)

    return np.array(features), np.array(labels)

# 函數：執行訓練流程並返回準確率
def train_and_evaluate(features_train, labels_train, features_val, labels_val, features_test, labels_test, num_classes):
    model = create_model(features_train.shape[1], features_train.shape[2], num_classes)
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    model.fit(features_train, to_categorical(labels_train),
              validation_data=(features_val, to_categorical(labels_val)),
              batch_size=2048, epochs=1000, verbose=0)

    predictions = argmax(model.predict(features_test), axis=-1)
    accuracy = np.sum(predictions == labels_test) / len(labels_test)
    return accuracy

# 主迴圈
accuracies = []
for i in range(2):
    stock_data = get_stock_data("ETH-USD", "10y")
    特徵集, 標籤集 = process_data(stock_data, 100)
    過去窗口長度 = 100  # 定義過去的數據窗口長度
    類別 = ["Bull", "Bear"]  # 定義類別為漲(Bull)和跌(Bear)
    標籤_漲 = 類別.index("Bull")  # 獲取漲的索引值 --> 標籤_漲 = 0
    標籤_跌 = 類別.index("Bear")  # 獲取跌的索引值 --> 標籤_跌 = 1

    訓練集比例, 驗證集比例, 測試集比例 = 0.7, 0.2, 0.1

    # 取最後一部分為測試數據集
    測試分割索引 = -round(len(特徵集) * 測試集比例)
    特徵集其他, 特徵集測試 = np.split(特徵集, [測試分割索引])
    標籤集其他, 標籤集測試 = np.split(標籤集, [測試分割索引])

    # 打亂剩餘部分並分割成訓練和驗證數據集
    訓練分割索引 = round(len(特徵集其他) * 訓練集比例)
    索引 = np.arange(len(特徵集其他))
    np.random.shuffle(索引)  # 隨機打亂索引
    訓練索引, 驗證索引 = np.split(索引, [訓練分割索引])
    特徵集訓練, 特徵集驗證 = 特徵集其他[訓練索引], 特徵集其他[驗證索引]
    標籤集訓練, 標籤集驗證 = 標籤集其他[訓練索引], 標籤集其他[驗證索引]

    # 平衡測試數據集的標籤
    特徵集測試_漲 = 特徵集測試[標籤集測試 == 標籤_漲]
    特徵集測試_跌 = 特徵集測試[標籤集測試 == 標籤_跌]

    最小標籤數 = min(len(特徵集測試_漲), len(特徵集測試_跌)) # min(126, 116) 最小標籤數=116

    特徵集測試_漲 = 特徵集測試_漲[np.random.choice(len(特徵集測試_漲), 最小標籤數, replace=False), :]
    特徵集測試_跌 = 特徵集測試_跌[np.random.choice(len(特徵集測試_跌), 最小標籤數, replace=False), :]
    特徵集測試 = np.vstack([特徵集測試_漲, 特徵集測試_跌])

    標籤集測試 = np.array([標籤_漲] * 最小標籤數 + [標籤_跌] * 最小標籤數)

    # 進行訓練和評估
    accuracy = train_and_evaluate(特徵集訓練, 標籤集訓練, 特徵集驗證, 標籤集驗證, 特徵集測試, 標籤集測試, len(["Bull", "Bear"]))
    accuracies.append(accuracy)

# 計算每個精度值的頻率
accuracy_counts = pd.value_counts(accuracies)
sorted_accuracies = np.sort(np.unique(accuracies))
frequency = [accuracy_counts[acc] for acc in sorted_accuracies]

# 產生線圖
plt.plot(sorted_accuracies, frequency, marker='o', linestyle='-', color='b', label='Multilayer Perceptron')

# 添加標籤和標題
plt.title('Multilayer Perceptron Accuracy Distribution')
plt.xlabel('Accuracy')
plt.ylabel('Frequency')
plt.legend()
plt.show()

""" InceptionTime """
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.math import argmax

# 函數：從股票代碼和歷史年數取得數據
def get_stock_data(stock_code, history_period):
    stock_data = yf.Ticker(stock_code).history(period=history_period, interval="1d")
    stock_data = stock_data.drop(columns=["Dividends", "Stock Splits"])
    stock_data = stock_data[:-1]
    return stock_data

from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Concatenate, \
                    BatchNormalization, Activation, \
                    Add, GlobalAveragePooling1D, Dense, Dropout
from tensorflow.keras.models import Model

# 定義一個 Inception 模組
def inception_module(輸入張量):
    # 1x1 瓶頸層
    瓶頸 = Conv1D(filters=32, kernel_size=1, padding="same", activation=None,
              use_bias=False)(輸入張量)
    # 3x3 卷積層
    conv3 = Conv1D(filters=32, kernel_size=3, padding="same", activation=None,
            use_bias=False)(瓶頸)
    # 5x5 卷積層
    conv5 = Conv1D(filters=32, kernel_size=5, padding="same", activation=None,
            use_bias=False)(瓶頸)
    # 7x7 卷積層
    conv7 = Conv1D(filters=32, kernel_size=7, padding="same", activation=None,
            use_bias=False)(瓶頸)
    # 最大池化層
    mp = MaxPooling1D(pool_size=3, strides=1, padding="same")(輸入張量)
    # 池化後的瓶頸層
    mp瓶頸 = Conv1D(filters=32, kernel_size=1, padding="same", activation=None,
            use_bias=False)(mp)
    # 合併層
    x = Concatenate(axis=-1)([conv3, conv5, conv7, mp瓶頸])
    # 批量標準化
    x = BatchNormalization()(x)
    # 激活函數
    x = Activation("relu")(x)
    return x

# 定義捷徑層，實現殘差學習
def shortcut_layer(輸入張量1, 輸入張量2):
    # 捷徑卷積層，以適配張量維度
    shortcut = Conv1D(filters=輸入張量2.shape[-1], kernel_size=1, padding="same",
              activation=None, use_bias=False)(輸入張量1)
    # 批量標準化
    shortcut = BatchNormalization()(shortcut)
    # 張量相加
    x = Add()([shortcut, 輸入張量2])
    # 激活函數
    x = Activation("relu")(x)
    return x

# 函數：創建模型
def create_model(timesteps, features, num_classes):
    # 構建模型
    時間步長 = timesteps
    特徵數 = features

    # 輸入層
    輸入層 = Input(shape=(時間步長, 特徵數))
    x = 輸入層
    輸入殘差 = 輸入層

    # 迴圈添加 6 個 Inception 模組，每 3 個之後添加一個捷徑層
    for 索引 in range(6):
      x = inception_module(x)
      x = Dropout(0.2)(x)

      if (索引 % 3 == 2):
        x = shortcut_layer(輸入殘差, x)
        輸入殘差 = x

    # 全局平均池化層
    x = GlobalAveragePooling1D()(x)
    # 輸出層
    輸出層 = Dense(num_classes, activation="softmax")(x)

    # 定義模型的輸入和輸出
    模型 = Model(inputs=輸入層, outputs=輸出層)

    return 模型
    # 打印模型結構
    # 模型.summary()

# 函數：處理數據，創建特徵集和標籤集
def process_data(stock_data, past_window_length):
    categories = ["Bull", "Bear"]
    label_bull = categories.index("Bull")
    label_bear = categories.index("Bear")

    features, labels = [], []
    for today_index in range(len(stock_data)):
        past_data = stock_data[:today_index + 1]
        future_data = stock_data[today_index + 1:]

        if (len(past_data) < past_window_length) or (len(future_data) < 1):
            continue

        past_window = past_data[-past_window_length:]
        future_window = future_data[:1]

        today_close_price = past_window.iloc[-1]["Close"]
        next_day_close_price = future_window.iloc[0]["Close"]

        label = label_bull if next_day_close_price > today_close_price else label_bear

        features.append(past_window.values)
        labels.append(label)

    return np.array(features), np.array(labels)

# 函數：執行訓練流程並返回準確率
def train_and_evaluate(features_train, labels_train, features_val, labels_val, features_test, labels_test, num_classes):
    model = create_model(features_train.shape[1], features_train.shape[2], num_classes)
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    model.fit(features_train, to_categorical(labels_train),
              validation_data=(features_val, to_categorical(labels_val)),
              batch_size=2048, epochs=1000, verbose=0)

    predictions = argmax(model.predict(features_test), axis=-1)
    accuracy = np.sum(predictions == labels_test) / len(labels_test)
    return accuracy

# 主迴圈
accuracies_2 = []
for i in range(100):
    stock_data = get_stock_data("0700.HK", "10y")
    特徵集, 標籤集 = process_data(stock_data, 100)
    過去窗口長度 = 100  # 定義過去的數據窗口長度
    類別 = ["Bull", "Bear"]  # 定義類別為漲(Bull)和跌(Bear)
    標籤_漲 = 類別.index("Bull")  # 獲取漲的索引值 --> 標籤_漲 = 0
    標籤_跌 = 類別.index("Bear")  # 獲取跌的索引值 --> 標籤_跌 = 1

    訓練集比例, 驗證集比例, 測試集比例 = 0.7, 0.2, 0.1

    # 取最後一部分為測試數據集
    測試分割索引 = -round(len(特徵集) * 測試集比例)
    特徵集其他, 特徵集測試 = np.split(特徵集, [測試分割索引])
    標籤集其他, 標籤集測試 = np.split(標籤集, [測試分割索引])

    # 打亂剩餘部分並分割成訓練和驗證數據集
    訓練分割索引 = round(len(特徵集其他) * 訓練集比例)
    索引 = np.arange(len(特徵集其他))
    np.random.shuffle(索引)  # 隨機打亂索引
    訓練索引, 驗證索引 = np.split(索引, [訓練分割索引])
    特徵集訓練, 特徵集驗證 = 特徵集其他[訓練索引], 特徵集其他[驗證索引]
    標籤集訓練, 標籤集驗證 = 標籤集其他[訓練索引], 標籤集其他[驗證索引]

    # 平衡測試數據集的標籤
    特徵集測試_漲 = 特徵集測試[標籤集測試 == 標籤_漲]
    特徵集測試_跌 = 特徵集測試[標籤集測試 == 標籤_跌]

    最小標籤數 = min(len(特徵集測試_漲), len(特徵集測試_跌)) # min(126, 116) 最小標籤數=116

    特徵集測試_漲 = 特徵集測試_漲[np.random.choice(len(特徵集測試_漲), 最小標籤數, replace=False), :]
    特徵集測試_跌 = 特徵集測試_跌[np.random.choice(len(特徵集測試_跌), 最小標籤數, replace=False), :]
    特徵集測試 = np.vstack([特徵集測試_漲, 特徵集測試_跌])

    標籤集測試 = np.array([標籤_漲] * 最小標籤數 + [標籤_跌] * 最小標籤數)

    # 進行訓練和評估
    accuracy = train_and_evaluate(特徵集訓練, 標籤集訓練, 特徵集驗證, 標籤集驗證, 特徵集測試, 標籤集測試, len(["Bull", "Bear"]))
    accuracies_2.append(accuracy)

# 計算每個精度值的頻率
accuracy_counts = pd.value_counts(accuracies_2)
sorted_accuracies = np.sort(np.unique(accuracies_2))
frequency = [accuracy_counts[acc] for acc in sorted_accuracies]

# 產生線圖
plt.plot(sorted_accuracies, frequency, marker='o', linestyle='-', color='b', label='Multilayer Perceptron')

# 添加標籤和標題
plt.title('Multilayer Perceptron Accuracy Distribution')
plt.xlabel('Accuracy')
plt.ylabel('Frequency')
plt.legend()
plt.show()

""" InceptionTime+ """
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.math import argmax

# 函數：從股票代碼和歷史年數取得數據
def get_stock_data(stock_code, history_period):
    stock_data = yf.Ticker(stock_code).history(period=history_period, interval="1d")
    stock_data = stock_data.drop(columns=["Dividends", "Stock Splits"])
    stock_data = stock_data[:-1]
    return stock_data

from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Concatenate, \
                    BatchNormalization, Activation, \
                    Add, GlobalAveragePooling1D, Dense, Dropout
from tensorflow.keras.models import Model

# 定義一個 Inception 模組
def inception_module(輸入張量):
    # 1x1 卷積作為瓶頸層，減少參數數量
    瓶頸 = Conv1D(filters=32, kernel_size=1, padding="same", activation=None,
              use_bias=False)(輸入張量)
    # 3x3 卷積層
    conv3 = Conv1D(filters=32, kernel_size=3, padding="same", activation=None,
            use_bias=False)(瓶頸)
    # 5x5 卷積層
    conv5 = Conv1D(filters=32, kernel_size=5, padding="same", activation=None,
            use_bias=False)(瓶頸)
    # 7x7 卷積層
    conv7 = Conv1D(filters=32, kernel_size=7, padding="same", activation=None,
            use_bias=False)(瓶頸)
    # 最大池化層
    mp = MaxPooling1D(pool_size=3, strides=1, padding="same")(輸入張量)
    # 池化後的瓶頸層
    mp瓶頸 = Conv1D(filters=32, kernel_size=1, padding="same", activation=None,
            use_bias=False)(mp)
    # 合併層
    x = Concatenate(axis=-1)([conv3, conv5, conv7, mp瓶頸])
    # 批量標準化
    x = BatchNormalization()(x)
    # 激活函數
    x = Activation("LeakyReLU")(x)
    return x

# 定義捷徑層，實現殘差學習
def shortcut_layer(輸入張量1, 輸入張量2):
    # 捷徑卷積層，以適配張量維度
    shortcut = Conv1D(filters=輸入張量2.shape[-1], kernel_size=1, padding="same",
              activation=None, use_bias=False)(輸入張量1)
    # 批量標準化
    shortcut = BatchNormalization()(shortcut)
    # 張量相加
    x = Add()([shortcut, 輸入張量2])
    # 激活函數
    x = Activation("LeakyReLU")(x)
    return x

# 函數：創建模型
def create_model(timesteps, features, num_classes):
    # 構建模型
    時間步長 = 特徵集訓練.shape[1]
    特徵數 = 特徵集訓練.shape[2]

    # 輸入層
    輸入層 = Input(shape=(時間步長, 特徵數))
    x = 輸入層
    輸入殘差 = 輸入層

    # 迴圈添加 6 個 Inception 模組，每 3 個之後添加一個捷徑層
    for 索引 in range(6):
      x = inception_module(x)

      if (索引 % 3 == 2):
        x = shortcut_layer(輸入殘差, x)
        輸入殘差 = x

    # 全局平均池化層
    x = GlobalAveragePooling1D()(x)
    # 輸出層
    輸出層 = Dense(len(類別), activation="softmax")(x)

    # 定義模型的輸入和輸出
    模型 = Model(inputs=輸入層, outputs=輸出層)

    # 打印模型結構
    # 模型.summary()
    return 模型

# 函數：處理數據，創建特徵集和標籤集
def process_data(stock_data, past_window_length):
    categories = ["Bull", "Bear"]
    label_bull = categories.index("Bull")
    label_bear = categories.index("Bear")

    features, labels = [], []
    for today_index in range(len(stock_data)):
        past_data = stock_data[:today_index + 1]
        future_data = stock_data[today_index + 1:]

        if (len(past_data) < past_window_length) or (len(future_data) < 1):
            continue

        past_window = past_data[-past_window_length:]
        future_window = future_data[:1]

        today_close_price = past_window.iloc[-1]["Close"]
        next_day_close_price = future_window.iloc[0]["Close"]

        label = label_bull if next_day_close_price > today_close_price else label_bear

        features.append(past_window.values)
        labels.append(label)

    return np.array(features), np.array(labels)

# 函數：執行訓練流程並返回準確率
def train_and_evaluate(features_train, labels_train, features_val, labels_val, features_test, labels_test, num_classes):
    # 導入所需的模組和函數
    from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
    from tensorflow.keras.utils import to_categorical

    模型 = create_model(features_train.shape[1], features_train.shape[2], num_classes)

    # 編譯模型
    模型.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    訓練標籤計數 = 標籤分佈.iloc[0]
    類別權重 = {
        標籤_漲: 1.,
        標籤_跌: 訓練標籤計數["漲"] / 訓練標籤計數["跌"]
    }

    # 設定回呼函數
    reduce_1r = ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=30, min_1r=0.00001)
    模型檢查點 = ModelCheckpoint(filepath="best_model.hdf5", monitor="val_loss",
                      save_best_only=True)
    提前停止 = EarlyStopping(monitor="val_loss", patience=100, restore_best_weights=True)
    回呼函數 = [reduce_1r, 模型檢查點, 提前停止]

    # 訓練模型
    模型.fit(features_train, to_categorical(labels_train),
                  class_weight=類別權重,
                  validation_data=(features_val, to_categorical(labels_val)),
                  batch_size=2048, epochs=1000, callbacks=[回呼函數])

    predictions = argmax(模型.predict(features_test), axis=-1)
    accuracy = np.sum(predictions == labels_test) / len(labels_test)
    return accuracy

# 主迴圈
accuracies_3 = []
for i in range(100):
    stock_data = get_stock_data("0700.HK", "10y")
    特徵集, 標籤集 = process_data(stock_data, 100)
    過去窗口長度 = 100  # 定義過去的數據窗口長度
    類別 = ["Bull", "Bear"]  # 定義類別為漲(Bull)和跌(Bear)
    標籤_漲 = 類別.index("Bull")  # 獲取漲的索引值 --> 標籤_漲 = 0
    標籤_跌 = 類別.index("Bear")  # 獲取跌的索引值 --> 標籤_跌 = 1

    訓練集比例, 驗證集比例, 測試集比例 = 0.7, 0.2, 0.1

    # 取最後一部分為測試數據集
    測試分割索引 = -round(len(特徵集) * 測試集比例)
    特徵集其他, 特徵集測試 = np.split(特徵集, [測試分割索引])
    標籤集其他, 標籤集測試 = np.split(標籤集, [測試分割索引])

    # 打亂剩餘部分並分割成訓練和驗證數據集
    訓練分割索引 = round(len(特徵集其他) * 訓練集比例)
    索引 = np.arange(len(特徵集其他))
    np.random.shuffle(索引)  # 隨機打亂索引
    訓練索引, 驗證索引 = np.split(索引, [訓練分割索引])
    特徵集訓練, 特徵集驗證 = 特徵集其他[訓練索引], 特徵集其他[驗證索引]
    標籤集訓練, 標籤集驗證 = 標籤集其他[訓練索引], 標籤集其他[驗證索引]

    # 平衡測試數據集的標籤
    特徵集測試_漲 = 特徵集測試[標籤集測試 == 標籤_漲]
    特徵集測試_跌 = 特徵集測試[標籤集測試 == 標籤_跌]

    最小標籤數 = min(len(特徵集測試_漲), len(特徵集測試_跌)) # min(126, 116) 最小標籤數=116

    特徵集測試_漲 = 特徵集測試_漲[np.random.choice(len(特徵集測試_漲), 最小標籤數, replace=False), :]
    特徵集測試_跌 = 特徵集測試_跌[np.random.choice(len(特徵集測試_跌), 最小標籤數, replace=False), :]
    特徵集測試 = np.vstack([特徵集測試_漲, 特徵集測試_跌])

    標籤集測試 = np.array([標籤_漲] * 最小標籤數 + [標籤_跌] * 最小標籤數)

    標籤分佈 = pd.DataFrame([
        {
          "數據集": "訓練",
          "漲": np.count_nonzero(標籤集訓練 == 標籤_漲),  # 計算訓練集中漲的數量
          "跌": np.count_nonzero(標籤集訓練 == 標籤_跌)   # 計算訓練集中跌的數量
        },
        {
          "數據集": "驗證",
          "漲": np.count_nonzero(標籤集驗證 == 標籤_漲),  # 計算驗證集中漲的數量
          "跌": np.count_nonzero(標籤集驗證 == 標籤_跌)   # 計算驗證集中跌的數量
        },
        {
          "數據集": "測試",
          "漲": np.count_nonzero(標籤集測試 == 標籤_漲),  # 計算測試集中漲的數量
          "跌": np.count_nonzero(標籤集測試 == 標籤_跌)   # 計算測試集中跌的數量
        }
    ])

    # 進行訓練和評估
    accuracy = train_and_evaluate(特徵集訓練, 標籤集訓練, 特徵集驗證, 標籤集驗證, 特徵集測試, 標籤集測試, len(["Bull", "Bear"]))
    accuracies_3.append(accuracy)

# 計算每個精度值的頻率
accuracy_counts = pd.value_counts(accuracies_3)
sorted_accuracies = np.sort(np.unique(accuracies_3))
frequency = [accuracy_counts[acc] for acc in sorted_accuracies]

# 產生線圖
plt.plot(sorted_accuracies, frequency, marker='o', linestyle='-', color='b', label='Multilayer Perceptron')

# 添加標籤和標題
plt.title('Multilayer Perceptron Accuracy Distribution')
plt.xlabel('Accuracy')
plt.ylabel('Frequency')
plt.legend()
plt.show()

""" 疊圖 """
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 假設您有三個模型的準確率數據
accuracies_model_1 = accuracies
accuracies_model_2 = accuracies_2
accuracies_model_3 = accuracies_3

# 將這些準確率數據轉換為頻率數據
def accuracies_to_frequency(accuracies):
    accuracy_counts = pd.value_counts(accuracies)
    sorted_accuracies = np.sort(np.unique(accuracies))
    frequency = [accuracy_counts[acc] for acc in sorted_accuracies]
    return sorted_accuracies, frequency

# 為每個模型計算準確率和頻率
sorted_accuracies_1, frequency_1 = accuracies_to_frequency(accuracies_model_1)
sorted_accuracies_2, frequency_2 = accuracies_to_frequency(accuracies_model_2)
sorted_accuracies_3, frequency_3 = accuracies_to_frequency(accuracies_model_3)

# 在同一張圖上繪製每個模型的準確率分佈
plt.plot(sorted_accuracies_1, frequency_1, marker='o', linestyle='-', color='b', label='Original MLP')
plt.plot(sorted_accuracies_2, frequency_2, marker='x', linestyle='-', color='r', label='InceptionTime')
plt.plot(sorted_accuracies_3, frequency_3, marker='+', linestyle='-', color='g', label='InceptionTime+')

# 添加圖表的標籤和標題
plt.title('Accuracy Distribution Comparison')
plt.xlabel('Accuracy')
plt.ylabel('Frequency')
plt.legend()

# 顯示圖表
plt.show()
