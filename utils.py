import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
'''
전처리
'''
# 이상치 탐지 및 처리 함수 수정
def handle_outliers(data):
    columns_to_interpolate = [
        # 'fog_train.ws10_deg',
        'fog_train.ws10_ms',
        'fog_train.ta',
        'fog_train.hm',
        'fog_train.sun10',
        'fog_train.ts'
    ]
    
    # 공통 처리: -99.9 또는 -99를 NaN으로 대체하고 클래스가 4가 아닌 경우 선형 보간
    for column in columns_to_interpolate:
        data[column].replace(-99.9, np.nan, inplace=True)
        data.loc[(data['fog_train.class'] != 4) & (data[column].isna()), column] = data[column].interpolate(method='linear')
        data = data[~((data['fog_train.class'] == 4) & (data[column].isna()))]
    
    # # 'fog_train.ws10_deg'의 추가 처리 (0도 너무 많아서 보간법으로 처리)
    # data['fog_train.ws10_deg'].replace(0, np.nan, inplace=True)
    # data['fog_train.ws10_deg'] = data['fog_train.ws10_deg'].interpolate(method='linear')
    
    # 'fog_train.re'는 -99.9를 drop
    data = data[data['fog_train.re'] != -99.9]

    # 'fog_train.class'는 -99를 drop
    data = data[data['fog_train.class'] != -99]

    return data

# 'I', 'J', 'K'를 연도 값으로 변환하는 함수
def convert_year(year):
    year = np.where(year == 'I', 2020, year)
    year = np.where(year == 'J', 2021, year)
    year = np.where(year == 'K', 2022, year)
    return year

# 계절 추가 함수
def get_season(month):
    if month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    elif month in [9, 10, 11]:
        return 'Fall'
    else:
        return 'Winter'
# 'fog_train.day'를 카테고리로 변환하는 함수
def categorize_day(day):
    if day <= 10:
        return 'Early'
    elif day <= 20:
        return 'Mid'
    else:
        return 'Late'
'''
평가 척도
'''
# 다중 CSI 계산을 위한 함수 정의
def calculate_csi(y_true, y_pred):
    H11 = H22 = H33 = 0
    F12 = F13 = F21 = F23 = F31 = F32 = 0
    M14 = M24 = M34 = 0
    C44 = F41 = F42 = F43 = 0
    # y_true = y_true+1
    # y_pred = y_pred+1
    
    for true, pred in zip(y_true, y_pred):
        if true == 1:  # 0 < 시정 < 200
            if pred == 1:
                H11 += 1
            elif pred == 2:
                F12 += 1
            elif pred == 3:
                F13 += 1
            else:  # pred == 4
                M14 += 1
        elif true == 2:  # 200 ≤ 시정 < 500
            if pred == 1:
                F21 += 1
            elif pred == 2:
                H22 += 1
            elif pred == 3:
                F23 += 1
            else:  # pred == 4
                M24 += 1
        elif true == 3:  # 500 ≤ 시정 < 1000
            if pred == 1:
                F31 += 1
            elif pred == 2:
                F32 += 1
            elif pred == 3:
                H33 += 1
            else:  # pred == 4
                M34 += 1
        else:  # true == 4 1000 ≤ 시정
            if pred == 1:
                F41 += 1
            elif pred == 2:
                F42 += 1
            elif pred == 3:
                F43 += 1
            else:  # pred == 4
                C44 += 1
    
    H = H11 + H22 + H33
    F = F12 + F13 + F21 + F23 + F31 + F32 + F41 + F42 + F43
    M = M14 + M24 + M34
    
    CSI = H / (H + F + M)
    return CSI

def classify_vis1(predictions):
    """
    Classify predicted fog_train.vis1 values into categories:
    0 이상 200 미만: 1
    200 이상 500 미만: 2
    500 이상 1000 미만: 3
    1000 이상: 4
    
    Parameters:
    predictions (pd.Series or np.array): The predicted values of fog_train.vis1.
    
    Returns:
    pd.Series: A new series with the classified categories.
    """
    conditions = [
        (predictions >= 0) & (predictions < 200),
        (predictions >= 200) & (predictions < 500),
        (predictions >= 500) & (predictions < 1000),
        (predictions >= 1000)
    ]
    
    classes = [1, 2, 3, 4]
    
    return pd.Series(np.select(conditions, classes, default=np.nan), index=predictions.index)

'''
시각화
'''

# 각 열의 분포 시각화
def plot_numerical(data, col):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.histplot(data[col], ax=axes[0], kde=True)
    axes[0].set_title(f'Histogram of {col}')
    sns.boxplot(x=data[col], ax=axes[1])
    axes[1].set_title(f'Boxplot of {col}')
    plt.show()

def plot_categorical(data, col):
    plt.figure(figsize=(8, 4))
    sns.countplot(x=col, data=data)
    plt.title(f'Countplot of {col}')
    plt.show()
