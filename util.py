# coding: utf-8

import pandas as pd
import numpy as np

"""
    header: Review, Brand, Bariety, Style, Country, Stars, Top, Ten
    shape: 2580*7

"""
def load_data():
    ramen = pd.read_csv("../assets/ramen-ratings.csv")
    
    # 不要な行の削除
    mask = ramen.index[ramen["Stars"] == "Unrated"]
    ramen = ramen.drop(index = mask)
    ramen["Stars"] = ramen["Stars"].astype(float)
    ramen = ramen.drop(columns=['Review #', 'Top Ten', 'Variety'])

    # 特徴量をダミー値に変換
    Country = pd.get_dummies(ramen["Country"], prefix="Country", drop_first=True)
    Brand = pd.get_dummies(ramen["Brand"], prefix="Brand", drop_first=True)
    Style = pd.get_dummies(ramen["Style"], prefix="Style", drop_first=True)
    ramendf = pd.concat([Country, Brand, Style], axis=1)
    
    X = np.array(ramendf, dtype=np.float32)
    Y = np.array(ramen[["Stars"]], dtype=np.float32)

    return X, Y

if __name__ == "__main__":
    x, y = load_data()
    print(y)