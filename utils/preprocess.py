import pandas as pd

def preprocess_data(df):

    X = df.drop(["Food Insecurity Rate","Area","Year"], axis=1)
    y = df["Food Insecurity Rate"]

    X = pd.get_dummies(X, drop_first=True)

    return X, y


def preprocess_input(irrigation, water):

    data = pd.DataFrame({
        "irrigation":[irrigation],
        "water_access":[water]
    })

    return data
