import pandas as pd

def clean_nans(x):
    
    cols_to_keeps = []
    for col in x.columns:
        if pd.isna(x[col]).sum()/len(x) > 1/3:
            continue
        cols_to_keeps.append(col)

    df = x[cols_to_keeps].copy()
    df.dropna(axis=0, inplace=True)

    return df


def convert_categorical_to_binary(df, variables):

    for var in variables:

        encoded = pd.get_dummies(df[var], drop_first=True)
        encoded.columns = [var + "_" + str(col) for col in encoded.columns]
        # Drop the original column
        df.drop(var, axis=1, inplace=True)
        df = pd.concat([df, encoded], axis=1)

    return df