import pandas as pd

def keepcolumns(input_file, columns):
    csv = pd.read_csv(input_file)
    df = pd.DataFrame(csv)
    for col in df:
        if(col not in columns):
            df = df.drop(col, axis=1)
    return df

def keepcolumnsdf(df, columns):
    for col in df:
        if(col not in columns):
            df = df.drop(col, axis=1)
    return df
fixed = keepcolumns('holdingsbad.csv', ['Holding Ticker', 'Weight'])
fixed.to_csv('holdings.csv', index=False)
        