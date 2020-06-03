import argparse
import numpy as np
import pandas as pd
import pickle
import joblib
import warnings

warnings.filterwarnings('ignore')


parser = argparse.ArgumentParser(description="Input your file name.")
parser.add_argument('file_name')

args = parser.parse_args()
file_name = args.file_name

df = pd.read_csv(file_name)

def y_check(df):
    return "class" in df.columns

def na_check(df):
    return bool(sum(df.isnull().sum()))

if na_check(df):

    total_assets = pd.Series(np.exp(df['Attr29']), index=df['Attr29'].index)
    assert not bool(total_assets.isnull().sum())

    total_equity = df['Attr10'] * total_assets
    assert not bool(total_equity.isnull().sum())

    total_liabilities = total_assets - total_equity
    assert not bool(total_liabilities.isnull().sum())

    short_term_liabilities = df['Attr51'] * total_assets
    assert not bool(short_term_liabilities.isnull().sum())

    long_term_liabilities = total_liabilities - short_term_liabilities
    assert not bool(long_term_liabilities.isnull().sum())

    total_sales = pd.Series(np.array(total_assets) * np.array(df['Attr36']))
    assert not bool(total_sales.isnull().sum())

    sales = total_assets * df['Attr9'].values
    assert not bool(sales.isnull().sum())

    gross_profit = total_assets * df['Attr18']
    assert not bool(gross_profit.isnull().sum())

    EBIT = total_assets * df['Attr7']
    assert not bool(EBIT.isnull().sum())

    EBITDA = total_assets * df['Attr48']
    assert not bool(EBITDA.isnull().sum())

    retained_earnings = total_assets * df['Attr6']
    assert not bool(retained_earnings.isnull().sum())

    net_profit = total_assets * df['Attr1']
    assert not bool(net_profit.isnull().sum())

    working_capital = total_assets * df['Attr3']
    profit_on_sales = total_assets * df['Attr35']
    assert not bool(working_capital.isnull().sum())
    assert not bool(profit_on_sales.isnull().sum())

    current_assets = df['Attr50'] * total_liabilities
    current_assets.fillna(30.04, inplace=True)
    assert not bool(current_assets.isnull().sum())

    min_short_term_liabilities = [x for x in np.sort(np.abs(short_term_liabilities)) if x][0]
    short_term_liabilities.replace(0, min_short_term_liabilities, inplace=True)

    min_sales = [x for x in np.sort(np.abs(sales)) if x][0]
    sales.replace(0, min_sales, inplace=True)

    min_total_liabilities = [x for x in np.sort(np.abs(total_liabilities)) if x][0]
    total_liabilities.replace(0, min_total_liabilities, inplace=True)

    min_total_equity = [x for x in np.sort(np.abs(total_equity)) if x][0]
    total_equity.replace(0, min_total_equity, inplace=True)

    df['Attr12'] = gross_profit / short_term_liabilities
    df['Attr23'] = net_profit / sales
    df['Attr39'] = profit_on_sales / sales
    df['Attr49'] = EBITDA / sales
    df['Attr50'] = current_assets / total_liabilities
    df['Attr59'] = long_term_liabilities / total_equity
    df['Attr63'] = sales / short_term_liabilities

    drop_list = []
    for i in (15, 17, 19, 20, 28, 31, 34, 37, 41, 42, 44, 46, 55, 57, 62):
        drop_list.append("Attr" + str(i))

    df.drop(drop_list, axis=1, inplace=True)

    imputer = joblib.load("median_imputer.pkl")

    if y_check(df):
        X, y = df.iloc[:, :-1], df.iloc[:, -1]
        df_imputed = pd.DataFrame(imputer.transform(X))
        df_imputed = pd.concat([df_imputed, y], axis=1)
        df_imputed.columns = df.columns
    else:
        X = df
        df_imputed = imputer.transform(X)
        df_imputed.columns = df.columns


    drop_cols = {'Attr2', 'Attr3', 'Attr25', 'Attr22', 'Attr16', 'Attr30', 'Attr6', 'Attr26', 'Attr32', 'Attr39', 'Attr54', 'Attr7',
     'Attr13', 'Attr11', 'Attr8', 'Attr1', 'Attr53', 'Attr14', 'Attr10'}
    df_imputed.drop(drop_cols, axis=1, inplace=True)

    df_imputed.to_csv(file_name.rstrip(".csv") + "_" + "imputed.csv", index=False)


