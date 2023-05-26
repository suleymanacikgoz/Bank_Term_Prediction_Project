################################################
# End-to-End Diabetes Machine Learning Pipeline III
################################################

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler



def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    # print(f"Observations: {dataframe.shape[0]}")
    # print(f"Variables: {dataframe.shape[1]}")
    # print(f'cat_cols: {len(cat_cols)}')
    # print(f'num_cols: {len(num_cols)}')
    # print(f'cat_but_car: {len(cat_but_car)}')
    # print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

def bank_data_prep(dataframe):


    dataframe = dataframe.drop(labels=["duration", 'poutcome'], axis=1)

    # Drop the Job Occupations that are "Unknown"
    dataframe = dataframe.drop(dataframe.loc[dataframe["job"] == "unknown"].index)
    dataframe = dataframe.drop(dataframe.loc[dataframe["education"] == "unknown"].index)

    # Manager and admin. are basically the same, added under the same categorical value.
    lst = [dataframe]

    for col in lst:
        col.loc[col["job"] == "admin.", "job"] = "management"

    dataframe["NEW_AGE_CAT"] = pd.cut(dataframe['age'], bins=[0, 35, 55, 70, float('Inf')],
                               labels=['0-35', '35-55', '55-70', '70-100'])

    balance_categories = {
        "debtor": "borçlu",
        'low': 'Düşük',
        'medium': 'Orta',
        'high': 'Yüksek'
    }

    dataframe["NEW_BALANCE_CAT"] = dataframe['balance'].apply(
        lambda x: balance_categories['debtor'] if x < 0 else balance_categories['low'] if x < 1000 else
        balance_categories['high'] if x > 2000 else balance_categories['medium'])

    kış = ["dec", "jan", "feb"]
    ilkbahar = ["mar", "apr", "may"]
    yaz = ["jun", "jul", "aug"]
    sonbahar = ["oct", "nov", "sep"]

    dataframe["NEW_WEATHER_CAT"] = dataframe["month"].apply(
        lambda x: "kış" if x in kış else "ilkbahar" if x in ilkbahar else "yaz" if x in yaz else "sonbahar")


    cat_cols, num_cols, cat_but_car = grab_col_names(dataframe, cat_th=5, car_th=20)

    cat_cols = [col for col in cat_cols if "deposit" not in col]

    df = one_hot_encoder(dataframe, cat_cols, drop_first=True)

    df.columns = [col.upper() for col in df.columns]

    # Son güncel değişken türlerimi tutuyorum.
    cat_cols, num_cols, cat_but_car = grab_col_names(df, cat_th=5, car_th=20)
    cat_cols = [col for col in cat_cols if "DEPOSIT" not in col]

    # replace_with_thresholds(df, "BALANCE")
    # replace_with_thresholds(df, "CAMPAIGN")
    # replace_with_thresholds(df, "PDAYS")
    # replace_with_thresholds(df, "PREVIOUS")

    # Standartlaştırma
    X_scaled = StandardScaler().fit_transform(df[num_cols])
    df[num_cols] = pd.DataFrame(X_scaled, columns=df[num_cols].columns)

    y = df["DEPOSIT"]
    X = df.drop(["DEPOSIT"], axis=1)


    return X, y

df = pd.read_csv("Proje/bank.csv")


X, y = bank_data_prep(df)

random_user = X.sample(1)

new_model = joblib.load("voting_clf1.pkl")

new_model.predict(random_user)



