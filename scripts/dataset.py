import pandas as pd
import numpy as np
import warnings

from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer # enable IterativeImputer
from sklearn.impute import IterativeImputer, SimpleImputer

from configs import MEDIAN_PATH, MICE_PATH, RAW_PATH
from sklearn.exceptions import ConvergenceWarning

def random_mask(df, columns, frac=0.1, random_state=None):
    # set random seed for reproducibility
    np.random.seed(random_state) 

    # work on a copy of the dataframe
    df_copy = df.copy()            
    
    for col in columns["numeric"]:
        # total number of rows
        n_rows = len(df_copy)                       

        # number of values to mask in row
        n_missing = int(frac * n_rows)             

        # generate random row indices
        missing_indices = np.random.choice(n_rows, n_missing, replace=False)

        # set selected entries to NaN
        df_copy.loc[missing_indices, col] = np.nan
    
    # return (copied) dataframe with random NaNs
    return df_copy  

def one_hot(df, nominal):
    """One-hot encode nominal columns."""
    return pd.get_dummies(df, columns=nominal)

def scale(df, numeric):
    """Scale numeric columns using StandardScaler."""
    scaler = StandardScaler()
    df[numeric] = scaler.fit_transform(df[numeric])
    return df

def median_impute(df, columns):
    # Impute numeric columns
    imputer = SimpleImputer(strategy="median")
    df[columns["numeric"]] = imputer.fit_transform(df[columns["numeric"]])
    #print("met median!")

    # Scale numeric columns
    df = scale(df, columns["numeric"])
    
    # One-hot encode nominal columns
    df = one_hot(df, columns["nominal"])
    
    return df

def mice_impute(df, columns):
    # MICE imputer for numeric columns
    imputer = IterativeImputer(random_state=100, max_iter=10)
    df[columns["numeric"]] = imputer.fit_transform(df[columns["numeric"]])
    #print("met mice!")

    # Scale numeric columns
    df = scale(df, columns["numeric"])
    
    # One-hot encode nominal columns
    df = one_hot(df, columns["nominal"])
    
    return df

def get_columns(df):
    """return nominal/numeric columns as dictionary"""
    return {
        "numeric" : df.select_dtypes(include=['number']).columns.tolist(),
        "nominal" : df.select_dtypes(exclude=['number']).columns.tolist()
    }

def preprocess(frac=0.1, random_state=None, target="ST_100", drop_depths=True):

    # surpess convergence warnings
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    # read raw data
    df = pd.read_csv(RAW_PATH)

    # drop other depth columns if target is a soil temperature depth
    if drop_depths:
        depth_cols = [col for col in ["ST_10", "ST_50", "ST_100"] if col != target]
        df.drop(columns=depth_cols, inplace=True, errors="ignore")

    columns = get_columns(df)

    # create random missingness first
    df_masked = random_mask(df, columns, frac, random_state)

    # then impute
    df_mice = mice_impute(df_masked.copy(), columns)
    df_median = median_impute(df_masked.copy(), columns)

    # save to .csv files
    print(f"Attempting to save MICE imputed dataset to .csv file at {MICE_PATH}...")
    df_mice.to_csv(MICE_PATH)
    print("Save successful.\n")

    print(f"Attempting to save median imputed dataset to .csv file at {MEDIAN_PATH}...")
    df_median.to_csv(MEDIAN_PATH)
    print("Save successful.\n")

