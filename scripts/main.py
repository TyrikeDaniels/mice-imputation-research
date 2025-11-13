from dataset import preprocess
from configs import MEDIAN_PATH, MICE_PATH
from model import rf_kfold_eval

# Change this for testing
FRAC = 0.10         # Fraction of imputed data
TARGET = "ST_50"    # Label 
DROP__DEPTHS = True # Dropping other labels in the dataframe

def main():

    # Load and preprocess data
    preprocess(frac=FRAC, random_state=43, target=TARGET, drop_depths=DROP__DEPTHS)
    
    # NOTE: uncomment to print configurations
    print(
        f"Target label: {TARGET}\n" +
        f"Fraction of values replaced with NaN: {FRAC:.2f}\n" + 
        f"Other depth columns dropped: {'Yes' if DROP__DEPTHS else 'No'}\n")
    
    # Evaluate MICE-imputed data
    print("Attempting to model MICE immputed data...")
    mice_train, mice_val = rf_kfold_eval(path=MICE_PATH, target=TARGET)
    print(f"MICE Imputation - Mean Train R^2: {mice_train:.4f}, Mean Validation R^2: {mice_val:.4f}\n")

    # Evaluate Median-imputed data
    print("Attempting to model media immputed data...")
    median_train, median_val = rf_kfold_eval(path=MEDIAN_PATH, target=TARGET)
    print(f"Median Imputation - Mean Train R^2: {median_train:.4f}, Mean Validation R^2: {median_val:.4f}")

if __name__ == "__main__":
    main()

