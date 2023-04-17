
import pandas as pd

# abalone_train = pd.read_csv(
#     "https://storage.googleapis.com/download.tensorflow.org/data/abalone_train.csv",
#     names=["Length", "Diameter", "Height", "Whole weight", "Shucked weight",
#            "Viscera weight", "Shell weight", "Age"])
#
# print(abalone_train.head())

df_train = pd.read_csv("New issues and IPOs_37.csv",encoding= 'unicode_escape')

print(df_train.head(10))
print(df_train.shape)
print(df_train.isna().sum())

