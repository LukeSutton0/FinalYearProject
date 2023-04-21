import os
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_classif, chi2, f_regression
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from scipy.stats import linregress
import numpy as np
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks, optimizers
from kerastuner.tuners import Hyperband
from kerastuner.engine.hyperparameters import HyperParameters

physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)


def get_data_csv():
    main_df = pd.read_csv(os.getcwd() + '\\MainData\\TickerData.csv', encoding='latin1', header=0)
    return main_df


def remove_cols(main_df):
    if 'Transfer' in main_df.columns:
        main_df.drop(['Transfer'], axis=1, inplace=True)
    if 'LSE IPO' in main_df.columns:
        main_df.drop(['LSE IPO'], axis=1, inplace=True)
    if 'notes' in main_df.columns:
        main_df.drop(['notes'], axis=1, inplace=True)
    if 'TIDM' in main_df.columns:
        main_df.drop(['TIDM'], axis=1, inplace=True)
    if 'Company' in main_df.columns:
        main_df.drop(['Company'], axis=1, inplace=True)
    return main_df


def check_for_types(main_df):
    print(main_df.isna().sum(), "\n")


def fix_empty_data(main_df):
    main_df['FTSE Industry'].fillna(0, inplace=True)
    main_df['FTSE Sector'].fillna('Not Identified', inplace=True)
    main_df[' FTSE Subsector '].fillna('Not Identified', inplace=True)  # not sure why there is spaces
    main_df['Currency'].fillna('GBX', inplace=True)
    main_df['Nominated Advisor (AIM only)'].fillna('No Advisor', inplace=True)
    return main_df


def show_rows_with_missing_vals(main_df):
    # values_missing = ["n.a.", "?", "NA", "n/a", "na", "--"]
    missing_val_col = (main_df.isnull().sum())
    print(missing_val_col[missing_val_col > 0])


def remove_empty_rows(main_df):
    # drop row if column TIDM is NAN
    main_df.dropna(subset=['TIDM'], inplace=True)
    main_df.dropna(subset=['Initial Trading Open'], inplace=True)
    return main_df


def row_count(main_df):
    print("Dataframe has", len(main_df.index), "rows")


def remove_anomalous_rows(main_df):
    # print(main_df['Initial Trading Open'].dtype)
    tickers_to_remove = ['SENX', 'MTPH', 'MYSQ', 'BARK', 'CRTM', 'FISH', 'IL0A', 'PRSM']
    for ticker in tickers_to_remove:
        main_df.drop(index=main_df[main_df['TIDM'] == ticker].index, inplace=True)
    # Yahoo picking up old data for old company tickers/Delisted-
    # SENX,#MTPH#MYSQ#BARK#Fish
    # Random jumps in data
    # CRTM#IL0A#PRSM
    return main_df


def clean_columns(main_df):
    main_df = clean_date(main_df)
    main_df = clean_month(main_df)
    main_df = clean_issue_price(main_df)
    main_df = clean_market_cap(main_df)
    main_df = clean_money_raised_existing(main_df)
    main_df = clean_total_raised(main_df)
    main_df = clean_country(main_df)
    return main_df


def clean_date(main_df):
    main_df['Date'] = pd.to_datetime(main_df['Date'])
    main_df['Date'] = main_df['Date'].dt.day
    main_df.rename(columns={'Date': 'Day'}, inplace=True)
    # main_df = main_df.drop('Day',axis=1)
    return main_df


def clean_month(main_df):
    main_df['Month'] = pd.to_datetime(main_df['Month'])
    main_df['Month'] = main_df['Month'].dt.month
    return main_df


def clean_issue_price(main_df):
    temp_df = main_df.dropna(subset=['Issue Price'])
    slope, intercept, rvalue, pvalue, stderr = linregress(temp_df['Issue Price'], temp_df['Adj Close Day 1'])
    # x = temp_df['Issue Price']
    # y = slope * x + intercept
    main_df['Issue Price'] = main_df.apply(
        lambda row: row['Adj Close Day 1'] / slope - intercept if np.isnan(row['Issue Price']) else row['Issue Price'],
        axis=1)
    return main_df


def clean_market_cap(main_df):
    main_df.loc[(main_df[' Market Cap - Opening Price (Â£m) '] == '-') | (
            main_df[' Market Cap - Opening Price (Â£m) '] == " -   "), ' Market Cap - Opening Price (Â£m) '] = 0
    return main_df


def clean_money_raised_existing(main_df):
    main_df.loc[(main_df['Money Raised - Existing ()'] == '-') | (
            main_df['Money Raised - Existing ()'] == ' -   '), 'Money Raised - Existing ()'] = 0
    return main_df


def clean_total_raised(main_df):
    main_df.loc[(main_df['TOTAL RAISED ()'] == '-') | (main_df['TOTAL RAISED ()'] == ' -   '), 'TOTAL RAISED ()'] = 23
    return main_df


def clean_country(main_df):
    for index, row in main_df.iterrows():
        if row['Country of Inc.'] != 'United Kingdom':
            main_df.at[index, 'Country of Inc.'] = 'Not United Kingdom'
    return main_df


def ftse_combine(main_df):
    pd.set_option('display.max_columns', None)
    cat_df = main_df.copy()
    if 'FTSE Sector' in cat_df.columns:
        cat_df.drop(['FTSE Sector'], axis=1, inplace=True)
    if ' FTSE Subsector ' in cat_df.columns:
        cat_df.drop([' FTSE Subsector '], axis=1, inplace=True)
    cat_df['FTSE Industry'] = cat_df['FTSE Industry'].astype(str)
    cat_df.loc[cat_df['FTSE Industry'] == 0, 'FTSE Industry'] = "No Industry Data"
    # new system
    cat_df.loc[(cat_df['FTSE Industry'].astype(str).str[:2] == '10') & (
            cat_df['FTSE Industry'].astype(str).str.len() > 6), 'FTSE Industry'] = "Technology"
    cat_df.loc[(cat_df['FTSE Industry'].astype(str).str[:2] == '15') & (
            cat_df['FTSE Industry'].astype(str).str.len() > 6), 'FTSE Industry'] = "Telecommunications"
    cat_df.loc[(cat_df['FTSE Industry'].astype(str).str[:2] == '20') & (
            cat_df['FTSE Industry'].astype(str).str.len() > 6), 'FTSE Industry'] = "Health Care"
    cat_df.loc[(cat_df['FTSE Industry'].astype(str).str[:2] == '30') & (
            cat_df['FTSE Industry'].astype(str).str.len() > 6), 'FTSE Industry'] = "Financials"
    cat_df.loc[(cat_df['FTSE Industry'].astype(str).str[:2] == '35') & (
            cat_df['FTSE Industry'].astype(str).str.len() > 6), 'FTSE Industry'] = "Real Estate"
    cat_df.loc[(cat_df['FTSE Industry'].astype(str).str[:2] == '40') & (
            cat_df['FTSE Industry'].astype(str).str.len() > 6), 'FTSE Industry'] = "Consumer Discretionary"
    cat_df.loc[(cat_df['FTSE Industry'].astype(str).str[:2] == '45') & (
            cat_df['FTSE Industry'].astype(str).str.len() > 6), 'FTSE Industry'] = "Consumer Staples"
    cat_df.loc[(cat_df['FTSE Industry'].astype(str).str[:2] == '50') & (
            cat_df['FTSE Industry'].astype(str).str.len() > 6), 'FTSE Industry'] = "Industrials"
    cat_df.loc[(cat_df['FTSE Industry'].astype(str).str[:2] == '55') & (
            cat_df['FTSE Industry'].astype(str).str.len() > 6), 'FTSE Industry'] = "Basic Materials"
    cat_df.loc[(cat_df['FTSE Industry'].astype(str).str[:2] == '60') & (
            cat_df['FTSE Industry'].astype(str).str.len() > 6), 'FTSE Industry'] = "Energy"
    cat_df.loc[(cat_df['FTSE Industry'].astype(str).str[:2] == '65') & (
            cat_df['FTSE Industry'].astype(str).str.len() > 6), 'FTSE Industry'] = "Utilities"
    # for old system
    cat_df.loc[(cat_df['FTSE Industry'].astype(str).str[:1] == '9') & (
            cat_df['FTSE Industry'].astype(str).str.len() < 7), 'FTSE Industry'] = "Technology"
    cat_df.loc[(cat_df['FTSE Industry'].astype(str).str[:1] == '6') & (
            cat_df['FTSE Industry'].astype(str).str.len() < 7), 'FTSE Industry'] = "Telecommunications"
    cat_df.loc[(cat_df['FTSE Industry'].astype(str).str[:1] == '4') & (
            cat_df['FTSE Industry'].astype(str).str.len() < 7), 'FTSE Industry'] = "Health Care"
    cat_df.loc[(cat_df['FTSE Industry'].astype(str).str[:1] == '8') & (
            cat_df['FTSE Industry'].astype(str).str.len() < 7), 'FTSE Industry'] = "Financials"
    cat_df.loc[(cat_df['FTSE Industry'].astype(str).str[:1] == '3') & (cat_df['FTSE Industry'].astype(
        str).str.len() < 7), 'FTSE Industry'] = "Consumer Discretionary"  # maybe look at this later
    cat_df.loc[(cat_df['FTSE Industry'].astype(str).str[:1] == '5') & (
            cat_df['FTSE Industry'].astype(str).str.len() < 7), 'FTSE Industry'] = "Consumer Discretionary"
    cat_df.loc[(cat_df['FTSE Industry'].astype(str).str[:1] == '2') & (
            cat_df['FTSE Industry'].astype(str).str.len() < 7), 'FTSE Industry'] = "Industrials"
    cat_df.loc[(cat_df['FTSE Industry'].astype(str).str[:1] == '1') & (
            cat_df['FTSE Industry'].astype(str).str.len() < 7), 'FTSE Industry'] = "Basic Materials"
    cat_df.loc[
        (cat_df['FTSE Industry'].astype(str).str[:1] == '0') & (cat_df['FTSE Industry'].astype(str).str.len() < 7) & (
                cat_df['FTSE Industry'].astype(
                    str).str.len() > 1), 'FTSE Industry'] = "Energy"  # check this isn't interferring
    cat_df.loc[(cat_df['FTSE Industry'].astype(str).str[:1] == '7') & (
            cat_df['FTSE Industry'].astype(str).str.len() < 7), 'FTSE Industry'] = "Utilities"
    return cat_df


def encoding_catagorical(cat_df):
    columns_to_catagorise = ['Market', 'Issue type', 'Country of Inc.', 'FTSE Industry', 'Currency']
    for col in columns_to_catagorise:
        # cat_df['Market'] = cat_df['Market'].astype('category') #change to category
        encoder = OneHotEncoder()
        enc_df = pd.DataFrame(encoder.fit_transform(cat_df[[col]]).toarray())
        col_names = encoder.get_feature_names_out([col])
        enc_df.columns = col_names
        cat_df.reset_index(drop=True, inplace=True)  # reset index to ensure alignment
        cat_df = pd.concat([cat_df, enc_df], axis=1)
        cat_df = cat_df.drop(col, axis=1)
    cat_df.loc[cat_df['Nominated Advisor (AIM only)'] != "No Advisor", 'Nominated Advisor (AIM only)'] = "Advisor"
    binarycol_to_catagorise = ['Nominated Advisor (AIM only)']
    for col in binarycol_to_catagorise:
        encoder = OneHotEncoder(drop='first')
        enc_df = pd.DataFrame(encoder.fit_transform(cat_df[[col]]).toarray())
        col_names = encoder.get_feature_names_out([col])
        enc_df.columns = col_names
        cat_df.reset_index(drop=True, inplace=True)  # reset index to ensure alignment
        cat_df = pd.concat([cat_df, enc_df], axis=1)
        cat_df = cat_df.drop(col, axis=1)
    return cat_df


def data_preparation(main_df):
    # check_for_types(main_df) #all rows shown #row_count(main_df) #how many rows in df #show_rows_with_missing_vals(main_df)
    main_df = remove_empty_rows(main_df)
    main_df = remove_anomalous_rows(main_df)
    main_df = remove_cols(main_df)
    main_df = fix_empty_data(main_df)
    main_df = clean_columns(main_df)
    for col in main_df:
        if "Adj Close Day" not in col:
            main_df.dropna(subset=[col], inplace=True)
    cat_df = ftse_combine(main_df)
    cat_df = encoding_catagorical(cat_df)
    return main_df, cat_df


def issue_type(main_df):
    mode = main_df['Issue type'].mode()
    main_df['Issue type'].value_counts().plot.bar()
    plt.title("Types of issues placed onto the LSE")
    plt.ylabel("Quantity")
    plt.show()
    print(mode)


def country_of_inc(main_df):
    mode = main_df['Country of Inc.'].mode()
    main_df['Country of Inc.'].value_counts().plot.bar()
    plt.show()
    print(mode)
    counts = main_df.groupby(main_df['Country of Inc.'].eq('United Kingdom')).size()
    counts.plot.bar()
    plt.xticks([1, 0], ['UK', 'Non-Uk'], rotation=0)
    plt.xlabel('Country of Inc.')
    plt.ylabel('Count')
    plt.show()


def ftse_industry(main_df):
    main_df['FTSE Industry'].value_counts().plot.bar()
    plt.title("Bar chart showing quantity of IPOs per Industry sector")
    plt.ylabel("Quantity")
    plt.show()

    # subset_df = main_df[['FTSE Industry', 'FTSE Sector', ' FTSE Subsector ', 'Adj Close Day 1']]
    # print(subset_df)
    # # Compute the correlation matrix
    #
    # encoder = OneHotEncoder()
    # enc_df = pd.DataFrame(encoder.fit_transform(main_df[['FTSE Sector']]).toarray())
    # col_names = encoder.get_feature_names_out(['FTSE Sector'])
    #
    # enc_df.columns = col_names
    # subset_df.reset_index(drop=True, inplace=True)  # reset index to ensure alignment
    # subset_df = pd.concat([subset_df, enc_df], axis=1)
    # subset_df = subset_df.drop('FTSE Sector', axis=1)
    # corr_matrix = subset_df.corr()
    # # Plot the heatmap
    # sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    # #create heatmap that shows correlation between FTSE industry, Ftse sector , Ftse subsector and target variable
    # plt.show()


def ftse_sector(main_df):
    print(main_df)
    main_df['FTSE Sector'].value_counts().plot.bar()
    plt.title('IPO quantity per sector')
    plt.xlabel('Quantity')
    plt.ylabel('FTSE Sector')
    plt.show()

    sns.stripplot(x="Year", y="FTSE Sector", data=main_df, jitter=False)
    plt.xlabel('Year')
    plt.ylabel('FTSE Sector')
    plt.show()

    sector_counts = main_df.groupby(['Year', 'FTSE Sector']).size().reset_index(name='counts')
    # Initialize empty list for plotted labels
    plotted_labels = []
    for sector in sector_counts['FTSE Sector'].unique():
        data = sector_counts[sector_counts['FTSE Sector'] == sector]
        if len(data) >= 3:  # exclude values with less than 5
            plt.plot(data['Year'], data['counts'], label=sector)
            plotted_labels.append(sector)  # add label to plotted labels list

    # Set legend to only include plotted labels
    plt.legend(loc='upper left', labels=plotted_labels, bbox_to_anchor=(1.05, 1))
    plt.title('IPOs released per sector over the last 20 years')

    plt.xlabel('Year')
    plt.xticks(rotation=0)
    plt.gca().xaxis.set_major_formatter('{:.0f}'.format)
    plt.ylabel('Quantity')
    plt.show()


def ftse_subsector(main_df):
    main_df[' FTSE Subsector '].value_counts().plot.bar()
    plt.title("Bar chart showing quantity of IPOs per Industry Subsector")
    plt.ylabel("Quantity")
    plt.show()


def issue_price(main_df):
    temp_df = main_df.dropna(subset=['Issue Price'])
    plt.scatter(temp_df['Issue Price'], temp_df['Adj Close Day 1'])
    plt.xlabel('Issue Price')
    plt.ylabel('Adj Close Day 1')
    slope, intercept, rvalue, pvalue, stderr = linregress(temp_df['Issue Price'], temp_df['Adj Close Day 1'])
    print('Slope:', slope)
    print('Intercept:', intercept)
    print('R-squared:', rvalue ** 2)
    x = temp_df['Issue Price']
    y = slope * x + intercept
    plt.plot(x, y, color='r')
    plt.show()
    main_df['Issue Price'] = main_df.apply(
        lambda row: row['Adj Close Day 1'] / slope - intercept if np.isnan(row['Issue Price']) else row['Issue Price'],
        axis=1)
    plt.boxplot(main_df['Issue Price'], showmeans=True)
    plt.xlabel("")
    plt.ylabel("Price (£)")
    plt.title("Issue price before Day 1 of trading")
    plt.show()


def money_raised(main_df):
    temp_df = main_df.drop(main_df[main_df['Money Raised - Existing ()'] == '-'].index)
    temp_df['Money Raised - Existing ()'] = temp_df['Money Raised - Existing ()'].astype(float)
    mean = temp_df['Money Raised - Existing ()'].mean()
    median = temp_df['Money Raised - Existing ()'].median()
    print("Money Raised - Existing ()", mean)
    print("Money Raised - Existing ()", median)
    plt.hist(temp_df['Money Raised - Existing ()'], bins=70)
    plt.axvline(x=mean, color='r', linestyle='--')  # Add a vertical line for the mean
    plt.axvline(x=median, color='#FFA500', linestyle='--')  # Add a vertical line for the mean
    plt.xlabel('Money Raised - Existing ()')
    plt.ylabel('Frequency')
    plt.title('Histogram of Money Raised - Existing ()')
    plt.show()

    temp_df['Money Raised - Existing ()'].plot(kind='density')
    plt.xlabel('Money Raised - Existing ()')
    plt.ylabel('Density')
    plt.grid(True)
    plt.show()


def total_raised(main_df):
    temp_df = main_df.drop(main_df[main_df['TOTAL RAISED ()'] == '-'].index)
    temp_df['TOTAL RAISED ()'] = temp_df['TOTAL RAISED ()'].astype(float)
    mean = temp_df['TOTAL RAISED ()'].mean()
    median = temp_df['TOTAL RAISED ()'].median()
    print("Mean of TOTAL RAISED ():", mean)
    print("Median of TOTAL RAISED ():", median)
    plt.hist(temp_df['TOTAL RAISED ()'], bins=70)
    plt.axvline(x=mean, color='r', linestyle='--')  # Add a vertical line for the mean
    plt.axvline(x=median, color='#FFA500', linestyle='--')  # Add a vertical line for the mean
    plt.xlabel('TOTAL RAISED £')
    plt.ylabel('Number of IPOs')
    plt.title('Histogram of total capital gained before sale')
    plt.show()

    temp_df['TOTAL RAISED ()'].plot(kind='density')
    plt.xlabel('TOTAL RAISED ()')
    plt.ylabel('Density')
    plt.grid(True)
    plt.show()


def explore_remove_cols(main_df):
    column_to_explore = ['LSE IPO', 'TIDM', 'Company']
    target_variable = main_df['Adj Close Day 1']
    sns.histplot(target_variable, kde=True)
    plt.title('Distribution of Adj Close Day 1')
    plt.show()
    print(main_df)
    for col in column_to_explore:
        sns.countplot(x=col, data=main_df)
        plt.title('Distribution of ' + col)
        plt.xticks(rotation=90)
        plt.show()
    for col in column_to_explore:
        sns.boxplot(x=col, y=target_variable, data=main_df)
        plt.title('Boxplot of ' + col + ' vs. Adj Close Day 1')
        plt.show()


def day1_top_feat(cat_df):
    # day 1
    temp_df = cat_df.copy()
    drop_cols = list(temp_df.columns[9:21])
    temp_df = temp_df.drop(drop_cols, axis=1)
    for col in temp_df:
        temp_df[col] = temp_df[col].astype(float)
    print(temp_df.shape)
    selector = SelectKBest(score_func=f_regression, k=7)
    top_features = selector.fit_transform(temp_df, cat_df['Adj Close Day 1'])
    filter = selector.get_support()
    features = temp_df.columns[filter]
    feat_df = pd.DataFrame(top_features)
    feat_df.columns = features  # assign column names to feat_df DataFrame
    print("All features:", features)
    corr = feat_df.corr()
    plt.figure()
    plt.title("A heatmap showing the correlation between the variables used for linear regression")
    sns.heatmap(corr, annot=True, xticklabels=corr.columns, yticklabels=corr.columns)
    plt.show()

    selector = SelectKBest(score_func=f_regression, k=7)
    # top_features = selector.fit_transform(temp_df, cat_df['Adj Close Day 1'])
    filter = selector.get_support()
    feat_scores = pd.DataFrame({'Feature': features, 'Fischer Score': selector.scores_[filter]})
    feat_scores = feat_scores.sort_values('Fischer Score', ascending=False)
    # Create a bar plot of the Fischer scores
    plt.bar(feat_scores['Feature'], feat_scores['Fischer Score'])
    plt.xticks(rotation=90)
    plt.xlabel('Feature')
    plt.ylabel('Fischer Score')
    plt.title('Top 7 Fischer Scores')
    plt.show()


def top_features(cat_df):
    day1_top_feat(cat_df)


def adjcloseday1(cat_df):
    plt.hist(cat_df['Adj Close Day 1'], bins=50)
    plt.xlabel('Adj Close Day 1')
    plt.ylabel('Frequency')
    plt.title('Histogram of Target Variable')
    plt.show()


def exploratory_data_analysis(main_df, cat_df):
    # issue_type(main_df)
    # country_of_inc(main_df)
    # ftse_industry(main_df)
    # ftse_sector(main_df)
    # ftse_subsector(main_df)
    # issue_price(main_df)
    # money_raised(main_df)
    # total_raised(main_df)
    # top_features(cat_df)
    # adj_close_day1(cat_df)
    # explore_remove_cols(main_df)
    return main_df


def mlp(predict_df):
    bin_edges = [-float("inf"), 0, 50, 100, 150, float("inf")]
    # Create a new column 'Target_bin' in the dataframe by binning the 'Adj Close Day 1' column
    predict_df['Adj Close Day 1'] = pd.cut(predict_df['Adj Close Day 1'], bins=bin_edges, labels=[0, 1, 2, 3, 4])
    train_df, test_df = train_test_split(predict_df, test_size=0.2, random_state=41)
    x_train = np.array(train_df.drop("Adj Close Day 1", axis=1))
    y_train = np.array(train_df["Adj Close Day 1"])
    x_test = np.array(test_df.drop("Adj Close Day 1", axis=1))
    y_test = np.array(test_df["Adj Close Day 1"])

    # plt.hist(x_train[:, 0], bins=50)
    # plt.title("Histogram of first feature in original data")
    # #plt.show()
    scaler = RobustScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)  # Transform the test data using the scaler
    print("Mean of scaled features: ")
    print(x_train.mean(axis=0))
    print("\nVariance of scaled features: ")
    print(x_train.var(axis=0))
    # Plot histogram of first feature in original data
    # plt.hist(x_train[:, 0], bins=50)
    # plt.title("Histogram of first feature in scaled data")
    # plt.show()
    model = keras.Sequential(
        [
            keras.Input(shape=(x_train.shape[1])),
            layers.Dense(512, activation="relu"),
            layers.Dense(256, activation="relu"),
            layers.Dense(128, activation="relu"),
            layers.Dense(64, activation="relu"),
            layers.Dense(32, activation="relu"),
            layers.Dense(10),
        ]
    )
    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=keras.optimizers.Adam(lr=0.001),
        metrics=["accuracy"],
    )
    model.fit(x_train, y_train, batch_size=100, epochs=100, verbose=2)
    model.evaluate(x_test, y_test, batch_size=100, verbose=2)
    y_pred = model.predict(x_test)
    cm = confusion_matrix(y_test, y_pred.argmax(axis=1))
    print(cm)
    y_pred_classes = np.argmax(y_pred, axis=1)
    print(classification_report(y_test, y_pred_classes, target_names=['<0', '0-50', '50-100', '100-150', '>150'],
                                zero_division=1))


def mlp_day1_get_hyper_param(predict_df):
    bin_edges = [-float("inf"), 0, 50, 100, 150, float("inf")]
    # Create a new column 'Target_bin' in the dataframe by binning the 'Adj Close Day 1' column
    predict_df['Adj Close Day 1'] = pd.cut(predict_df['Adj Close Day 1'], bins=bin_edges, labels=[0, 1, 2, 3, 4])
    # print(predict_df)
    # Print the count of values in each bin
    # print(predict_df['Adj Close Day 1'].value_counts())

    train_df, test_df = train_test_split(predict_df, test_size=0.2, random_state=41)
    x_train = np.array(train_df.drop("Adj Close Day 1", axis=1))
    y_train = np.array(train_df["Adj Close Day 1"])

    x_test = np.array(test_df.drop("Adj Close Day 1", axis=1))
    y_test = np.array(test_df["Adj Close Day 1"])

    hp = HyperParameters()
    hp.Fixed('num_hidden_layers', value=1)
    hp.Int('num_units', min_value=32, max_value=512, step=32)
    hp.Float('learning_rate', min_value=1e-5, max_value=1e-2, sampling='LOG')
    hp.Choice('activation', values=['relu', 'tanh', 'sigmoid'])

    # Create a Hyperband tuner and pass it the function to tune and the hyperparameters to tune
    tuner = Hyperband(
        build_mlp,
        objective='val_accuracy',
        max_epochs=50,
        factor=3,
        hyperparameters=hp,
        directory='tuner_dir',
        project_name='mlp_tuner'
    )

    # Run the tuner to search for the best hyperparameters for the MLP model
    tuner.search(x_train, y_train, validation_split=0.2, callbacks=[callbacks.EarlyStopping(patience=3)])

    # Print the best hyperparameters found by the tuner
    best_hp = tuner.get_best_hyperparameters(1)[0]
    print(f"Best hyperparameters: {best_hp}")
    print(
        f"Best hyperparameters: {best_hp.get('num_hidden_layers')}, {best_hp.get('num_units')},"
        f" {best_hp.get('learning_rate')}, {best_hp.get('activation')}")


def build_mlp(hp):
    # Create the model
    model = keras.Sequential([
        keras.Input(shape=7),
        layers.Dense(7, activation="relu"),
        layers.Dense(6, activation="relu"),
        layers.Dense(2, activation="relu"),
        layers.Dense(1, activation="relu"),
        layers.Dense(7, activation="relu"),
        layers.Dense(7, activation='softmax')
    ])
    # Compile the model
    optimizer = keras.optimizers.Adam(learning_rate=hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4]))
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


def mlp_day1_tuned(predict_df):
    bin_edges = [-float("inf"), 0, 50, 100, 150, float("inf")]
    predict_df['Adj Close Day 1'] = pd.cut(predict_df['Adj Close Day 1'], bins=bin_edges, labels=[0, 1, 2, 3, 4])
    train_df, test_df = train_test_split(predict_df, test_size=0.2, random_state=41)
    x_train = np.array(train_df.drop("Adj Close Day 1", axis=1))
    y_train = np.array(train_df["Adj Close Day 1"])
    x_test = np.array(test_df.drop("Adj Close Day 1", axis=1))
    y_test = np.array(test_df["Adj Close Day 1"])

    scaler = RobustScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    best_hp = {'num_hidden_layers': 1, 'num_units': 384, 'learning_rate': 0.0005147737867565353,
               'activation': 'relu'}  # hyperparams found from hyperparam run

    # Create the model with the best hyperparameters found by the tuner
    model = keras.Sequential()
    model.add(keras.Input(shape=7))
    for i in range(best_hp['num_hidden_layers']):
        model.add(layers.Dense(units=best_hp['num_units'], activation=best_hp['activation']))
    model.add(layers.Dense(5, activation="softmax"))

    # Compile the model
    optimizer = keras.optimizers.Adam(learning_rate=best_hp['learning_rate'])
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    # Fit the model on training data
    model.fit(x_train, y_train, epochs=50, batch_size=32, verbose=2)

    # Evaluate the model on test data
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    print("Test accuracy:", test_acc)


def ran_forest_day1(predict_df):
    bin_edges = [-float("inf"), 0, 50, 100, 150, float("inf")]
    predict_df['Adj Close Day 1'] = pd.cut(predict_df['Adj Close Day 1'], bins=bin_edges, labels=[0, 1, 2, 3, 4])
    # Define target column and feature columns
    target_col = "Adj Close Day 1"
    feature_cols = [col for col in predict_df.columns if col != target_col]
    # Split data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(predict_df[feature_cols], predict_df[target_col], test_size=0.2,
                                                        random_state=41)

    # Create the Random Forest classifier
    rf = RandomForestClassifier(n_estimators=1000, random_state=41)
    # Train the classifier on the training set
    rf.fit(x_train, y_train)
    # Evaluate the classifier on the testing set
    y_pred = rf.predict(x_test)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=['<0', '0-50', '50-100', '100-150', '>150']))


def ran_forest_day1_grid(predict_df):
    bin_edges = [-float("inf"), 0, 50, 100, 150, float("inf")]
    predict_df['Adj Close Day 1'] = pd.cut(predict_df['Adj Close Day 1'], bins=bin_edges, labels=[0, 1, 2, 3, 4])
    # Define target column and feature columns
    target_col = "Adj Close Day 1"
    feature_cols = [col for col in predict_df.columns if col != target_col]
    # Split data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(predict_df[feature_cols], predict_df[target_col], test_size=0.2,
                                                        random_state=41)

    # Create the Random Forest classifier
    rf = RandomForestClassifier(random_state=41, max_depth=10, min_samples_split=5, n_estimators=100)

    param_grid = {
        'n_estimators': [100, 500, 1000],
        'max_depth': [None, 5, 10],
        'min_samples_split': [2, 5, 10]
    }

    # Create a GridSearchCV object and fit it to the training data
    grid_search = GridSearchCV(rf, param_grid, cv=5, n_jobs=-1)
    grid_search.fit(x_train, y_train)

    # Print the best hyperparameters found by GridSearchCV
    print("Best hyperparameters: ", grid_search.best_params_)

    # Evaluate the classifier on the testing set using the best hyperparameters
    y_pred = grid_search.predict(x_test)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=['<0', '0-50', '50-100', '100-150', '>150']))


def ran_forest_day1_tuned(predict_df):
    bin_edges = [-float("inf"), 0, 50, 100, 150, float("inf")]
    predict_df['Adj Close Day 1'] = pd.cut(predict_df['Adj Close Day 1'], bins=bin_edges, labels=[0, 1, 2, 3, 4])
    target_col = "Adj Close Day 1"
    feature_cols = [col for col in predict_df.columns if col != target_col]
    # Split data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(predict_df[feature_cols], predict_df[target_col], test_size=0.2,
                                                        random_state=41)
    # Create the Random Forest classifier with the best hyperparameters
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=5, random_state=41)
    # Train and evaluate the classifier using k-fold cross-validation
    k = 5
    kf = KFold(n_splits=k, shuffle=True, random_state=41)
    for train_index, test_index in kf.split(x_train):
        x_train_k, x_test_k = x_train.iloc[train_index], x_train.iloc[test_index]
        y_train_k, y_test_k = y_train.iloc[train_index], y_train.iloc[test_index]
        rf.fit(x_train_k, y_train_k)
        y_pred_k = rf.predict(x_test_k)

    # Train the classifier on the full training set and evaluate on the testing set
    rf.fit(x_train, y_train)
    y_pred = rf.predict(x_test)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=['<0', '0-50', '50-100', '100-150', '>150']))
    print(accuracy_score(y_test, y_pred))


def log_reg_day1(predict_df):
    bin_edges = [-float("inf"), 0, 50, 100, 150, float("inf")]
    predict_df['Adj Close Day 1'] = pd.cut(predict_df['Adj Close Day 1'], bins=bin_edges, labels=[0, 1, 2, 3, 4])
    target_col = "Adj Close Day 1"
    feature_cols = [col for col in predict_df.columns if col != target_col]
    # Split into training and testing
    x_train, x_test, y_train, y_test = train_test_split(predict_df[feature_cols], predict_df[target_col], test_size=0.2,
                                                        random_state=41)
    lr = LogisticRegression(random_state=42, max_iter=20000)  # Create Log Regression class
    lr.fit(x_train, y_train)
    # Evaluate the classifier on the testing set
    y_pred = lr.predict(x_test)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=['<0', '0-50', '50-100', '100-150', '>150']))


def log_reg_day1_tuned(predict_df):
    bin_edges = [-float("inf"), 0, 50, 100, 150, float("inf")]
    predict_df['Adj Close Day 1'] = pd.cut(predict_df['Adj Close Day 1'], bins=bin_edges, labels=[0, 1, 2, 3, 4])
    # Define target column and feature columns
    target_col = "Adj Close Day 1"
    feature_cols = [col for col in predict_df.columns if col != target_col]
    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(predict_df[feature_cols], predict_df[target_col], test_size=0.2,
                                                        random_state=41)

    k = 5
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    # Create the Logistic Regression classifier
    lr = LogisticRegression(random_state=42, max_iter=20000)

    # Train and evaluate the classifier using k-fold cross-validation
    for train_index, test_index in kf.split(x_train):
        x_train_k, x_test_k = x_train.iloc[train_index], x_train.iloc[test_index]
        y_train_k, y_test_k = y_train.iloc[train_index], y_train.iloc[test_index]

        lr.fit(x_train_k, y_train_k)
        y_pred_k = lr.predict(x_test_k)

    # Train the classifier on the full training set and evaluate on the testing set
    lr.fit(x_train, y_train)
    y_pred = lr.predict(x_test)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=['<0', '0-50', '50-100', '100-150', '>150']))


def predict_day1(cat_df):
    # drop data for later prediction
    target_col = cat_df['Adj Close Day 1'].copy()
    drop_cols = list(cat_df.columns[9:21])
    cat_df = cat_df.drop(drop_cols, axis=1)
    for col in cat_df:
        cat_df[col] = cat_df[col].astype(float)
    # indicators from EDA : Issue Price', ' Market Cap - Opening Price (Â£m) ', 'Money Raised - Existing ()',
    # 'TOTAL RAISED ()', 'Initial Trading Open', 'Market_International Main Market', 'FTSE Industry_Industrials'
    predict_df = cat_df.copy()
    columns_to_keep = ['Issue Price', ' Market Cap - Opening Price (Â£m) ', 'Money Raised - Existing ()',
                     'TOTAL RAISED ()', 'Initial Trading Open', 'Market_International Main Market',
                     'FTSE Industry_Industrials']
    for col in predict_df:
        if col not in columns_to_keep:
            # drop col if not in columns to keep
            predict_df = predict_df.drop(col, axis=1)
    predict_df = pd.concat([predict_df, target_col], axis=1)

    # mlp(predict_df)
    # mlp_day1_get_hyper_param(predict_df):
    # mlp_day1_tuned(predict_df)
    
    # ran_forest_day1(predict_df)
    # ran_forest_day1_grid(predict_df)
    # ran_forest_day1_tuned(predict_df)
    
    # log_reg_day1(predict_df)
    # log_reg_day1_tuned(predict_df)


def main():
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 280)
    main_df = get_data_csv()
    main_df, cat_df = data_preparation(main_df)
    main_df = exploratory_data_analysis(main_df, cat_df)
    predict_day1(cat_df)


main()

# old
