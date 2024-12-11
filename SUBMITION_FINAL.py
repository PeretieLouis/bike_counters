import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

# Custom transformer for encoding time features
class TimeFeatureEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, date_col='date'):
        self.date_col = date_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X['year'] = X[self.date_col].dt.year
        X['quarter'] = X[self.date_col].dt.quarter
        X['month'] = X[self.date_col].dt.month
        X['day'] = X[self.date_col].dt.day
        X['weekday'] = X[self.date_col].dt.weekday
        X['hour'] = X[self.date_col].dt.hour

        # Cyclical encodings
        X['month_cos'] = np.cos(2 * np.pi * X['month'] / 12)
        X['month_sin'] = np.sin(2 * np.pi * X['month'] / 12)
        X['hour_cos'] = np.cos(2 * np.pi * X['hour'] / 24)
        X['hour_sin'] = np.sin(2 * np.pi * X['hour'] / 24)

        # Weekend indicator
        X['is_weekend'] = (X['weekday'] >= 5).astype(int)
        return X.drop(columns=[self.date_col])


# Function to calculate distance to the center
def calculate_distance(X):
    X = X.copy()
    X['dist_center'] = np.sqrt(
        (X['latitude'] - 48.8566) ** 2 + (X['longitude'] - 2.3522) ** 2
    )
    return X


# Preprocessing pipeline
def create_pipeline(numerical_cols, categorical_cols):
    # Preprocessing steps
    preprocess = ColumnTransformer(
        transformers=[
            ('time_features', TimeFeatureEncoder(date_col='date'), ['date']),
            ('distance', FunctionTransformer(calculate_distance), 
             ['latitude', 'longitude']),
            ('numerical_scaler', StandardScaler(), numerical_cols),
            ('counter_id_enc', OneHotEncoder(handle_unknown='ignore', 
                                             sparse_output=False), 
                                             ['counter_id']),
            ('site_id_enc', OneHotEncoder(handle_unknown='ignore', 
                                          sparse_output=False), ['site_id']),
        ],
        remainder='drop'
    )

    # Final pipeline with preprocessing and model
    pipeline = Pipeline([
        ('preprocess', preprocess),
        ('model', ExtraTreesRegressor(n_estimators=100, random_state=42, 
                                      verbose=3))
    ])

    return pipeline


# Prepare and merge data
def prepare_and_merge_data(train_df, weather_df, school_hols_df):
    train_df['date'] = pd.to_datetime(train_df['date'])
    weather_df['date'] = pd.to_datetime(weather_df['date'])
    school_hols_df['date'] = pd.to_datetime(school_hols_df['date'])

    school_hols_df['vacances_zone_c'] = school_hols_df['vacances_zone_c'].astype(int)

    # Add bank holidays
    import holidays
    fr_holidays = holidays.France()
    train_df['is_bank_holiday'] = train_df['date'].dt.date.apply(
        lambda d: 1 if d in fr_holidays else 0)

    # Add lockdown periods
    lockdown_periods = [
        ('2020-03-18', '2020-05-10'),
        ('2020-10-31', '2020-12-14'),
        ('2021-04-04', '2021-05-02'),
    ]

    def in_lockdown(dt):
        d_str = dt.strftime('%Y-%m-%d')
        return 1 if any(start <= d_str <= end for start, end in 
                        lockdown_periods) else 0
    train_df['is_lockdown'] = train_df['date'].apply(in_lockdown)

    # Merge school holidays
    train_df['date_only'] = train_df['date'].dt.floor('D')
    train_df = train_df.merge(
        school_hols_df[['date', 'vacances_zone_c']],
        left_on='date_only',
        right_on='date',
        how='left'
    )
    train_df.rename(columns={'vacances_zone_c': 'school_holidays'}, 
                    inplace=True)
    train_df.drop(columns=['date_only', 'date_y'], inplace=True)
    train_df.rename(columns={'date_x': 'date'}, inplace=True)

    # Merge weather data
    train_df = pd.merge(train_df, weather_df, on='date', how='left')
    return train_df

# Load datasets
#train_df = pd.read_parquet("../input/mdsb-2023/train.parquet")
#test_df = pd.read_parquet("../input/mdsb-2023/final_test.parquet")
#school_hols_df = pd.read_csv("../input/mdsb-2023/holidays.csv")
train_df = pd.read_parquet("data/train.parquet")
test_df = pd.read_parquet("data/final_test.parquet")
school_hols_df = pd.read_csv("external_data/holidays.csv")

# Load and preprocess weather data
weather_df = pd.read_csv(
    #"../input/mdsb-2023/H_75_previous-2020-2022.csv.gz",
    "external_data/H_75_previous-2020-2022.csv.gz",
    parse_dates=["AAAAMMJJHH"],
    date_format="%Y%m%d%H",
    compression="gzip",
    sep=";").rename(columns={"AAAAMMJJHH": "date"})

weather_df = weather_df[['NUM_POSTE', 'date', 'RR1', 'DRR1', 'FF', 
                         'T', 'TCHAUSSEE', 'U', 'GLO']]
weather_df = weather_df[weather_df['NUM_POSTE'] == 75114001]
weather_df.drop('NUM_POSTE', axis=1, inplace=True)
weather_df.set_index('date', inplace=True)
weather_df = weather_df.interpolate(method='time')
weather_df.reset_index(inplace=True)

# Prepare and merge data
final_train_df = prepare_and_merge_data(train_df, weather_df, school_hols_df)
final_test_df = prepare_and_merge_data(test_df, weather_df, school_hols_df)

# Define target, numerical columns, and categorical columns
y_train = final_train_df['log_bike_count']
X_train = final_train_df.drop(columns=['log_bike_count'])
X_test = final_test_df

numerical_cols = ['RR1', 'DRR1', 'FF', 'T', 'TCHAUSSEE', 'U', 'GLO', 
                  'is_bank_holiday', 'is_lockdown', 'school_holidays']
categorical_cols = ['counter_id', 'site_id']

# Create pipeline
pipeline = create_pipeline(numerical_cols, categorical_cols)

# Train pipeline
pipeline.fit(X_train, y_train)

# Make predictions
y_test_pred = pipeline.predict(X_test)

# Save results
results = pd.DataFrame({
    "Id": range(len(y_test_pred)),
    "log_bike_count": y_test_pred,
})
results.to_csv("submission_2.csv", index=False)