import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

# Load datasets
train_df = pd.read_parquet("data/train.parquet")
test_df = pd.read_parquet("data/final_test.parquet")
school_hols_df = pd.read_csv("external_data/holidays.csv")

# Load and preprocess weather data
weather_df = pd.read_csv(
    "external_data/H_75_previous-2020-2022.csv.gz",
    parse_dates=["AAAAMMJJHH"],
    date_format="%Y%m%d%H",
    compression="gzip",
    sep=";",
).rename(columns={"AAAAMMJJHH": "date"})

weather_df = weather_df[['NUM_POSTE', 'date', 'RR1',
                         'DRR1', 'FF', 'T', 'TCHAUSSEE', 'U', 'GLO']]
weather_df = weather_df[weather_df['NUM_POSTE'] == 75114001]
weather_df.drop('NUM_POSTE', axis=1, inplace=True)
weather_df.set_index('date', inplace=True)
weather_df = weather_df.interpolate(method='time')
weather_df.reset_index(inplace=True)

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
        return 1 if any(start <= d_str <= end for start, end in lockdown_periods) else 0
    train_df['is_lockdown'] = train_df['date'].apply(in_lockdown)

    # Merge school holidays
    train_df['date_only'] = train_df['date'].dt.floor('D')
    train_df = train_df.merge(
        school_hols_df[['date', 'vacances_zone_c']],
        left_on='date_only',
        right_on='date',
        how='left'
    )
    train_df.rename(columns={'vacances_zone_c': 'school_holidays'}, inplace=True)
    train_df.drop(columns=['date_only', 'date_y'], inplace=True)
    train_df.rename(columns={'date_x': 'date'}, inplace=True)

    # Merge weather data
    train_df = pd.merge(train_df, weather_df, on='date', how='left')
    return train_df


final_train_df = prepare_and_merge_data(train_df, weather_df, school_hols_df)
final_test_df = prepare_and_merge_data(test_df, weather_df, school_hols_df)

# Function to encode date features


def encode_time_features(df, date_col='date'):
    df['year'] = df[date_col].dt.year
    df['quarter'] = df[date_col].dt.quarter
    df['month'] = df[date_col].dt.month
    df['day'] = df[date_col].dt.day
    df['weekday'] = df[date_col].dt.weekday
    df['hour'] = df[date_col].dt.hour

    # Cyclical encodings
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)

    # Weekend indicator
    df['is_weekend'] = (df['weekday'] >= 5).astype(int)

    return df.drop(columns=[date_col])


final_train_df = encode_time_features(final_train_df, date_col='date')
final_test_df = encode_time_features(final_test_df, date_col='date')

# Calculate distance feature
final_train_df['dist_center'] = np.sqrt(
    (final_train_df['latitude'] - 48.8566)**2 +
    (final_train_df['longitude'] - 2.3522)**2
)
final_test_df['dist_center'] = np.sqrt(
    (final_test_df['latitude'] - 48.8566)**2 +
    (final_test_df['longitude'] - 2.3522)**2
)

cols_to_drop = [
    'counter_name', 'counter_technical_id', 'site_name', 'latitude', 'longitude',
    'coordinates', 'bike_count', 'counter_installation_date'
]
# Drop columns with errors='ignore' to prevent KeyError if columns are missing
final_train_df.drop(columns=cols_to_drop, errors='ignore', inplace=True)
final_test_df.drop(columns=cols_to_drop, errors='ignore', inplace=True)

# Encode counter_id and site_id separately
encoder_counter = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
encoder_site = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

# Fit and transform on counter_id
counter_id_encoded = encoder_counter.fit_transform(final_train_df[['counter_id']])
counter_id_encoded_df = pd.DataFrame(
    counter_id_encoded, columns=encoder_counter.get_feature_names_out(['counter_id']))
final_train_df = pd.concat([final_train_df, counter_id_encoded_df], axis=1)

# Fit and transform on site_id
site_id_encoded = encoder_site.fit_transform(final_train_df[['site_id']])
site_id_encoded_df = pd.DataFrame(
    site_id_encoded, columns=encoder_site.get_feature_names_out(['site_id']))
final_train_df = pd.concat([final_train_df, site_id_encoded_df], axis=1)

final_train_df.drop(columns=['counter_id', 'site_id'], inplace=True)

# Define target and features
y_train = final_train_df['log_bike_count']
X_train = final_train_df.drop(columns=['log_bike_count'])

# Transform test dataset
counter_id_encoded_test = encoder_counter.transform(final_test_df[['counter_id']])
counter_id_encoded_df_test = pd.DataFrame(
    counter_id_encoded_test, columns=encoder_counter.get_feature_names_out(['counter_id']))
final_test_df = pd.concat([final_test_df, counter_id_encoded_df_test], axis=1)

site_id_encoded_test = encoder_site.transform(final_test_df[['site_id']])
site_id_encoded_df_test = pd.DataFrame(
    site_id_encoded_test, columns=encoder_site.get_feature_names_out(['site_id']))
final_test_df = pd.concat([final_test_df, site_id_encoded_df_test], axis=1)

final_test_df.drop(columns=['counter_id', 'site_id'], inplace=True)

X_test = final_test_df

# Train the model
model = ExtraTreesRegressor(n_estimators=100, random_state=42, verbose=3)
model.fit(X_train, y_train)

# Make predictions
y_test_pred = model.predict(X_test)

# Save results
results = pd.DataFrame({
    "Id": range(len(y_test_pred)),
    "log_bike_count": y_test_pred,
})
results.to_csv("submission.csv", index=False)
