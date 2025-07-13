import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# --- 1. Data Loading ---
# Load all the provided datasets.
# It's crucial to handle potential errors during file loading.
data_folder = 'assets'
try:
    print("Loading data...")
    train_df = pd.read_parquet(f'{data_folder}/train_data.parquet')
    test_df = pd.read_parquet(f'{data_folder}/test_data.parquet')
    add_trans_df = pd.read_parquet(f'{data_folder}/add_trans.parquet')
    add_event_df = pd.read_parquet(f'{data_folder}/add_event.parquet')
    offer_metadata_df = pd.read_parquet(f'{data_folder}/offer_metadata.parquet')
    print("Data loaded successfully.")
except FileNotFoundError as e:
    print(f"Error loading data: {e}. Make sure all parquet files are in the same directory.")
    exit()

# --- 2. Feature Engineering ---
# This is a critical step. We'll create new features to improve model accuracy.
print("Starting feature engineering...")

# Ensure consistent data types for merging
offer_metadata_df['id3'] = offer_metadata_df['id3'].astype(str)

# Merge additional data into the main training and testing dataframes.
train_df = train_df.merge(offer_metadata_df, on='id3', how='left')
test_df = test_df.merge(offer_metadata_df, on='id3', how='left')

# Ensure consistent data types for 'id2' column
train_df['id2'] = train_df['id2'].astype(str)
test_df['id2'] = test_df['id2'].astype(str)
add_trans_df['id2'] = add_trans_df['id2'].astype(str)

# Feature engineering on transaction data
# Calculate aggregate transaction features for each customer.
trans_agg = add_trans_df.groupby('id2').agg(
    trans_count=('f367', 'count'),
    trans_amount_sum=('f367', 'sum'),
    trans_amount_mean=('f367', 'mean'),
    trans_amount_std=('f367', 'std')
).reset_index()

train_df = train_df.merge(trans_agg, on='id2', how='left')
test_df = test_df.merge(trans_agg, on='id2', how='left')

# Ensure consistent data types for 'id2' in event data
add_event_df['id2'] = add_event_df['id2'].astype(str)

# Feature engineering on event data
# Count the number of events for each customer.
event_agg = add_event_df.groupby('id2').agg(
    event_count=('id6', 'count')
).reset_index()

train_df = train_df.merge(event_agg, on='id2', how='left')
test_df = test_df.merge(event_agg, on='id2', how='left')

# Convert timestamp columns to datetime objects
train_df['id4'] = pd.to_datetime(train_df['id4'])
test_df['id4'] = pd.to_datetime(test_df['id4'])
train_df['id12'] = pd.to_datetime(train_df['id12'])
test_df['id12'] = pd.to_datetime(test_df['id12'])
train_df['id13'] = pd.to_datetime(train_df['id13'])
test_df['id13'] = pd.to_datetime(test_df['id13'])


# Create time-based features
train_df['day_of_week'] = train_df['id4'].dt.dayofweek
train_df['hour_of_day'] = train_df['id4'].dt.hour
test_df['day_of_week'] = test_df['id4'].dt.dayofweek
test_df['hour_of_day'] = test_df['id4'].dt.hour

# Offer duration
train_df['offer_duration'] = (train_df['id13'] - train_df['id12']).dt.days
test_df['offer_duration'] = (test_df['id13'] - test_df['id12']).dt.days


# --- 3. Data Preprocessing ---
print("Preprocessing data...")

# Identify categorical and numerical features
# Exclude identifier columns and the target variable 'y'.
features = [col for col in train_df.columns if col.startswith('f')]
categorical_features = [col for col in features if train_df[col].dtype == 'object']
numerical_features = [col for col in features if train_df[col].dtype != 'object']


# Fill missing values.
# For numerical features, we'll use the median.
# For categorical features, we'll use the mode.
for col in numerical_features:
    median_val = train_df[col].median()
    train_df[col] = train_df[col].fillna(median_val)
    test_df[col] = test_df[col].fillna(median_val)

for col in categorical_features:
    mode_series = train_df[col].mode()
    if len(mode_series) > 0:  # Check if mode exists
        mode_val = mode_series[0]
    else:
        # If no mode exists (all values are unique), use the first non-null value or a default
        non_null_vals = train_df[col].dropna()
        mode_val = non_null_vals.iloc[0] if len(non_null_vals) > 0 else 'Unknown'
    
    train_df[col] = train_df[col].fillna(mode_val)
    test_df[col] = test_df[col].fillna(mode_val)

# Encode categorical features using Label Encoding with handling for unseen categories
for col in categorical_features:
    le = LabelEncoder()
    # Fit on training data
    train_df[col] = le.fit_transform(train_df[col].astype(str))
    
    # Transform test data, using a special value for unseen categories
    test_series = test_df[col].astype(str)
    mask = test_series.isin(le.classes_)
    
    # Initialize with a value that will be out of range for the encoder
    test_encoded = np.full(len(test_series), -1)
    
    # Only transform values that were seen during training
    test_encoded[mask] = le.transform(test_series[mask])
    
    # Assign back to the dataframe
    test_df[col] = test_encoded


# --- 4. Model Training ---
print("Training the LightGBM model...")

# Define features (X) and target (y).
# We will use all available features after preprocessing.
all_features = numerical_features + categorical_features
X = train_df[all_features]
y = train_df['y']
X_test = test_df[all_features]

# LightGBM model parameters - optimized for faster training
lgb_params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'n_estimators': 200,  # Reduced for faster training
    'learning_rate': 0.1,  # Increased for faster convergence
    'num_leaves': 15,  # Reduced to prevent overfitting and speed up
    'max_depth': 5,  # Limited depth for faster training
    'min_child_samples': 20,  # Prevents overfitting on small leaves
    'reg_alpha': 0.1,  # L1 regularization
    'reg_lambda': 0.1,  # L2 regularization
    'seed': 42,
    'n_jobs': -1,  # Use all cores
    'verbose': 1,  # Show progress
    'colsample_bytree': 0.8,
    'subsample': 0.8,
    'subsample_freq': 5,  # Frequency for bagging
}

print("Splitting data into training and validation sets...")
# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y
)

print("Training with early stopping...")
print("Feature count:", len(all_features))
print(f"Training samples: {len(X_train):,}, Validation samples: {len(X_val):,}")

# Initialize and train the model with callbacks
model = lgb.LGBMClassifier(**lgb_params)

# Callbacks for better control
callbacks = [
    lgb.early_stopping(stopping_rounds=50, verbose=True),  # Stop if no improvement for 50 rounds
    lgb.log_evaluation(period=10),  # Print progress every 10 rounds
]

# Train the model
model.fit(
    X_train, 
    y_train,
    eval_set=[(X_val, y_val)],
    eval_metric='auc',
    callbacks=callbacks
)

# Print feature importance
print("\nTop 20 most important features:")
feature_importance = pd.DataFrame({
    'feature': all_features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)
print(feature_importance.head(20).to_string())


# --- 5. Prediction and Submission File Generation ---
print("Generating predictions and submission file...")

# Predict probabilities on the test set.
predictions = model.predict_proba(X_test)[:, 1]

# Create the submission dataframe with all required columns
submission_df = pd.DataFrame({
    'id1': test_df['id1'],
    'id2': test_df['id1'].str.split('_').str[1].astype(int),  # Extract id2 from id1
    'id3': (pd.to_datetime(test_df['id1'].str.split('_').str[3] + ' ' + 
                           test_df['id1'].str.split('_').str[4], 
                          format='%Y-%m-%d %H:%M:%S.%f')
            .dt.strftime('%-m/%-d/%Y')),  # Format date as M/D/YYYY
    'id5': '',  # This will be filled from the template if available
    'pred': predictions
})

# Try to match with the template to ensure correct order and additional columns
try:
    template_df = pd.read_csv('685404e30cfdb_submission_template.csv')
    # Keep only the prediction column from our model
    submission_df = template_df[['id1', 'id2', 'id3', 'id5']].merge(
        submission_df[['id1', 'pred']], 
        on='id1', 
        how='left'
    )
except FileNotFoundError:
    print("Template file not found. Creating submission with extracted IDs.")


# Save the submission file.
# Replace `<team-name>` with your actual team name.
team_name = "YourTeamName"  # IMPORTANT: Change this to your team name (no spaces, use underscores)
submission_filename = f'r2_submission_{team_name}.csv'
submission_df.to_csv(submission_filename, index=False)

print(f"Submission file '{submission_filename}' created successfully!")
print("Good luck in the competition!")
