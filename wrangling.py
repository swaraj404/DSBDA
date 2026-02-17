import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

def wrangle():
    
    df = pd.read_csv('djt_posts_dec2025.csv')
    print(f"Dataset loaded successfully with {len(df)} records.\n")
    
  
    print(f"Shape: {df.shape}")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}\n")
    
    # 4.2 Check for missing values
    print("CHECKING FOR MISSING VALUES:")
    missing_values = df.isnull().sum()
    print(missing_values)
    print(f"\nTotal missing values: {missing_values.sum()}")
   

    print("\nColumn Names and Data Types:")
    print(df.dtypes)
    print("\nDetailed Information:")
    print(df.info())
 
    # 4.4 Statistical summary
    print("\n4.4 STATISTICAL SUMMARY:")
    print(df.describe())
    
    print("\nDescriptive statistics for object/categorical variables:")
    print(df.describe(include='object'))
    
    # Variable classification
    print("\n4.5 VARIABLE CLASSIFICATION:")
    numerical_vars = df.select_dtypes(include=['int', 'float']).columns.tolist()
    categorical_vars = df.select_dtypes(include=['object', 'bool']).columns.tolist()
    
    print(f"\nNumerical Variables ({len(numerical_vars)}): {numerical_vars}")
    print(f"\nCategorical Variables ({len(categorical_vars)}): {categorical_vars}")
    
   
    # STEP 5: DATA FORMATTING AND NORMALIZATION
    # Convert date column to datetime
    df['date'] = pd.to_datetime(df['date'])
    print("Converted 'date' to datetime format")
    
    # Extract year, month, day, hour from date
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['hour'] = df['date'].dt.hour
    df['day_of_week'] = df['date'].dt.dayofweek
    print("Extracted temporal features (year, month, day, hour, day_of_week)")
    
    # Convert boolean flags to int (0/1)
    bool_columns = ['quote_flag', 'repost_flag', 'deleted_flag']
    for col in bool_columns:
        if col in df.columns:
            df[col] = df[col].astype(int)
    print(f"Converted boolean columns to integers: {bool_columns}")
    
    # Handle missing values appropriately
    # Numeric columns: fill with median
    numeric_cols = df.select_dtypes(include=['int', 'float']).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)
            print(f"Filled missing values in '{col}' with median")
    
    # Categorical columns: fill with mode or 'Unknown'
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna('Unknown', inplace=True)
            print(f"Filled missing values in '{col}' with 'Unknown'")
    
    # 5.2 Data Normalization
    print("\n5.2 DATA NORMALIZATION:")
    
    # Normalize numerical features using StandardScaler
    scaler = StandardScaler()
    numeric_features = ['favorite_count', 'repost_count', 'word_count', 'media_count']
    
    # Create normalized versions of these columns
    for col in numeric_features:
        if col in df.columns:
            df[f'{col}_normalized'] = scaler.fit_transform(df[[col]])
    print(f"Normalized features: {numeric_features}")
    print("(Created new columns with '_normalized' suffix)")
    
    # Display summary after conversions
    print("\n5.3 DATA TYPES AFTER CONVERSION:")
    print(df.dtypes)
    
    
    # STEP 6: CATEGORICAL TO QUANTITATIVE
    print("STEP 6: TURN CATEGORICAL VARIABLES INTO QUANTITATIVE")
    
    
    # 6.1 Label Encoding for binary/ordinal categorical variables
    print("\n6.1 LABEL ENCODING:")
    
    le_platform = LabelEncoder()
    if 'platform' in df.columns:
        df['platform_encoded'] = le_platform.fit_transform(df['platform'])
        print(f"Encoded 'platform': {dict(zip(le_platform.classes_, le_platform.transform(le_platform.classes_)))}")
    
    le_handle = LabelEncoder()
    if 'handle' in df.columns:
        df['handle_encoded'] = le_handle.fit_transform(df['handle'])
        print(f"Encoded 'handle': {dict(zip(le_handle.classes_, le_handle.transform(le_handle.classes_)))}")
    
    # 6.2 One-Hot Encoding for nominal categorical variables
    print("\n6.2 ONE-HOT ENCODING:")
    
    # Example: if there are multiple platforms or handles, use one-hot encoding
    # One-hot encode platform
    if 'platform' in df.columns and df['platform'].nunique() > 2:
        platform_dummies = pd.get_dummies(df['platform'], prefix='platform', drop_first=True)
        df = pd.concat([df, platform_dummies], axis=1)
        print(f"Created one-hot encoded columns for 'platform': {platform_dummies.columns.tolist()}")
    
    # 6.3 Binary encoding for boolean text columns
    print("\n6.3 BINARY FEATURES:")
    
    # Create binary features from text analysis
    if 'text' in df.columns:
        df['has_url'] = df['urls'].notna().astype(int)
        df['has_hashtag'] = df['hashtags'].notna().astype(int)
        df['has_mention'] = df['user_mentions'].notna().astype(int)
        df['has_media'] = (df['media_count'] > 0).astype(int)
        print("Created binary features: has_url, has_hashtag, has_mention, has_media")
    
    # 6.4 Feature engineering - create engagement score
    print("\n6.4 FEATURE ENGINEERING:")
    df['engagement_score'] = df['favorite_count'] + (df['repost_count'] * 2)
    df['engagement_rate'] = df['engagement_score'] / (df['word_count'] + 1)  # +1 to avoid division by zero
    print("Created 'engagement_score' and 'engagement_rate' features")
    
    # ===========================
    # FINAL SUMMARY
    # ===========================
    print("\n" + "="*60)
    print("FINAL DATASET SUMMARY")
    print("="*60)
    print(f"\nFinal shape: {df.shape}")
    print(f"\nFirst few rows of processed data:")
    print(df.head())
    
    print("\n\nColumn names in final dataset:")
    print(df.columns.tolist())
    
    print("\n\nData types in final dataset:")
    print(df.dtypes)
    
    print("\n\nMissing values after preprocessing:")
    print(df.isnull().sum())
    
    print("\n" + "="*60)
    print("DATA WRANGLING COMPLETED SUCCESSFULLY!")
    print("="*60)
    
    return df

# Run the wrangling function
if __name__ == "__main__":
    processed_df = wrangle()
    
    # Save processed data to CSV
    processed_df.to_csv('djt_posts_processed.csv', index=False)
    print("\n✓ Processed dataset saved to 'djt_posts_processed.csv'")