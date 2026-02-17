# Import pandas library
import pandas as pd

# Step 1-3: Load the Excel file
print("Loading Excel file...")
df = pd.read_excel('Demo Marks.xlsx')
print("File loaded successfully!")
print()

# STEP 4: DATA PREPROCESSING
print("=" * 50)
print("STEP 4: DATA PREPROCESSING")
print("=" * 50)

# Check dimensions
print("\n1. Dimensions of dataset:")
print("Number of rows:", df.shape[0])
print("Number of columns:", df.shape[1])

# Check for missing values
print("\n2. Missing values in each column:")
print(df.isnull().sum())

# Use describe() function
print("\n3. Statistical summary using describe():")
print(df.describe())

# Check data types
print("\n4. Data types of each column:")
print(df.dtypes)

# STEP 5: DATA FORMATTING AND NORMALIZATION
print("\n" + "=" * 50)
print("STEP 5: DATA FORMATTING AND NORMALIZATION")
print("=" * 50)

# Fill missing values with median
print("\nFilling missing values...")
df['%'] = df['%'].fillna(df['%'].median())
df['UT 1 Marks'] = df['UT 1 Marks'].fillna(df['UT 1 Marks'].median())
df['%.1'] = df['%.1'].fillna(df['%.1'].median())
print("Missing values filled!")

# Normalize Total marks (between 0 and 1)
print("\nNormalizing marks...")
df['Normalized_Total'] = df['Total 100%'] / 100
print("Normalization done!")

# STEP 6: TURN CATEGORICAL TO QUANTITATIVE
print("\n" + "=" * 50)
print("STEP 6: CATEGORICAL TO QUANTITATIVE")
print("=" * 50)

# Convert Gender to numbers: M=1, F=0
print("\nConverting Gender to numbers...")
df['Gender_Numeric'] = df['Gender '].apply(lambda x: 1 if x.strip() == 'M' else 0)
print("M = 1, F = 0")

# Create performance category
print("\nCreating performance categories...")
for i in range(len(df)):
    marks = df.loc[i, 'Total 100%']
    if marks >= 75:
        df.loc[i, 'Performance'] = 'Excellent'
    elif marks >= 60:
        df.loc[i, 'Performance'] = 'Good'
    elif marks >= 40:
        df.loc[i, 'Performance'] = 'Pass'
    else:
        df.loc[i, 'Performance'] = 'Fail'

# Convert performance to numbers
df['Performance_Numeric'] = df['Performance'].replace({
    'Excellent': 3,
    'Good': 2,
    'Pass': 1,
    'Fail': 0
})
print("Performance categories created!")

# FINAL SUMMARY
print("\n" + "=" * 50)
print("FINAL SUMMARY")
print("=" * 50)
print("\nFirst 5 rows of processed data:")
print(df[['Name of Student', 'Total 100%', 'Gender_Numeric', 'Performance']].head())

print("\nFinal shape:", df.shape)
print("Missing values:", df.isnull().sum().sum())

# Save to CSV
df.to_csv('Demo_Marks_Processed.csv', index=False)
print("\nProcessed data saved to 'Demo_Marks_Processed.csv'")
print("\nDone!")
