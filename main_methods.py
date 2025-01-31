from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from datetime import date
from constants import *
import pandas as pd
import numpy as np
import os


def analyse_csv_file(csv_name):
    file_path = data_directory + csv_name
    os.makedirs(info_directory, exist_ok=True)  # Ensure the 'info' directory exists

    pd.set_option('display.max_columns', None)  # Ensure all columns are displayed
    pd.set_option('display.width', 1000)  # Optional: adjust display width to prevent wrapping

    # Define the path for the new .txt file
    info_file_path = os.path.join(info_directory, f"{csv_name}_info.txt")

    # Read the CSV file into a DataFrame
    try:
        df = pd.read_csv(file_path)

        rows, columns = df.shape  # Get dimensions

        # Get column names and data types
        column_details = df.dtypes

        # Prepare the output
        output = (
            f"{csv_name}:\n"
            f"{rows} rows, {columns} columns\n\n"
            f"Preview:\n{df.head(5)}\n\n"
            "Columns and Data Types:\n"
        )

        max_length = max(len(column) for column in column_details.keys()) + 1  # Add 1 for padding
        columns_info = "\n".join([f"{column.ljust(max_length)} {dtype}" for column, dtype in column_details.items()])

        output += columns_info

        # Print the output to console
        print(output)

        # Write the output to a .txt file
        with open(info_file_path, 'w') as f:
            f.write(output)

    except FileNotFoundError:
        print(f"The file {file_path} was not found.")
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")


def process_all_csv_files():
    # Ensure the directory exists
    if not os.path.isdir(data_directory):
        print(f"The directory {data_directory} does not exist.")
        return

    # List all files in the directory
    files = os.listdir(data_directory)

    # Filter for CSV files
    csv_files = [file for file in files if file.endswith('.csv')]

    # Process each CSV file
    for csv_file in csv_files:
        analyse_csv_file(csv_file)


def merge_csv_files(merge_parameters, data_directory):
    # Construct full file paths
    file_1_path = f"{data_directory}/{merge_parameters['file_1']}"
    file_2_path = f"{data_directory}/{merge_parameters['file_2']}"
    merge_file_path = f"{data_directory}/{merge_parameters['merge_file']}"

    # Read the data files
    file_1_df = pd.read_csv(file_1_path)
    file_2_df = pd.read_csv(file_2_path)
    merge_df = pd.read_csv(merge_file_path)

    # Ensure all merge columns and ID columns are of the same type, typically string
    file_1_df[merge_parameters['file_1_id_column']] = file_1_df[merge_parameters['file_1_id_column']].astype(str)
    file_2_df[merge_parameters['file_2_id_column']] = file_2_df[merge_parameters['file_2_id_column']].astype(str)
    merge_df[merge_parameters['merge_file_1_key']] = merge_df[merge_parameters['merge_file_1_key']].astype(str)
    merge_df[merge_parameters['merge_file_2_key']] = merge_df[merge_parameters['merge_file_2_key']].astype(str)

    # Merge operations
    merged_file_1 = pd.merge(merge_df, file_1_df, left_on=merge_parameters['merge_file_1_key'], right_on=merge_parameters['file_1_id_column'], how='inner')
    final_merged_df = pd.merge(merged_file_1, file_2_df, left_on=merge_parameters['merge_file_2_key'], right_on=merge_parameters['file_2_id_column'], how='inner', suffixes=('_1', '_2'))

    return final_merged_df


def merge_three_csv_files(merge_parameters):
    file_1_path = f"{data_directory}/{merge_parameters['file_1']}"
    file_2_path = f"{data_directory}/{merge_parameters['file_2']}"
    file_3_path = f"{data_directory}/{merge_parameters['file_3']}"

    file_1_df = pd.read_csv(file_1_path)
    file_2_df = pd.read_csv(file_2_path)
    file_3_df = pd.read_csv(file_3_path)

    print("Columns in file_1:", file_1_df.columns)
    print("Columns in file_2:", file_2_df.columns)
    print("Columns in file_3:", file_3_df.columns)

    file_1_df[merge_parameters['file_1_id_column']] = file_1_df[merge_parameters['file_1_id_column']].astype(str)
    file_2_df[merge_parameters['file_2_id_column']] = file_2_df[merge_parameters['file_2_id_column']].astype(str)
    file_3_df[merge_parameters['file_3_id_column']] = file_3_df[merge_parameters['file_3_id_column']].astype(str)

    # Perform the merges
    # Assuming that the correct keys have been verified and updated as needed
    first_merge = pd.merge(
        file_1_df,
        file_2_df,
        left_on=merge_parameters['merge_file_1_2_key_1'],
        right_on=merge_parameters['merge_file_1_2_key_2'],
        how='inner',
        suffixes=('_1', '_2')
    )
    final_merged_df = pd.merge(
        first_merge,
        file_3_df,
        left_on=merge_parameters['merge_file_1_3_key_1'],
        right_on=merge_parameters['merge_file_1_3_key_3'],
        how='inner',
        suffixes=('_12', '_3')
    )

    return final_merged_df

def prepare_features_labels_df(csv_files_specification):
    # Load the features and labels information
    features_info = csv_files_specification['features'][0]
    labels_info = csv_files_specification['labels']

    # Load and select columns from the features file
    features_df = pd.read_csv(f"{data_directory}/{features_info['file']}")
    features_df = features_df[['id'] + features_info['columns']]  # Ensure 'id' is included for merging

    # Load and select columns from the labels file
    labels_df = pd.read_csv(f"{data_directory}/{labels_info['file']}")
    labels_df = labels_df[['id'] + labels_info['columns']]  # Ensure 'id' is included for merging

    # Check if 'merge' dictionary is provided and not empty
    if csv_files_specification.get('merge'):
        merge_info = csv_files_specification['merge']
        merge_df = pd.read_csv(f"{data_directory}/{merge_info['file']}")

        # Merge the features DataFrame with the merge_df to map feature IDs to label IDs
        features_merged = pd.merge(merge_df, features_df, left_on=merge_info['first_file_id'], right_on='id', how='inner')

        # Merge the labels DataFrame with the merge_df (now features_merged) to include labels
        final_df = pd.merge(features_merged, labels_df, left_on=merge_info['second_file_id'], right_on='id', how='inner',
                            suffixes=('_feat', '_label'))

        # Drop the unnecessary 'id' columns based on suffixes from the merge operation
        final_df.drop(columns=[merge_info['first_file_id'], merge_info['second_file_id'], 'id_feat'], inplace=True)
    else:
        # If merge dictionary is empty, work directly with features and labels DataFrames
        final_df = pd.concat([features_df.set_index('id'), labels_df.set_index('id')], axis=1,
                             join='inner').reset_index()

    # Ensuring only specified feature and label columns are included, plus the 'id' column
    desired_columns = features_info['columns'] + labels_info['columns']
    final_df = final_df[desired_columns]

    return final_df


def infer_data_types(df):
    datetime_format = '%Y-%m-%d %H:%M:%S'

    for column in df.columns:
        # Drop NA values for type inference
        sample_data = df[column].dropna().unique()

        # Attempt to convert to datetime with the specified format
        try:
            converted = pd.to_datetime(df[column], format=datetime_format, errors='raise')
            # If conversion is successful and no errors are raised, update the column
            df[column] = converted
            continue
        except ValueError:
            # If conversion fails, proceed with other type detection
            pass

        # Numeric detection
        if pd.to_numeric(df[column], errors='coerce').notna().all():
            df[column] = pd.to_numeric(df[column])
            continue

        # Boolean detection
        if set(sample_data).issubset({True, False, 'True', 'False', 'true', 'false', 1, 0, '1', '0'}):
            df[column] = df[column].map(
                {'True': True, 'true': True, 'False': False, 'false': False, '1': True, '0': False, 1: True,
                 0: False}).astype('boolean')
            continue

        # Category detection based on the number of unique values
        if len(sample_data) / df.shape[0] < 0.05:
            df[column] = df[column].astype('category')
            continue

        # Default to string if no other type matches
        df[column] = df[column].astype('string')

    return df


def split_date_data(df, date_columns):
    df_split = df.copy()

    for column in date_columns:
        if column in df_split.columns:
            # Convert column to datetime if it's not already
            df_split[column] = pd.to_datetime(df_split[column])

            # Spread out into multiple columns
            df_split[f'{column}_year'] = df_split[column].dt.year
            df_split[f'{column}_month'] = df_split[column].dt.month
            df_split[f'{column}_day'] = df_split[column].dt.day
            df_split[f'{column}_hour'] = df_split[column].dt.hour

            # Optionally, drop the original datetime column
            df_split.drop(columns=[column], inplace=True)

    return df_split


def add_is_weekend(df, year_col, month_col, day_col, col_name):
    def is_weekend(year, month, day):
        try:
            # Ensure year, month, and day are explicitly converted to integers
            year = int(year)
            month = int(month)
            day = int(day)
            # Create a date object for the given year, month, and day
            date_obj = date(year, month, day)
            # Determine if the date is a weekend
            return 1 if date_obj.weekday() >= 5 else 0
        except ValueError:
            # Handle invalid date cases
            return 0

    # Apply the is_weekend function to each row
    # Access scalar values for year, month, and day directly
    df[f'{col_name}_is_weekend'] = df.apply(lambda row: is_weekend(row[year_col].item(), row[month_col].item(), row[day_col].item()), axis=1)

    return df


def encode_columns(df, columns_to_encode):
    """
    Encodes specified columns in the DataFrame using label encoding.
    Returns DataFrame with specified columns encoded, and a dictionary of encoding mappings.
    """
    df_encoded = df.copy()
    encodings = {}  # Dictionary to hold encoding mappings for each column

    for column in columns_to_encode:
        if column in df_encoded.columns:
            le = LabelEncoder()
            df_encoded[column] = le.fit_transform(df_encoded[column])
            # Store the encoding mapping for the current column
            encodings[column] = {encoded: original for original, encoded in enumerate(le.classes_)}
            print(f"Encoding for '{column}': {encodings[column]}")
        else:
            print(f"Column {column} not found in DataFrame.")

    return df_encoded, encodings

