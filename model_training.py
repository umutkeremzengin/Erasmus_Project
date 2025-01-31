from model_training_methods import *
from main_methods import *
from constants import *
import pandas as pd
import os


# Pandas settings
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

########################################################################################################################

# """
# # Append the following to our DF
# # Using government data (2017 - 2022)
# """
# - Traffic
# - Weather forecast
# """

"""
Feature Columns:    
slotbooking.csv
    id    
    startdate
    palletamount
    handling
    duration

carrier.csv
    code
    name

order.csv
    productcategory
    loadingpta
    punctuality - CLASS

To generate:
slotbooking.csv
    day_of_week (startdate) 
    is_holiday (startdate)
"""

slotbooking_carrier_description = {
    'file_1': 'slotbooking.csv',
    'file_2': 'carrier.csv',
    'merge_file': 'slotbooking_carrier.csv',
    'merge_file_1_key': 'slotplanning$slotbookingid',
    'merge_file_2_key': 'crm$carrierid',
    'file_1_id_column': 'id',
    'file_2_id_column': 'id',
}

slotbooking_order_description = {
    'file_1': 'slotbooking.csv',
    'file_2': 'order.csv',
    'merge_file': 'slotbooking_order.csv',
    'merge_file_1_key': 'slotplanning$slotbookingid',
    'merge_file_2_key': 'slotplanning$orderid',
    'file_1_id_column': 'id',
    'file_2_id_column': 'id',
}

# Merge the .csv files
slotbooking_carrier_df = merge_csv_files(slotbooking_carrier_description, "data_files")

slotbooking_order_df = merge_csv_files(slotbooking_order_description, "data_files")

merged_df = pd.merge(slotbooking_carrier_df, slotbooking_order_df, on='id_1', how='inner')

columns_to_exclude = [col for col in slotbooking_carrier_df.columns if col in slotbooking_order_df.columns and col != 'id_1']

merged_df = pd.merge(slotbooking_carrier_df, slotbooking_order_df.drop(columns=columns_to_exclude), on='id_1', how='inner')

# Sort by 'id_1' and 'changeddate' to ensure the latest 'changeddate' is kept
merged_df = merged_df.sort_values(by=['id_1', 'changeddate'], ascending=[True, False])

# Drop useless columns
features_labels_df = merged_df[["id_1", "startdate", "palletamount", "handling", "duration", "code", "name", "productcategory", "loadingpta", "punctuality"]]  # Label: "punctuality"

# Remove duplicates, keeping the first occurrence of each id_1 (which will be the row with the latest 'changeddate')
features_labels_df = features_labels_df.drop_duplicates(subset='id_1')

# Drop rows with missing data
features_labels_df = features_labels_df.dropna()

# Remove all non E02_OUTBOUND instances of the "handling" column
features_labels_df = features_labels_df[features_labels_df["handling"] == "E02_OUTBOUND"] # "handling" = "E02_OUTBOUND"

# Since we only have E02_OUTBOUND instances, we can remove the "handling" column
features_labels_df = features_labels_df.drop(columns=["handling"])

# Split startdate columns
# Split loadingpta columns
features_labels_df = split_date_data(features_labels_df, ['startdate'])
features_labels_df = split_date_data(features_labels_df, ['loadingpta'])

# Add "startdate_is_weekend" column
# Add "loadingpta_is_weekend" column
features_labels_df = add_is_weekend(features_labels_df, ["startdate_year"], ["startdate_month"], ["startdate_day"], "startdate")
features_labels_df = add_is_weekend(features_labels_df, ["loadingpta_year"], ["loadingpta_month"], ["loadingpta_day"], "loadingpta")

# Remove "startdate_year" column
# Remove "loadingpta_year" column
features_labels_df = features_labels_df.drop(columns=["startdate_year"])
features_labels_df = features_labels_df.drop(columns=["loadingpta_year"])

# Encode features
features_labels_df, encodings_dictionary = encode_columns(features_labels_df, ["code", "name", "productcategory"])

# Remove decimal values
features_labels_df['duration'] = features_labels_df['duration'].round().astype(int)
features_labels_df['punctuality'] = features_labels_df['punctuality'].round().astype(int)

# Encode "punctuality" ("punctuality mod 60" for negative values, and "punctuality - duration mod 60" for positive values)
features_labels_df['punctuality'] = np.where(
    features_labels_df['punctuality'] > 0,
    (features_labels_df['punctuality'] - features_labels_df['duration']) // 60,
    features_labels_df['punctuality'] // 60
)

# Combine all cases <= -4 into "-4"
features_labels_df['punctuality'] = np.where(
    features_labels_df['punctuality'] < -0, -1, features_labels_df['punctuality']
)

# Combine all cases >= 4 into "4"
features_labels_df['punctuality'] = np.where(
    features_labels_df['punctuality'] > 0, 1, features_labels_df['punctuality']
)

# Remove extreme values
features_labels_df = features_labels_df[(features_labels_df['punctuality'] <= 24) & (features_labels_df['punctuality'] >= -48)]

# Reset indices
features_labels_df.reset_index(drop=True, inplace=True)
#
# # Uncomment this to set "status" to either 0 (completed) or 1 (not completed).
# # features_labels_df["status"] = features_labels_df["status"].apply(lambda x: 1 if x != 0 else 0)
# # encodings_dictionary = {'handling': {'E01_INBOUND': 0, 'E02_OUTBOUND': 1}, 'status': {'COMPLETED': 0, 'NOT COMPLETED': 1}}
#
# # Train a decision tree model
# # trained_dt_model = train_decision_tree_model(features_labels_df, ["startdate_month", "startdate_day", "startdate_hour", "loadingpta_month", "loadingpta_day", "loadingpta_hour", "palletamount", "duration", "code", "name", "productcategory"], ["punctuality"], encodings_dictionary)
#
# # Train a NB model
# # trained_model = train_naive_bayes_model(features_labels_df)
#
# Train a random forest model
trained_rf_model, rf_probabilities = train_random_forest_model(features_labels_df)






