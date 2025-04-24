import streamlit as st
import joblib
import pandas as pd
import numpy as np
import json

# Load the trained model
model = joblib.load("model.pkl")

# Define consistent column names, file paths, date ranges, and categorical features list
DATE_COLUMN = 'Week_Start_Date'
SKU_COLUMN = 'SKU'
TARGET_COLUMN = 'Target_Weekly_Quantity_Needed'
CATEGORICAL_FEATURES = ['Item_Category', 'Unit_of_Measure']
START_DATE = "2021-01-01"
END_DATE = "2024-12-31"

numeric_cols_with_lags = ['Needed_Qty_Lag_1W', 'Needed_Qty_Lag_2W', 'Needed_Qty_Lag_4W', 'Avg_Needed_Qty_Last_4W']

# Define the function to predict future demand and calculate order quantities
print("\n--- Defining Order Prediction Function ---")
def predict_and_suggest_orders(
    model, future_ops_data: pd.DataFrame, latest_demand_hist: pd.DataFrame,
    products_info: pd.DataFrame, current_stock: pd.DataFrame,
    categorical_features_list: list, training_cols: list ):
    """ Predicts demand for the next week(s) and suggests order quantities. """
    print("\nStarting prediction for future week(s)...")
    # --- Input Checks ---
    if future_ops_data.empty or latest_demand_hist.empty or products_info.empty or current_stock.empty: print("Error: One or more input DataFrames are empty."); return pd.DataFrame()
    required_product_cols = [SKU_COLUMN, 'Item_Category', 'Unit_of_Measure', 'Avg_Unit_Cost_Base', 'Min_Order_Quantity']; required_stock_cols = [SKU_COLUMN, 'Current_Stock_Level']; required_hist_cols = [DATE_COLUMN, SKU_COLUMN, TARGET_COLUMN]; required_ops_cols = [DATE_COLUMN]
    if not all(col in products_info.columns for col in required_product_cols): print(f"Error: products_info missing required columns."); return pd.DataFrame()
    if not all(col in current_stock.columns for col in required_stock_cols): print(f"Error: current_stock missing required columns."); return pd.DataFrame()
    if not all(col in latest_demand_hist.columns for col in required_hist_cols): print(f"Error: latest_demand_hist missing required columns."); return pd.DataFrame()
    if not all(col in future_ops_data.columns for col in required_ops_cols): print(f"Error: future_ops_data missing required columns."); return pd.DataFrame()

    # --- Prepare features ---
    future_ops_data[DATE_COLUMN] = pd.to_datetime(future_ops_data[DATE_COLUMN]); latest_demand_hist[DATE_COLUMN] = pd.to_datetime(latest_demand_hist[DATE_COLUMN])
    prediction_features = future_ops_data.copy()
    latest_demand_hist = latest_demand_hist.sort_values(by=[SKU_COLUMN, DATE_COLUMN]); sku_list = products_info[SKU_COLUMN].unique(); lag_features_list = []
    for sku in sku_list:
        sku_hist = latest_demand_hist[latest_demand_hist[SKU_COLUMN] == sku].tail(4)
        if len(sku_hist) < 1: lag_1w, lag_2w, lag_4w, avg_4w = 0, 0, 0, 0
        else: lag_1w = sku_hist[TARGET_COLUMN].iloc[-1] if len(sku_hist) >= 1 else 0; lag_2w = sku_hist[TARGET_COLUMN].iloc[-2] if len(sku_hist) >= 2 else 0; lag_4w = sku_hist[TARGET_COLUMN].iloc[-4] if len(sku_hist) >= 4 else 0; avg_4w = sku_hist[TARGET_COLUMN].mean()
        lag_features_list.append({ SKU_COLUMN: sku, 'Needed_Qty_Lag_1W': lag_1w, 'Needed_Qty_Lag_2W': lag_2w, 'Needed_Qty_Lag_4W': lag_4w, 'Avg_Needed_Qty_Last_4W': avg_4w })
    lag_features_df = pd.DataFrame(lag_features_list)
    sku_df_template = products_info[[SKU_COLUMN, 'Item_Category', 'Unit_of_Measure', 'Avg_Unit_Cost_Base', 'Supplier_Lead_Time_Days', 'Min_Order_Quantity', 'Shelf_Life_Days']].copy()
    prediction_features = prediction_features.merge(sku_df_template, how='cross')
    prediction_features = pd.merge(prediction_features, lag_features_df, on=SKU_COLUMN, how='left')
    prediction_features[numeric_cols_with_lags] = prediction_features[numeric_cols_with_lags].fillna(0) # Use global list name
    prediction_features['Avg_Unit_Cost'] = prediction_features['Avg_Unit_Cost_Base']
    if 'Avg_Unit_Cost_Base' not in training_cols: prediction_features = prediction_features.drop(columns=['Avg_Unit_Cost_Base'], errors='ignore')
    existing_categorical_pred = [col for col in categorical_features_list if col in prediction_features.columns]
    if existing_categorical_pred: prediction_features = pd.get_dummies(prediction_features, columns=existing_categorical_pred, drop_first=True, dummy_na=False)

    # Store Identifiers BEFORE aligning
    if DATE_COLUMN not in prediction_features.columns or SKU_COLUMN not in prediction_features.columns: print(f"CRITICAL Error: Identifiers missing before alignment."); return pd.DataFrame()
    identifiers = prediction_features[[DATE_COLUMN, SKU_COLUMN]].copy()

    # Align columns
    print(f"Aligning prediction columns with {len(training_cols)} training columns...")
    X_pred = pd.DataFrame(columns=training_cols, index=prediction_features.index)
    present_cols = prediction_features.columns
    for col in training_cols:
        if col in present_cols: X_pred[col] = prediction_features[col]
        else: X_pred[col] = 0

    # Make Predictions
    print(f"Predicting demand for {len(X_pred)} SKU-Week combinations...")
    predicted_needs = model.predict(X_pred)
    predicted_needs[predicted_needs < 0] = 0

    # Prepare Output
    output_df = identifiers; output_df['Predicted_Need'] = predicted_needs
    output_df = pd.merge(output_df, current_stock, on=SKU_COLUMN, how='left'); output_df['Current_Stock_Level'] = output_df['Current_Stock_Level'].fillna(0)
    if SKU_COLUMN not in products_info.columns: print(f"Error: '{SKU_COLUMN}' missing from products_info for MOQ merge."); return pd.DataFrame()
    output_df = pd.merge(output_df, products_info[[SKU_COLUMN, 'Min_Order_Quantity']], on=SKU_COLUMN, how='left'); output_df['Min_Order_Quantity'] = output_df['Min_Order_Quantity'].fillna(1)
    output_df['Calculated_Need_Vs_Stock'] = output_df['Predicted_Need'] - output_df['Current_Stock_Level']
    output_df['Suggested_Order_Qty'] = output_df['Calculated_Need_Vs_Stock'].apply(lambda x: max(0, np.ceil(x) if x > 0.1 else np.round(x,2) ) )

    return output_df[[DATE_COLUMN, SKU_COLUMN, 'Predicted_Need', 'Current_Stock_Level', 'Calculated_Need_Vs_Stock', 'Min_Order_Quantity', 'Suggested_Order_Qty']]

def predict_demand_and_orders(
    future_ops_data,
    latest_demand_hist_csv,
    products_info_csv,
    current_stock_csv
) :
    """
    Predicts demand for the next week(s) and suggests order quantities.

    Args:
        future_ops_data (Dict): Dictionary representing future operational data.
        latest_demand_hist_csv (modelbit.Csv): Latest demand history as a CSV file.
        products_info_csv (modelbit.Csv): Product information as a CSV file.
        current_stock_csv (modelbit.Csv): Current stock levels as a CSV file.

    Returns:
        List[Dict]: List of dictionaries containing order suggestions.
    """

    # Read CSV files into Pandas DataFrames
    latest_demand_hist_df = pd.read_csv(latest_demand_hist_csv)
    products_info_df = pd.read_csv(products_info_csv)
    current_stock_df = pd.read_csv(current_stock_csv)

    # Convert future_ops_data to DataFrame
    #future_ops_df = pd.DataFrame([future_ops_data])
    #future_ops_df[DATE_COLUMN]= pd.Timestamp(future_ops_df[DATE_COLUMN])

    # Call predict_and_suggest_orders() to generate order suggestions
    order_suggestions_df = predict_and_suggest_orders(
        model=model,  # Assuming 'lgbm' is your trained model
        future_ops_data=future_ops_data,
        latest_demand_hist=latest_demand_hist_df,
        products_info=products_info_df,
        current_stock=current_stock_df,
        categorical_features_list=CATEGORICAL_FEATURES,
        training_cols=['Avg_Unit_Cost', 'Needed_Qty_Lag_1W', 'Needed_Qty_Lag_2W', 'Needed_Qty_Lag_4W', 'Avg_Needed_Qty_Last_4W', 'Forecasted_Occupancy_Percent', 'Forecasted_Event_Guests', 'Week_of_Year', 'Month', 'Is_Peak_Season', 'Is_Low_Season', 'Is_Ramadan_Week', 'Is_Eid_Week', 'Is_Public_Holiday_Week', 'Percent_Guests_GCC', 'Percent_Guests_Europe', 'Percent_Guests_Asia', 'Percent_Guests_Americas', 'Percent_Guests_Africa_ME', 'Percent_Guests_Other_Region', 'Percent_Guests_Leisure', 'Percent_Guests_Business', 'Percent_Guests_Group', 'Percent_Families', 'Percent_Adult_Only_Bookings', 'Min_Order_Quantity', 'Shelf_Life_Days', 'Supplier_Lead_Time_Days', 'Item_Category_Beverage_Alcoholic_Beer', 'Item_Category_Beverage_Alcoholic_Spirit', 'Item_Category_Beverage_Alcoholic_Wine_White', 'Item_Category_Beverage_Soft_Cola', 'Item_Category_Beverage_Water_Still', 'Item_Category_Dairy_Cheese', 'Item_Category_Dairy_Milk', 'Item_Category_Dairy_Yogurt', 'Item_Category_Dates_Sweets', 'Item_Category_Dry_Goods_Flour', 'Item_Category_Dry_Goods_Oil', 'Item_Category_Dry_Goods_Pasta', 'Item_Category_Dry_Goods_Rice', 'Item_Category_Dry_Goods_Spice', 'Item_Category_Frozen_Goods', 'Item_Category_Meat_Beef', 'Item_Category_Meat_Lamb', 'Item_Category_Meat_Pork', 'Item_Category_Poultry', 'Item_Category_Produce_Fruit_Berry', 'Item_Category_Produce_Fruit_Tropical', 'Item_Category_Produce_Veg_Leafy', 'Item_Category_Seafood_Fish', 'Item_Category_Seafood_Shellfish', 'Unit_of_Measure_Bottle_330ml', 'Unit_of_Measure_Bottle_750ml', 'Unit_of_Measure_Box_10KG', 'Unit_of_Measure_Case_24', 'Unit_of_Measure_Dozen', 'Unit_of_Measure_Each', 'Unit_of_Measure_KG', 'Unit_of_Measure_Litre', 'Unit_of_Measure_Pack_6']
    )

    # Convert the resulting DataFrame back to a list of dictionaries
    order_suggestions = order_suggestions_df.to_dict(orient='records')

    return order_suggestions

def parse_json_string_to_dataframe(json_string):
  """
  Parses a JSON string (representing a single record) into a
  single-row Pandas DataFrame.

  It specifically converts the 'Week_Start_Date' field into a
  Pandas Timestamp object.

  Args:
    json_string (str): A JSON formatted string containing the data
                       for one record.

  Returns:
    pandas.DataFrame: A DataFrame containing the single record, with
                      'Week_Start_Date' correctly typed as Timestamp.
                      Returns None if input is not a valid JSON string,
                      or if processing fails.
  """
  if not isinstance(json_string, str):
      print("Error: Input must be a string.")
      return None

  # Step 1: Parse the JSON string into a Python dictionary
  try:
      data_dict = json.loads(json_string)
  except json.JSONDecodeError as e:
      print(f"Error: Invalid JSON string provided: {e}")
      return None
  except Exception as e:
      print(f"An unexpected error occurred during JSON parsing: {e}")
      return None

  # Check if the parsed result is a dictionary (expected for a single record)
  if not isinstance(data_dict, dict):
      print("Error: Parsed JSON content is not a dictionary (expected single record).")
      return None

  # Step 2: Create DataFrame from the parsed dictionary
  try:
      # Wrap dict in a list as DataFrame expects list of records for rows
      df = pd.DataFrame([data_dict])
  except Exception as e:
      print(f"Error creating DataFrame from parsed dictionary: {e}")
      return None

  # Step 3: Check if the required date column exists
  if 'Week_Start_Date' not in df.columns:
      print("Error: 'Week_Start_Date' column not found in the parsed JSON data.")
      return None

  # Step 4: Convert the 'Week_Start_Date' column to pandas Timestamp objects
  try:
      df['Week_Start_Date'] = pd.to_datetime(df['Week_Start_Date'])
  except Exception as e:
      print(f"Error converting 'Week_Start_Date' to Timestamp: {e}")
      # Depending on requirements, you might return None or the DataFrame
      # with the original string type here. Returning None for clarity.
      return None

  return df

st.title("📄 Upload CSV for Analysis")

with(st.form("form1")):
    ops= st.text_input("ops: ", type="default")
# Upload CSV
    products_df = st.file_uploader("Choose a CSV file", type="csv", key="p")
    lastest_hists = st.file_uploader("Choose a CSV file", type="csv", key="l")
    current_stock = st.file_uploader("Choose a CSV file", type="csv", key='c')

    submit= st.button("calc")

if(submit):
    text= predict_demand_and_orders(parse_json_string_to_dataframe(ops), lastest_hists, products_df, current_stock)
    st.code(text)
