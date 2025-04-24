import streamlit as st
import joblib
import pandas as pd
import numpy as np
import json # Keep json import in case model expects specific string formatting somewhere? Unlikely. Can be removed if not needed.
import datetime

# --- Model Loading & Constants ---
try:
    model = joblib.load("model.pkl")
    st.sidebar.success("Model loaded successfully!")
except FileNotFoundError:
    st.error("Error: model.pkl not found. Please ensure the model file is in the correct directory.")
    st.stop() # Stop execution if model file is missing
except Exception as e:
    st.error(f"Error loading model.pkl: {e}")
    st.stop()

# Define consistent column names, file paths, date ranges, and categorical features list
DATE_COLUMN = 'Week_Start_Date'
SKU_COLUMN = 'SKU'
TARGET_COLUMN = 'Target_Weekly_Quantity_Needed'
CATEGORICAL_FEATURES = ['Item_Category', 'Unit_of_Measure']
# Define numeric columns with lags (used within predict_and_suggest_orders)
numeric_cols_with_lags = ['Needed_Qty_Lag_1W', 'Needed_Qty_Lag_2W', 'Needed_Qty_Lag_4W', 'Avg_Needed_Qty_Last_4W']

# Define the fixed training columns list (as provided in the original snippet)
# CRITICAL: Ensure this list accurately reflects the columns the model was trained on.
TRAINING_COLS = ['Avg_Unit_Cost', 'Needed_Qty_Lag_1W', 'Needed_Qty_Lag_2W', 'Needed_Qty_Lag_4W', 'Avg_Needed_Qty_Last_4W', 'Forecasted_Occupancy_Percent', 'Forecasted_Event_Guests', 'Week_of_Year', 'Month', 'Is_Peak_Season', 'Is_Low_Season', 'Is_Ramadan_Week', 'Is_Eid_Week', 'Is_Public_Holiday_Week', 'Percent_Guests_GCC', 'Percent_Guests_Europe', 'Percent_Guests_Asia', 'Percent_Guests_Americas', 'Percent_Guests_Africa_ME', 'Percent_Guests_Other_Region', 'Percent_Guests_Leisure', 'Percent_Guests_Business', 'Percent_Guests_Group', 'Percent_Families', 'Percent_Adult_Only_Bookings', 'Min_Order_Quantity', 'Shelf_Life_Days', 'Supplier_Lead_Time_Days', 'Item_Category_Beverage_Alcoholic_Beer', 'Item_Category_Beverage_Alcoholic_Spirit', 'Item_Category_Beverage_Alcoholic_Wine_White', 'Item_Category_Beverage_Soft_Cola', 'Item_Category_Beverage_Water_Still', 'Item_Category_Dairy_Cheese', 'Item_Category_Dairy_Milk', 'Item_Category_Dairy_Yogurt', 'Item_Category_Dates_Sweets', 'Item_Category_Dry_Goods_Flour', 'Item_Category_Dry_Goods_Oil', 'Item_Category_Dry_Goods_Pasta', 'Item_Category_Dry_Goods_Rice', 'Item_Category_Dry_Goods_Spice', 'Item_Category_Frozen_Goods', 'Item_Category_Meat_Beef', 'Item_Category_Meat_Lamb', 'Item_Category_Meat_Pork', 'Item_Category_Poultry', 'Item_Category_Produce_Fruit_Berry', 'Item_Category_Produce_Fruit_Tropical', 'Item_Category_Produce_Veg_Leafy', 'Item_Category_Seafood_Fish', 'Item_Category_Seafood_Shellfish', 'Unit_of_Measure_Bottle_330ml', 'Unit_of_Measure_Bottle_750ml', 'Unit_of_Measure_Box_10KG', 'Unit_of_Measure_Case_24', 'Unit_of_Measure_Dozen', 'Unit_of_Measure_Each', 'Unit_of_Measure_KG', 'Unit_of_Measure_Litre', 'Unit_of_Measure_Pack_6']


# --- Core Prediction Logic (Function remains the same) ---
def predict_and_suggest_orders(
    model, future_ops_data: pd.DataFrame, latest_demand_hist: pd.DataFrame,
    products_info: pd.DataFrame, current_stock: pd.DataFrame,
    categorical_features_list: list, training_cols: list ):
    """ Predicts demand for the next week(s) and suggests order quantities. """
    st.write("\nStarting prediction for future week(s)...") # Use st.write for Streamlit output
    # --- Input Checks ---
    if future_ops_data is None or future_ops_data.empty: st.error("Error: Future operational data is missing or empty after parsing."); return pd.DataFrame()
    if latest_demand_hist.empty: st.error("Error: Latest demand history DataFrame is empty."); return pd.DataFrame()
    if products_info.empty: st.error("Error: Products info DataFrame is empty."); return pd.DataFrame()
    if current_stock.empty: st.error("Error: Current stock DataFrame is empty."); return pd.DataFrame()

    required_product_cols = [SKU_COLUMN, 'Item_Category', 'Unit_of_Measure', 'Avg_Unit_Cost_Base', 'Min_Order_Quantity', 'Supplier_Lead_Time_Days', 'Shelf_Life_Days']
    required_stock_cols = [SKU_COLUMN, 'Current_Stock_Level']
    required_hist_cols = [DATE_COLUMN, SKU_COLUMN, TARGET_COLUMN]
    # Define the operational columns expected in the input DataFrame based on training columns (excluding generated/product/history ones)
    ops_cols_needed_in_df = [
        DATE_COLUMN, 'Forecasted_Occupancy_Percent', 'Forecasted_Event_Guests', 'Week_of_Year', 'Month',
        'Is_Peak_Season', 'Is_Low_Season', 'Is_Ramadan_Week', 'Is_Eid_Week', 'Is_Public_Holiday_Week',
        'Percent_Guests_GCC', 'Percent_Guests_Europe', 'Percent_Guests_Asia', 'Percent_Guests_Americas',
        'Percent_Guests_Africa_ME', 'Percent_Guests_Other_Region', 'Percent_Guests_Leisure',
        'Percent_Guests_Business', 'Percent_Guests_Group', 'Percent_Families', 'Percent_Adult_Only_Bookings'
    ]

    if not all(col in products_info.columns for col in required_product_cols): st.error(f"Error: products_info missing one or more required columns: {required_product_cols}. Found: {products_info.columns.tolist()}"); return pd.DataFrame()
    if not all(col in current_stock.columns for col in required_stock_cols): st.error(f"Error: current_stock missing required columns: {required_stock_cols}. Found: {current_stock.columns.tolist()}"); return pd.DataFrame()
    if not all(col in latest_demand_hist.columns for col in required_hist_cols): st.error(f"Error: latest_demand_hist missing required columns: {required_hist_cols}. Found: {latest_demand_hist.columns.tolist()}"); return pd.DataFrame()
    # Check the operational data DataFrame built from form inputs
    if not all(col in future_ops_data.columns for col in ops_cols_needed_in_df): st.error(f"Error: future_ops_data (from form) missing required columns: {ops_cols_needed_in_df}. Found: {future_ops_data.columns.tolist()}"); return pd.DataFrame()


    # --- Prepare features ---
    try:
        future_ops_data[DATE_COLUMN] = pd.to_datetime(future_ops_data[DATE_COLUMN])
        latest_demand_hist[DATE_COLUMN] = pd.to_datetime(latest_demand_hist[DATE_COLUMN])
    except Exception as e:
        st.error(f"Error converting date columns to datetime: {e}")
        return pd.DataFrame()

    prediction_features = future_ops_data.copy()

    # Calculate Lag Features based on latest_demand_hist
    latest_demand_hist = latest_demand_hist.sort_values(by=[SKU_COLUMN, DATE_COLUMN])
    sku_list = products_info[SKU_COLUMN].unique()
    lag_features_list = []

    for sku in sku_list:
        sku_hist = latest_demand_hist[latest_demand_hist[SKU_COLUMN] == sku].tail(4)
        lag_1w = sku_hist[TARGET_COLUMN].iloc[-1] if len(sku_hist) >= 1 else 0
        lag_2w = sku_hist[TARGET_COLUMN].iloc[-2] if len(sku_hist) >= 2 else 0
        lag_4w = sku_hist[TARGET_COLUMN].iloc[-4] if len(sku_hist) >= 4 else 0
        avg_4w = sku_hist[TARGET_COLUMN].mean() if not sku_hist.empty else 0
        lag_features_list.append({
            SKU_COLUMN: sku,
            'Needed_Qty_Lag_1W': lag_1w,
            'Needed_Qty_Lag_2W': lag_2w,
            'Needed_Qty_Lag_4W': lag_4w,
            'Avg_Needed_Qty_Last_4W': avg_4w
        })
    lag_features_df = pd.DataFrame(lag_features_list)
    if lag_features_df.empty:
        st.warning("Warning: No lag features could be generated. Check history data and SKU matching.")
        lag_features_df = pd.DataFrame(columns=[SKU_COLUMN] + numeric_cols_with_lags)


    # Prepare template with SKU info
    sku_df_template = products_info[required_product_cols].copy()

    # Cross join future ops data (single row from form) with all SKUs
    prediction_features = pd.merge(sku_df_template, prediction_features, how='cross')
    prediction_features = pd.merge(prediction_features, lag_features_df, on=SKU_COLUMN, how='left')

    prediction_features[numeric_cols_with_lags] = prediction_features[numeric_cols_with_lags].fillna(0)
    prediction_features['Avg_Unit_Cost'] = prediction_features['Avg_Unit_Cost_Base']

    if 'Avg_Unit_Cost_Base' not in training_cols:
        prediction_features = prediction_features.drop(columns=['Avg_Unit_Cost_Base'], errors='ignore')

    # One-Hot Encode Categorical Features
    existing_categorical_pred = [col for col in categorical_features_list if col in prediction_features.columns]
    if existing_categorical_pred:
        try:
            prediction_features = pd.get_dummies(prediction_features, columns=existing_categorical_pred, drop_first=False, dummy_na=False)
        except Exception as e:
            st.error(f"Error during one-hot encoding: {e}")
            return pd.DataFrame()
    else:
        st.warning("Warning: No specified categorical features found in the combined prediction data.")

    # Store Identifiers BEFORE aligning
    if DATE_COLUMN not in prediction_features.columns or SKU_COLUMN not in prediction_features.columns:
        st.error(f"CRITICAL Error: Identifiers '{DATE_COLUMN}' or '{SKU_COLUMN}' missing before alignment.")
        return pd.DataFrame()
    try:
        identifiers = prediction_features[[DATE_COLUMN, SKU_COLUMN]].copy()
    except KeyError:
        st.error("Failed to extract identifiers. Check column names.")
        return pd.DataFrame()

    # Align columns with the training set
    st.write(f"Aligning prediction columns with {len(training_cols)} training columns...")
    X_pred = pd.DataFrame(columns=training_cols, index=prediction_features.index)
    present_cols = prediction_features.columns

    cols_found = []
    cols_missing = []

    for col in training_cols:
        if col in present_cols:
            X_pred[col] = prediction_features[col]
            cols_found.append(col)
        else:
            X_pred[col] = 0
            cols_missing.append(col)

    if cols_missing:
        st.warning(f"Warning: The following training columns were not found in the generated features and were filled with 0: {cols_missing}")

    try:
         X_pred = X_pred.astype(float)
    except Exception as e:
         st.error(f"Error converting aligned features to numeric types: {e}")
         st.dataframe(X_pred.dtypes)
         return pd.DataFrame()

    # Make Predictions
    st.write(f"Predicting demand for {len(X_pred)} SKU-Week combinations...")
    try:
        predicted_needs = model.predict(X_pred)
        predicted_needs[predicted_needs < 0] = 0
    except Exception as e:
        st.error(f"Error during model prediction: {e}")
        st.error("Input data shape for prediction:", X_pred.shape)
        st.error("Input data columns for prediction:", X_pred.columns.tolist())
        st.dataframe(X_pred.head())
        return pd.DataFrame()

    # Prepare Output DataFrame
    output_df = identifiers.copy()
    output_df['Predicted_Need'] = predicted_needs

    try:
        output_df = pd.merge(output_df, current_stock[[SKU_COLUMN, 'Current_Stock_Level']], on=SKU_COLUMN, how='left')
        output_df['Current_Stock_Level'] = output_df['Current_Stock_Level'].fillna(0)
    except Exception as e:
        st.error(f"Error merging with current stock: {e}")
        return pd.DataFrame()

    try:
        if SKU_COLUMN not in products_info.columns:
            st.error(f"Error: '{SKU_COLUMN}' missing from products_info for MOQ merge.")
            return pd.DataFrame()
        output_df = pd.merge(output_df, products_info[[SKU_COLUMN, 'Min_Order_Quantity']], on=SKU_COLUMN, how='left')
        output_df['Min_Order_Quantity'] = output_df['Min_Order_Quantity'].fillna(1)
        output_df['Min_Order_Quantity'] = pd.to_numeric(output_df['Min_Order_Quantity'], errors='coerce').fillna(1)
    except Exception as e:
        st.error(f"Error merging with products info for MOQ: {e}")
        return pd.DataFrame()

    output_df['Calculated_Need_Vs_Stock'] = output_df['Predicted_Need'] - output_df['Current_Stock_Level']

    def calculate_suggested(row):
        need_vs_stock = row['Calculated_Need_Vs_Stock']
        min_order = row['Min_Order_Quantity']
        if need_vs_stock <= 0:
            return 0
        else:
            suggested = np.ceil(need_vs_stock)
            return max(suggested, min_order) if suggested > 0 else 0

    output_df['Suggested_Order_Qty'] = output_df.apply(calculate_suggested, axis=1)

    st.write("Prediction and suggestion calculation complete.")
    # Return only the specified columns
    final_cols = [DATE_COLUMN, SKU_COLUMN, 'Predicted_Need', 'Current_Stock_Level', 'Calculated_Need_Vs_Stock', 'Min_Order_Quantity', 'Suggested_Order_Qty']
    # Add Item Name and Unit from products_info for better readability
    try:
        output_df = pd.merge(output_df, products_info[[SKU_COLUMN, 'Item_Name', 'Unit_of_Measure']], on=SKU_COLUMN, how='left')
        # Ensure 'Item_Name' is handled if missing
        if 'Item_Name' in output_df.columns:
            final_cols = [DATE_COLUMN, SKU_COLUMN, 'Item_Name', 'Predicted_Need', 'Current_Stock_Level', 'Calculated_Need_Vs_Stock', 'Min_Order_Quantity', 'Suggested_Order_Qty', 'Unit_of_Measure']
        else:
             st.warning("Column 'Item_Name' not found in products_info, skipping.")
    except Exception as e:
        st.warning(f"Could not merge Item_Name/Unit_of_Measure for output: {e}")


    return output_df[final_cols] # Select and reorder columns for final output


# --- Wrapper Function (Handles File Reading and Orchestration) ---
def predict_demand_and_orders(
    future_ops_data_df, # Expecting DataFrame created from form inputs
    latest_demand_hist_csv, # Expecting UploadedFile object
    products_info_csv,      # Expecting UploadedFile object
    current_stock_csv       # Expecting UploadedFile object
) :
    """
    Reads files, prepares data, calls prediction function, and formats output.
    """
    st.write("--- Reading Uploaded Files ---")
    try:
        latest_demand_hist_df = pd.read_csv(latest_demand_hist_csv)
        products_info_df = pd.read_csv(products_info_csv)
        current_stock_df = pd.read_csv(current_stock_csv)
        st.write("Files read successfully.")
    except Exception as e:
        st.error(f"Error reading one or more CSV files: {e}")
        return None

    if future_ops_data_df is None or not isinstance(future_ops_data_df, pd.DataFrame) or future_ops_data_df.empty:
        st.error("Operational parameters data is invalid or missing.")
        return None

    # Call predict_and_suggest_orders()
    order_suggestions_df = predict_and_suggest_orders(
        model=model,
        future_ops_data=future_ops_data_df, # Pass the DataFrame created from form inputs
        latest_demand_hist=latest_demand_hist_df,
        products_info=products_info_df,
        current_stock=current_stock_df,
        categorical_features_list=CATEGORICAL_FEATURES,
        training_cols=TRAINING_COLS
    )

    if order_suggestions_df is None or not isinstance(order_suggestions_df, pd.DataFrame):
        st.error("Order suggestion calculation failed.")
        return None

    return order_suggestions_df


# --- Streamlit UI ---
st.title("📦 Demand Forecast & Order Suggestion")
st.markdown("""
Fill in the operational parameters for the *single future week* you want to predict, and upload the required CSV files.
""")

# Use columns for better layout of parameters
col1, col2 = st.columns(2)

with st.form("prediction_form"):
    st.subheader("Operational Parameters for Target Week")

    with col1:
        # Date and Occupancy related inputs
        input_week_start_date = st.date_input(
            "Week Start Date",
            value=datetime.date.today() + datetime.timedelta(days=(7 - datetime.date.today().weekday()) % 7), # Default to next Sunday
            help="Select the Sunday marking the beginning of the forecast week."
        )
        input_forecasted_occupancy = st.number_input("Forecasted Occupancy (%)", min_value=0, max_value=100, value=75, step=1)
        input_forecasted_events = st.number_input("Forecasted Event Guests", min_value=0, value=180, step=10)
        input_week_of_year = st.number_input("Week of Year", min_value=1, max_value=53, value=datetime.date.today().isocalendar()[1], step=1)
        input_month = st.number_input("Month", min_value=1, max_value=12, value=datetime.date.today().month, step=1)

        st.markdown("---") # Separator
        # Seasonal/Event Flags
        st.write("*Flags (Select if applicable):*")
        input_is_peak = st.checkbox("Is Peak Season", value=False)
        input_is_low = st.checkbox("Is Low Season", value=False)
        input_is_ramadan = st.checkbox("Is Ramadan Week", value=False)
        input_is_eid = st.checkbox("Is Eid Week", value=False)
        input_is_holiday = st.checkbox("Is Public Holiday Week", value=False)

    with col2:
        # Guest Mix Percentages
        st.write("*Guest Mix Forecast (%):*")
        input_pct_gcc = st.number_input("% Guests GCC", min_value=0, max_value=100, value=30, step=1)
        input_pct_europe = st.number_input("% Guests Europe", min_value=0, max_value=100, value=40, step=1)
        input_pct_asia = st.number_input("% Guests Asia", min_value=0, max_value=100, value=15, step=1)
        input_pct_americas = st.number_input("% Guests Americas", min_value=0, max_value=100, value=5, step=1)
        input_pct_africa_me = st.number_input("% Guests Africa/ME", min_value=0, max_value=100, value=5, step=1)
        input_pct_other = st.number_input("% Guests Other Region", min_value=0, max_value=100, value=5, step=1)

        st.markdown("---") # Separator
        st.write("*Guest Segment Forecast (%):*")
        input_pct_leisure = st.number_input("% Guests Leisure", min_value=0, max_value=100, value=70, step=1)
        input_pct_business = st.number_input("% Guests Business", min_value=0, max_value=100, value=20, step=1)
        input_pct_group = st.number_input("% Guests Group", min_value=0, max_value=100, value=10, step=1)

        st.markdown("---") # Separator
        st.write("*Booking Type Forecast (%):*")
        input_pct_families = st.number_input("% Families", min_value=0, max_value=100, value=25, step=1)
        input_pct_adult_only = st.number_input("% Adult-Only Bookings", min_value=0, max_value=100, value=15, step=1)

    st.subheader("File Uploads")
    # File uploaders remain the same
    products_file = st.file_uploader("Upload Products Info CSV", type="csv", key="p")
    history_file = st.file_uploader("Upload Latest Product History CSV", type="csv", key="l")
    stock_file = st.file_uploader("Upload Current Stock CSV", type="csv", key='c')

    # Submit button
    submitted = st.form_submit_button("Calculate Suggestions")

# --- Post-Submission Logic ---
if submitted:
    # Basic validation: Check if all files are provided
    if products_file is None:
        st.warning("Please upload the Products Info CSV file.")
    elif history_file is None:
        st.warning("Please upload the Latest Product History CSV file.")
    elif stock_file is None:
        st.warning("Please upload the Current Stock CSV file.")
    else:
        st.info("Processing inputs...")

        # --- Create future_ops_data DataFrame from form inputs ---
        ops_data = {
            DATE_COLUMN: [pd.to_datetime(input_week_start_date)], # Ensure datetime type
            'Forecasted_Occupancy_Percent': [input_forecasted_occupancy],
            'Forecasted_Event_Guests': [input_forecasted_events],
            'Week_of_Year': [input_week_of_year],
            'Month': [input_month],
            # Convert checkbox True/False to 1/0
            'Is_Peak_Season': [1 if input_is_peak else 0],
            'Is_Low_Season': [1 if input_is_low else 0],
            'Is_Ramadan_Week': [1 if input_is_ramadan else 0],
            'Is_Eid_Week': [1 if input_is_eid else 0],
            'Is_Public_Holiday_Week': [1 if input_is_holiday else 0],
            'Percent_Guests_GCC': [input_pct_gcc],
            'Percent_Guests_Europe': [input_pct_europe],
            'Percent_Guests_Asia': [input_pct_asia],
            'Percent_Guests_Americas': [input_pct_americas],
            'Percent_Guests_Africa_ME': [input_pct_africa_me],
            'Percent_Guests_Other_Region': [input_pct_other],
            'Percent_Guests_Leisure': [input_pct_leisure],
            'Percent_Guests_Business': [input_pct_business],
            'Percent_Guests_Group': [input_pct_group],
            'Percent_Families': [input_pct_families],
            'Percent_Adult_Only_Bookings': [input_pct_adult_only]
        }
        try:
            ops_df = pd.DataFrame(ops_data)
            st.write("Operational parameters DataFrame created:")
            st.dataframe(ops_df.head()) # Display the created ops dataframe for verification

        except Exception as e:
             st.error(f"Failed to create Operational Parameters DataFrame: {e}")
             ops_df = None # Set to None to prevent further processing

        # Proceed only if DataFrame creation was successful
        if ops_df is not None:
            # Display spinner during calculation
            with st.spinner('Running prediction and generating suggestions... Please wait.'):
                try:
                    # Call the main prediction function
                    results_df = predict_demand_and_orders(
                        ops_df,          # Pass the DataFrame created from inputs
                        history_file,    # Pass the file object
                        products_file,   # Pass the file object
                        stock_file       # Pass the file object
                    )

                    if results_df is not None and not results_df.empty:
                        st.success("Calculation complete!")
                        st.subheader("Suggested Orders")
                        st.dataframe(results_df) # Display results as an interactive table

                        # Provide download link for the results
                        csv = results_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                           label="Download Results as CSV",
                           data=csv,
                           file_name='suggested_orders.csv',
                           mime='text/csv',
                        )

                    elif results_df is not None and results_df.empty:
                        st.warning("Calculation finished, but no suggestions were generated. This might happen if all needs are met by current stock or due to data issues.")
                    else:
                        st.error("Calculation failed. Please review the error messages above.")

                except Exception as e:
                    st.error("An unexpected error occurred during the main calculation process.")
                    st.exception(e)

st.markdown("---")
st.caption("Ensure CSV files have the correct columns.")