import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# File paths
orders = r"C:\Users\Radhika Dhoot\OneDrive\Desktop\project1\List of Orders.csv"
order_details = r"C:\Users\Radhika Dhoot\OneDrive\Desktop\project1\Order Details.csv"
sales_target = r"C:\Users\Radhika Dhoot\OneDrive\Desktop\project1\Sales target.csv"

# Read the CSV files (without headers)
df1 = pd.read_csv(orders, header=None)
df2 = pd.read_csv(order_details, header=None)
df3 = pd.read_csv(sales_target, header=None)

print("Orders:", df1.shape)
print("Order Details:", df2.shape)
print("Sales Target:", df3.shape)

print("Orders Columns:", df1.columns.tolist())
print("Order Details Columns:", df2.columns.tolist())


#  Merge on the first column (column index 0), which is 'Order ID'
sales_data = pd.merge(df2, df1, on=0)

# Display merged data
print("\nMerged Data Shape:", sales_data.shape)
print(sales_data.head())

#  Display first few rows to inspect which column is Order ID
print("\nFirst few rows of Orders:")
print(df1.head())

print("\nFirst few rows of Order Details:")
print(df2.head())

# The first row contains the column names
sales_data.columns = sales_data.iloc[0]   # Set the first row as header
sales_data = sales_data[1:]               # Drop the first row

# Reset index
sales_data.reset_index(drop=True, inplace=True)

#  Convert 'Order Date' to datetime
sales_data["Order Date"] = pd.to_datetime(sales_data["Order Date"], format='%d-%m-%Y', errors='coerce')

# Check the result
print("\nColumns after cleaning:", sales_data.columns.tolist())
print(sales_data.head())

# 1️ Check if 'Order Date' and 'Amount' exist and are correct
print("\nColumn Names:", sales_data.columns.tolist())

# 2️ Check a few sample rows
print("\nSample rows from sales_data:")
print(sales_data.head(3))

# 3️ Check data types
print("\nData Types:")
print(sales_data.dtypes)

# Convert Amount column to numeric (handle invalid values safely)
sales_data["Amount"] = pd.to_numeric(sales_data["Amount"], errors="coerce")

# Drop rows where Amount or Order Date is missing
sales_data = sales_data.dropna(subset=["Amount", "Order Date"])

# Extract Year-Month period
sales_data["YearMonth"] = sales_data["Order Date"].dt.to_period("M")

#  Calculate monthly total sales
monthly_sales = (
    sales_data.groupby("YearMonth")["Amount"]
    .sum()
    .reset_index()
    .rename(columns={"Amount": "ActualSales"})
)

print("\n Monthly Sales:\n", monthly_sales.head())

print("df3 shape:", df3.shape)
print("\ndf3 first few rows:")
print(df3.head())

# Remove the header row and assign proper column names
df3.columns = df3.iloc[0]   # First row as header
df3 = df3[1:]               # Drop the first row

# Reset index
df3.reset_index(drop=True, inplace=True)

# Convert Target to numeric
df3["Target"] = pd.to_numeric(df3["Target"], errors="coerce")

# Convert Month column to YearMonth period
df3["YearMonth"] = pd.to_datetime(
    df3["Month of Order Date"], format='%b-%y', errors='coerce'
).dt.to_period("M")

print("\n Cleaned Sales Target DataFrame:")
print(df3.head())

# Make sure to clean column names in df3
df3.columns = df3.columns.str.strip()

# Merge on YearMonth
comparison_df = pd.merge(monthly_sales, df3, on="YearMonth", how="inner")

print("\n Combined Data (Actual vs Target):")
print(comparison_df.head())

#########################

# Convert columns to numeric (in case any non-numeric values remain)
comparison_df["ActualSales"] = pd.to_numeric(comparison_df["ActualSales"], errors="coerce")
comparison_df["Target"] = pd.to_numeric(comparison_df["Target"], errors="coerce")

# Drop any missing values in these columns
comparison_df = comparison_df.dropna(subset=["ActualSales", "Target"])

# Double-check datatypes
print(comparison_df.dtypes)

#####################################

comparison_df["YearMonth"] = comparison_df["YearMonth"].astype(str)

plt.figure(figsize=(10,6))
sns.lineplot(data=comparison_df, x="YearMonth", y="ActualSales", marker="o", label="Actual Sales")
sns.lineplot(data=comparison_df, x="YearMonth", y="Target", marker="s", label="Target Sales", color="orange")

plt.title(" Monthly Actual Sales vs Target Sales (Overall)", fontsize=14)
plt.xlabel("Year-Month")
plt.ylabel("Sales Amount")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

##################################################

# Top 5 products by total sales
top_products = sales_data.groupby("Sub-Category")["Amount"].sum().sort_values(ascending=False).head(5)
print("\nTop 5 Products by Sales:\n", top_products)

# Optional: visualize top 5 products
plt.figure(figsize=(8,5))
top_products.plot(kind="bar", color="teal")
plt.title("Top 5 Products by Total Sales", fontsize=14)
plt.xlabel("Product Sub-Category")
plt.ylabel("Total Sales Amount")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Save combined report to CSV
comparison_df.to_csv("Monthly_Sales_Report.csv", index=False)
print("\n Monthly Sales Report saved as 'Monthly_Sales_Report.csv'")

##########################################################

# Find the best and worst performing months based on Actual Sales
best_month = comparison_df.loc[comparison_df["ActualSales"].idxmax()]
worst_month = comparison_df.loc[comparison_df["ActualSales"].idxmin()]

print(" Best Month:", best_month["YearMonth"], "with Sales:", best_month["ActualSales"])
print(" Worst Month:", worst_month["YearMonth"], "with Sales:", worst_month["ActualSales"])

# --- Visualization ---
plt.figure(figsize=(10,6))
sns.lineplot(data=comparison_df, x="YearMonth", y="ActualSales", marker="o", label="Actual Sales")
sns.lineplot(data=comparison_df, x="YearMonth", y="Target", marker="s", label="Target Sales", color="orange")

# Highlight Best and Worst points
plt.scatter(best_month["YearMonth"], best_month["ActualSales"], color="green", s=100, label="Best Month")
plt.scatter(worst_month["YearMonth"], worst_month["ActualSales"], color="red", s=100, label="Worst Month")

# Annotate the points
plt.text(best_month["YearMonth"], best_month["ActualSales"], f'  {best_month["YearMonth"]}\n  {best_month["ActualSales"]}', 
         color='green', fontsize=10, weight='bold')
plt.text(worst_month["YearMonth"], worst_month["ActualSales"], f'  {worst_month["YearMonth"]}\n  {worst_month["ActualSales"]}', 
         color='red', fontsize=10, weight='bold')

plt.title(" Monthly Actual Sales vs Target Sales with Best & Worst Months", fontsize=14)
plt.xlabel("Year-Month")
plt.ylabel("Sales Amount")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Calculate Achievement Percentage
comparison_df["Achievement (%)"] = (comparison_df["ActualSales"] / comparison_df["Target"]) * 100


plt.figure(figsize=(10,6))
sns.barplot(x=comparison_df["YearMonth"].astype(str), y=comparison_df["Achievement (%)"], palette="coolwarm")
plt.axhline(100, color='green', linestyle='--', label="Target Line")
plt.title(" Target Achievement Percentage by Month")
plt.xlabel("Month")
plt.ylabel("Achievement (%)")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

########################################

import matplotlib.pyplot as plt

# Group by Sub-Category and sum sales amount
top_products = sales_data.groupby("Sub-Category")["Amount"].sum().sort_values(ascending=False).head(5)

# Plot the top 5 products
top_products.plot(kind='bar', color='skyblue', figsize=(8,5))
plt.title("Top 5 Products by Sales")
plt.xlabel("Product Sub-Category")
plt.ylabel("Total Sales Amount")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Correlation Heatmap between Amount, Quantity, and Profit
plt.figure(figsize=(6, 4))
corr = sales_data[["Amount", "Quantity", "Profit"]].corr()

sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5,
    cbar_kws={'shrink': 0.8}
)

plt.title(" Correlation Heatmap: Amount, Quantity & Profit", fontsize=13, weight='bold')
plt.tight_layout()
plt.show()

avg_sales = comparison_df["ActualSales"].mean()
target_hit_months = (comparison_df["Achievement (%)"] >= 100).sum()
below_target_months = len(comparison_df) - target_hit_months

print("Sales Performance Summary")
print("-" * 35)
print(f"Average Monthly Sales: ₹{avg_sales:,.2f}")
print(f" Months Meeting/Exceeding Target: {target_hit_months}")
print(f" Months Below Target: {below_target_months}")

#############################

import matplotlib.pyplot as plt
import seaborn as sns

# Top 5 Products by Sales
top_products = sales_data.groupby("Sub-Category")["Amount"].sum().sort_values(ascending=False).head(5)

# Category-wise Sales
category_sales = sales_data.groupby("Category")["Amount"].sum().sort_values(ascending=False)

# Create subplots (1 row, 2 columns)
plt.figure(figsize=(14,6))

# --- Left: Top 5 Products ---
plt.subplot(1, 2, 1)
top_products.plot(kind='bar', color=sns.color_palette("Blues", 5))
plt.title("🏆 Top 5 Products by Sales", fontsize=13, weight='bold')
plt.xlabel("Product Sub-Category")
plt.ylabel("Total Sales (₹)")
plt.grid(axis='y', linestyle='--', alpha=0.7)

# --- Right: Category-wise Pie Chart ---
plt.subplot(1, 2, 2)
explode = [0.05] * len(category_sales)
category_sales.plot(
    kind='pie',
    autopct='%1.1f%%',
    startangle=140,
    colors=sns.color_palette("pastel"),
    explode=explode,
    shadow=True
)
plt.title(" Category-wise Sales Contribution", fontsize=13, weight='bold')
plt.ylabel("")  # remove y-label for a clean look

plt.tight_layout()
plt.show()

###############################

from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

# Ensure 'comparison_df' is used (as per your earlier variable)
comparison_df["Month_Num"] = range(1, len(comparison_df) + 1)

# Prepare data
X = comparison_df[["Month_Num"]]
y = comparison_df["ActualSales"]

# Train model
model = LinearRegression()
model.fit(X, y)

# Predict next month’s sales
next_month = np.array([[len(comparison_df) + 1]])
predicted_sales = model.predict(next_month)

print(f" Predicted Sales for Next Month: ₹{predicted_sales[0]:,.2f}")

# Plot actual vs predicted sales trend
plt.figure(figsize=(10,6))
plt.plot(comparison_df["Month_Num"], y, marker='o', label="Actual Sales", color='blue')
plt.plot(comparison_df["Month_Num"], model.predict(X), color='orange', linestyle='--', label="Trend Line")
plt.scatter(next_month, predicted_sales, color='red', s=100, label='Next Month Prediction')
plt.title(" Sales Trend & Next Month Prediction", fontsize=14, weight='bold')
plt.xlabel("Month Number")
plt.ylabel("Sales Amount (₹)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()





