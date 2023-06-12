
import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox, IntVar, ttk
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
from sklearn.feature_selection import RFE, SelectKBest, f_regression, mutual_info_regression
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


allowed_features =  [
"Ad Group Budget",
"Cost",
"Impression",
"Click",
"CTR",
"Conversions",
"CPA",
"CVR",
"Results",
"Results Rate",
"Frequency",
"Video Views at 25%",
"Video Views at 50%",
"Video Views at 75%",
"Video Views at 100%",
"6-Second Video Views",
"Total Add Payment Info",
"Total Initiate Checkout",
"Add to Cart",
"Total View Content",
"Total Page View",
"CTA Conversions",
"CTA Purchase",
"CTA Registration",
"VTA Conversions",
"VTA Purchase",
"VTA Registration",
"Clicks",
"Impressions",
"Paid Likes",
"Reach",
"Real-time Conversions",
"Total Engagement",
"Total Achieve Level",
"Total Add to Wishlist",
"Total Checkout",
"Total Complete Tutorial",
"Total Create Group",
"Total Create Role",
"Total Generate Lead",
"Total In-App Ad Click",
"Total In-App Ad Impression",
"Total Join Group",
"Total Launch App",
"Total Loan Apply",
"Total Loan Approval",
"Total Loan Disbursal",
"Total Login",
"Total Purchase",
"Total Rate",
"Total Registration",
"Total Search",
"Total Spend Credit",
"Total Start Trial",
"Total Subscribe",
"Total Unlock Achievement",
"Unique Achieve Level",
"Unique Add Payment Info",
"Unique Add to Cart",
"Unique Add to Wishlist",
"Unique Checkout",
"Unique Complete Tutorial",
"Unique Create Group",
"Unique Create Role",
"Unique Generate Lead",
"Unique In-App Ad Click",
"Unique In-App Ad Impression",
"Unique Join Group",
"Unique Launch App",
"Unique Loan Apply",
"Unique Loan Approval",
"Unique Loan Disbursal",
"Unique Login",
"Unique Purchase",
"Unique Rate",
"Unique Registration",
"Unique Search",
"Unique Spend Credit",
"Unique Start Trial",
"Unique Subscribe",
"Unique Unlock Achievement",
"Unique View Content",
"Total Add Billing",
"Total Button Click (App Download)",
"Total Button Click (Form)",
"Total Button Click (Phone Consultation)",
"Total Complete Payment",
"Total Details Page View (App Download)",
"Total Details Page View (Form)",
"Total Details Page View (Phone Consultation)",
"Total Form Submission",
"Total Online Consultation",
"Total Phone Consultation",
"Total Place an Order",
"Total Product Details Page View",
"Total User Registration",
"2-Second Video Views",
"Video Views",
"Complete Payment (Onsite)",
"Initiate Checkout (Onsite)",
"Product Details Page View (Onsite)",
"Add to Wishlist (Onsite)",
"Add Billing (Onsite)",
"Form Submission (Onsite)",
"App Store Click (Onsite)",
"Page Views (Onsite)",
"Call-to-Action Button Clicks (Onsite)",
"Product Clicks (Onsite)"
]

# Function to browse and select a CSV file
def browse_file():
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if file_path:
        process_csv(file_path, heatmaps_label)

# Function to process the selected CSV file
def process_csv(file_path, heatmaps_label):
    df = pd.read_csv(file_path)

    # Get all the feature names
    feature_names = df.columns.tolist()

    # Filter the feature names to include only the allowed features
    selected_features = [feature for feature in feature_names if feature in allowed_features]

    # Filter the DataFrame to include only the selected features
    df = df[selected_features]

    X = df.iloc[:, 1:]  # Features (all columns except the first one)
    target_variable_value = target_var_choice.get()

    if target_variable_value == 1:
        target_variable = "Results"
    elif target_variable_value == 2:
        target_variable = "Impression"
    elif target_variable_value == 3:
        target_variable = "Click"

    y = df[target_variable]  # Target variable

    # Calculate correlation between features and target variable
    corr = X.corrwith(y)
    corr_matrix = pd.DataFrame(corr, columns=['Results']).sort_values(by='Results', ascending=False)

    # Feature selection using Recursive Feature Elimination (RFE)
    rfe = RFE(estimator=LinearRegression(), n_features_to_select=5)
    X_rfe = rfe.fit_transform(X, y)
    selected_features_rfe = X.columns[rfe.support_]
    corr_matrix_rfe = X[selected_features_rfe].corrwith(y)
    corr_matrix_rfe = pd.DataFrame(corr_matrix_rfe, columns=['Results']).sort_values(by='Results', ascending=False)

    # Standardize the features using StandardScaler
    scaler = StandardScaler()
    X_standardized = scaler.fit_transform(X)

    # Feature selection using SelectKBest
    kbest = SelectKBest(score_func=f_regression, k=5)
    X_kbest = kbest.fit_transform(X_standardized, y)
    selected_features_kbest = X.columns[kbest.get_support()]
    corr_matrix_kbest = X[selected_features_kbest].corrwith(y)
    corr_matrix_kbest = pd.DataFrame(corr_matrix_kbest, columns=['Results']).sort_values(by='Results', ascending=False)

    # Fit a linear regression model
    model = LinearRegression()
    model.fit(X, y)

    # Calculate permutation importance
    perm_importance = permutation_importance(model, X, y, n_repeats=10, random_state=0)
    perm_importance_mean = pd.DataFrame(perm_importance.importances_mean, index=X.columns, columns=['Importance']).sort_values(by='Importance', ascending=False)

    # Calculate mutual information regression
    mutual_info = mutual_info_regression(X, y)
    mutual_info_series = pd.Series(mutual_info, index=X.columns).sort_values(ascending=False).head(6)

    # Clear the text box and display results
    text_box.delete(1.0, tk.END)
    text_box.insert(tk.END, "Correlation matrix:\n")
    text_box.insert(tk.END, corr_matrix.to_string())
    text_box.insert(tk.END, "\n\n")
    text_box.insert(tk.END, "Correlation matrix (RFE):\n")
    text_box.insert(tk.END, corr_matrix_rfe.to_string())
    text_box.insert(tk.END, "\n\n")
    text_box.insert(tk.END, "Correlation matrix (SelectKBest):\n")
    text_box.insert(tk.END, corr_matrix_kbest.to_string())
    text_box.insert(tk.END, "\n\n")
    text_box.insert(tk.END, "Permutation Importance:\n")
    text_box.insert(tk.END, perm_importance_mean.head(6).to_string())
    text_box.insert(tk.END, "\n\n")
    text_box.insert(tk.END, "Mutual Info Regression (Top 6):\n")
    text_box.insert(tk.END, mutual_info_series.to_string())
    text_box.insert(tk.END, "\n")

    # Clear and display the feature names in the text box
    text_box.delete(1.0, tk.END)
    text_box.insert(tk.END, "Features:\n")
    text_box.insert(tk.END, ", ".join(feature_names))
    text_box.insert(tk.END, "\n\n")

    # Combine the selected features from different methods
    overall_top_features = pd.Series(selected_features_rfe).value_counts().sort_values(ascending=False).index[:6]

    # Display the overall top features in the text box
    text_box_top_features.delete(1.0, tk.END)
    text_box_top_features.insert(tk.END, "Overall Top Features:\n")
    text_box_top_features.insert(tk.END, ", ".join(overall_top_features))
    text_box_top_features.insert(tk.END, "\n\n")

    # Create subplots for heatmaps
    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(10, 15), gridspec_kw={'wspace': 0.4, 'hspace': 0.4})

    # Plot correlation matrix heatmap
    sns.heatmap(corr_matrix, cmap='RdYlBu', annot=True, ax=ax1)
    ax1.set_title("Correlation Matrix")
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
    ax1.set_yticklabels(ax1.get_yticklabels(), rotation=0, ha='right')  

    # Plot RFE heatmap
    sns.heatmap(corr_matrix_rfe, cmap='RdYlBu', annot=True, ax=ax2)
    ax2.set_title("RFE")
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
    ax2.set_yticklabels(ax2.get_yticklabels(), rotation=0, ha='right')

    # Plot SelectKBest heatmap
    sns.heatmap(corr_matrix_kbest, cmap='RdYlBu', annot=True, ax=ax3)
    ax3.set_title("SelectKBest")
    ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45, ha='right')
    ax3.set_yticklabels(ax3.get_yticklabels(), rotation=0, ha='right')

    # Plot permutation importance heatmap
    sns.heatmap(perm_importance_mean.head(6).transpose(), cmap='RdYlBu', annot=True, ax=ax4)
    ax4.set_title("Permutation Importance")
    ax4.set_xticklabels(ax4.get_xticklabels(), rotation=45, ha='right')
    ax4.set_yticklabels(ax4.get_yticklabels(), rotation=0, ha='right')

    # Plot mutual info regression heatmap
    corr_matrix_mutual_info = pd.DataFrame(mutual_info_series, columns=['Results'])
    sns.heatmap(corr_matrix_mutual_info.transpose(), cmap='RdYlBu', annot=True, ax=ax5)
    ax5.set_title("Mutual Info Regression (Top 6)")
    ax5.set_xticklabels(ax5.get_xticklabels(), rotation=45, ha='right')
    ax5.set_yticklabels(['Results'], rotation=0, ha='right')

    # Disable ax6
    ax6.axis('off')

    # Save the figure as heatmaps.png
    plt.savefig('heatmaps.png')
    plt.clf()

    # Open and display the heatmaps image
    heatmaps_image = Image.open('heatmaps.png')
    heatmaps_image.thumbnail((800, 1000))
    heatmaps_photo = ImageTk.PhotoImage(heatmaps_image)
    heatmaps_label.configure(image=heatmaps_photo)
    heatmaps_label.image = heatmaps_photo


def save_heatmap():
    # Prompt the user to choose a save path
    save_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")])
    if save_path:
        # Save the current figure as the chosen file path
        plt.savefig(save_path)
        plt.clf()
        # Show a message box indicating that the heatmap has been saved
        messagebox.showinfo("Saved", f"Heatmap saved as {save_path}")


# Create the main Tkinter window
root = tk.Tk()
# Set the window icon
icon_path = "Logo.ico"
root.iconbitmap(icon_path)
# Set the window title
root.title("StratFlow - TikTok Ads Analyzer")
# Set the background color of the window
root.configure(background="#f2f2f2")

# Create the left frame for input and output elements
left_frame = tk.Frame(root, bg="#f2f2f2")
left_frame.pack(side=tk.LEFT, padx=10, pady=10)

# Create the right frame for displaying heatmaps
right_frame = tk.Frame(root, bg="#f2f2f2")
right_frame.pack(side=tk.LEFT, padx=10, pady=10)

# Create a label for the radio buttons
radio_label = tk.Label(left_frame, text="Select Target Objective:", font=("Arial", 12))
radio_label.pack()

# Create an IntVar to hold the selected target variable choice
target_var_choice = IntVar()

# Function to update the target variable choice
def update_target_variable():
    target_variable = target_var_choice.get()
    if target_variable == 1:
        # Target variable is "Results"
        target_variable_label.configure(text="Target Variable: Results")
    elif target_variable == 2:
        # Target variable is "Impression"
        target_variable_label.configure(text="Target Variable: Impression")
    elif target_variable == 3:
        # Target variable is "Click"
        target_variable_label.configure(text="Target Variable: Clicks")

# Create the radio buttons for target variable choice
radio_button1 = tk.Radiobutton(
    left_frame,
    text="Conversion",
    variable=target_var_choice,
    value=1,
    command=update_target_variable,
    font=("Arial", 12)
)
radio_button1.pack()

radio_button2 = tk.Radiobutton(
    left_frame,
    text="Awareness",
    variable=target_var_choice,
    value=2,
    command=update_target_variable,
    font=("Arial", 12)
)
radio_button2.pack()

radio_button3 = tk.Radiobutton(
    left_frame,
    text="Consideration",
    variable=target_var_choice,
    value=3,
    command=update_target_variable,
    font=("Arial", 12)
)
radio_button3.pack()

# Create a label to display the selected target variable
target_variable_label = tk.Label(left_frame, text="Target Variable: None", font=("Arial", 12))
target_variable_label.pack()


# Create the 'Browse CSV File' button
browse_button = tk.Button(
    left_frame,
    text="Browse CSV File",
    command=browse_file,
    bg="#2A2A2A",
    fg="white",
    font=("Arial", 12)
)
browse_button.pack(pady=10)

# Create a label for the text box
text_label = tk.Label(left_frame, text="Features being analyzed", font=("Arial", 12))
text_label.pack()

# Create the text box to display the analysis results
text_box = tk.Text(
    left_frame,
    width=40,
    height=10,
    bg="white",
    fg="black",
    font=("Arial", 12)
)
text_box.pack()

# Create a label for the top features text box
text_label = tk.Label(left_frame, text="Top features selection", font=("Arial", 12))
text_label.pack()

# Create the text box to display the top selected features
text_box_top_features = tk.Text(
    left_frame,
    width=40,
    height=4,
    bg="white",
    fg="black",
    font=("Arial", 12)
)
text_box_top_features.pack()

# Create the 'Save Heatmap' button
save_button = tk.Button(
    left_frame,
    text="Save Heatmap",
    command=save_heatmap,
    bg="#2A2A2A",
    fg="white",
    font=("Arial", 12)
)
save_button.pack(pady=10)

# Create the label to display the heatmaps image
heatmaps_label = tk.Label(right_frame, bg="#f2f2f2")
heatmaps_label.pack()

# Start the Tkinter event loop
root.mainloop()
