

# %%
# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# %% [markdown]
# Import các thư viện cần thiết

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

# Đọc dữ liệu từ tệp .csv trong thư mục hiện tại
file_path = 'vietnam_housing_dataset.csv'

# Tải dữ liệu
data = pd.read_csv(file_path)

# Xem thông tin dữ liệu
data.head()


# # Tải dữ liệu
# data = pd.read_csv(file_path)
# data = pd.read_csv(file_path)
data.head()
data.info()
data.isnull().sum()
avg_price = data['Price'].mean()
print(avg_price)


# %%
# Data cleaning
# Check for missing values
print("Missing values count before cleaning:")
print(data.isnull().sum())

# Fill missing numeric values with the median
numeric_columns = data.select_dtypes(include=[np.number]).columns
for col in numeric_columns:
    data[col].fillna(data[col].median(), inplace=True)

# Fill missing categorical values with the mode for other columns
categorical_columns = data.select_dtypes(include=[object]).columns
for col in categorical_columns:
    if col not in ['House direction', 'Balcony direction', 'Furniture state']:
        data[col].fillna(data[col].mode()[0], inplace=True)
    else:
        # For specific columns, fill missing values with 'Unknown'
        data[col].fillna('Unknown', inplace=True)

# Summary statistics
print("\nSummary statistics after cleaning:")
print(data.describe())

# Save cleaned data to a new CSV file
output_file = "cleaned_data.csv"
data.to_csv(output_file, index=False)
print(f"\nCleaned data has been saved to {output_file}. (Existing file has been overwritten if it already existed.)")


# %% [markdown]
# Sau khi tiền xử lí dữ liêu , bắt đầu huấn luyện model
# 

# %%
import pandas as pd
import re

# Biểu thức chính quy để trích xuất các thành phần
project_pattern = r'Dự án\s([\w\s]+)'  # Để trích xuất dự án như "Dự án The Empire"
ward_pattern = r'(Phường\s[\w\s]+|Xã\s[\w\s]+)'  # Để trích xuất Phường/Xã
district_pattern = r'(Quận\s[\w\s]+|Huyện\s[\w\s]+Thị Trấn\s[\w\s])'  # Để trích xuất Quận/Huyện

# Hàm trích xuất các thành phần từ địa chỉ
def extract_address_components(address):
    if pd.isna(address):
        return None, None, None, None, None
    
    # Tách địa chỉ thành các phần bằng dấu phẩy và dấu gạch ngang
    address_parts = [part.strip() for part in re.split(r'[,\-]', address)]  # Tách theo dấu ',' và '-'

    # Trích xuất thành phố: lấy phần cuối cùng
    city = address_parts[-1] if len(address_parts) > 0 else None
    address_parts = address_parts[:-1]  # Loại bỏ thành phố khỏi mảng

    # Trích xuất tên dự án
    project = None
    project_match = re.search(project_pattern, address)
    if project_match:
        project = project_match.group(1).strip()
        address_parts = [part for part in address_parts if project not in part]  # Loại bỏ dự án khỏi các phần còn lại

    # Tách Phường/Xã và Quận/Huyện
    ward = None
    district = None
    
    # Tìm Phường/Xã
    ward_match = re.search(ward_pattern, address)
    if ward_match:
        ward = ward_match.group(0).strip()
        # Xóa "Phường" hoặc "Xã" khỏi kết quả
        ward = re.sub(r'(Phường|Xã|Thị Trấn)\s', '', ward)  # Loại bỏ "Phường" hoặc "Xã" cùng với khoảng trắng
        


    # Tìm Quận/Huyện
    district_match = re.search(district_pattern, address)
    if district_match:
        district = district_match.group(0).strip()

    # Xử lý phần còn lại sau khi loại bỏ dự án, phường/xã, quận/huyện
    remaining_address = ', '.join(address_parts).strip()  # Tạo lại phần còn lại của địa chỉ

    # Nếu có Phường/Xã nhưng không có Quận/Huyện, lấy phần sau Phường/Xã làm Quận/Huyện
    if ward and not district:
        ward_index = remaining_address.find(ward)
        if ward_index != -1:
            remaining_after_ward = remaining_address[ward_index + len(ward):].strip()
            remaining_parts = remaining_after_ward.split(',')
            if len(remaining_parts) > 1:
                district = remaining_parts[1].strip()
            else:
                district = remaining_parts[0].strip()

    remaining_address = re.sub(r'\s*(Phường|Xã|Thị Trấn)\s*[\w\s]+', '', remaining_address).strip()
    # Loại bỏ Quận/Huyện cùng tên của chúng
    remaining_address = re.sub(r'\s*(Quận|Huyện)\s*[\w\s]+', '', remaining_address).strip()

    # Loại bỏ ward nếu có
    if ward:
        remaining_address = re.sub(rf'\s*{re.escape(ward)}\s*', '', remaining_address).strip()

    # Loại bỏ district nếu có
    if district:
        remaining_address = re.sub(rf'\s*{re.escape(district)}\s*', '', remaining_address).strip()

    # Loại bỏ mọi dấu phẩy thừa trong địa chỉ (bao gồm dấu phẩy liên tiếp)
    remaining_address = re.sub(r',\s*,', ',', remaining_address)  # Loại bỏ dấu phẩy liên tiếp
    remaining_address = re.sub(r',\s*$', '', remaining_address)  # Loại bỏ dấu phẩy cuối cùng nếu còn

    # Phần còn lại của địa chỉ sau khi loại bỏ tất cả sẽ là location
    location = remaining_address.strip()

    # Nếu location còn lại trống, có thể gán nó là None hoặc một giá trị khác để tránh dấu phẩy thừa
    if not location:
        location = None
        
    return project, city, ward, district, location


# Áp dụng hàm để tạo cột mới
data[['Project', 'City', 'Ward', 'District', 'Location']] = data['Address'].apply(
    lambda x: pd.Series(extract_address_components(x))
)

# Hiển thị kết quả
data.head(100)
data.isnull().sum()


# %%
import pandas as pd
import os
import pickle  # Thay vì joblib, sử dụng pickle để lưu trữ đối tượng
from sklearn.preprocessing import LabelEncoder  # Import đúng từ sklearn



# Tạo đối tượng LabelEncoder
label_encoders = {}

# Lặp qua từng cột trong DataFrame và mã hóa các cột có kiểu chuỗi (object)
for column in data.columns:
    if data[column].dtype == 'object':
        # Nếu có giá trị NaN trong cột, điền bằng 'Unknown'
        if data[column].isnull().any():
            data[column] = data[column].fillna('Unknown')
        
        # Khởi tạo LabelEncoder và thực hiện mã hóa
        encoder = LabelEncoder()  # Sử dụng đúng LabelEncoder từ sklearn
        data[column] = encoder.fit_transform(data[column])

        # Lưu lại LabelEncoder vào dictionary
        label_encoders[column] = encoder

# In dữ liệu đã được mã hóa
print("\nData after encoding:")
print(data.head())

# Kiểm tra thư mục lưu trữ các LabelEncoder
output_dir = 'saved_encoders'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Lưu các LabelEncoder vào thư mục bằng pickle
for column, encoder in label_encoders.items():
    encoder_file = os.path.join(output_dir, f'label_encoder_{column}.pkl')
    try:
        # Lưu bằng pickle
        with open(encoder_file, 'wb') as f:
            pickle.dump(encoder, f)
        print(f"Đã lưu LabelEncoder cho cột {column} vào {encoder_file}")
    except Exception as e:
        print(f"Lỗi khi lưu LabelEncoder cho cột {column}: {str(e)}")

# Kiểm tra lại các file đã được lưu
print("\nList of saved LabelEncoders:")
for column in label_encoders:
    encoder_file = os.path.join(output_dir, f'label_encoder_{column}.pkl')
    if os.path.exists(encoder_file):
        print(f"{column}: {os.path.abspath(encoder_file)}")
    else:
        print(f"{column}: Lỗi - không lưu được")
data.isnull().sum()


# %% [markdown]
# ## Chia bộ dữ liệu ra để test và train

# %%
# Step 4: Define features and target
X = data.drop(['Address', 'Price'], axis=1)  # Features
y = data['Price']  # Target

# Step 5: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train.head()
X_train.info()

# %% [markdown]
# ## Hyper tunning
# 

# %%
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import numpy as np



# Bước 1: Xác định các siêu tham số cần tối ưu hóa
# Số lượng cây trong rừng
n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]



# Chiều sâu tối đa của cây
max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
max_depth.append(None)  # Cho phép cây phát triển đến điều kiện dừng

# Số lượng đặc trưng được xem xét tại mỗi lần phân chia
max_features = ['auto', 'sqrt']

# Số mẫu tối thiểu để chia một nút
min_samples_split = [2, 5, 10]

# Số mẫu tối thiểu tại mỗi lá
min_samples_leaf = [1, 2, 4]

# Phương pháp chọn mẫu
bootstrap = [True, False]

# Bước 2: Tạo không gian tìm kiếm siêu tham số
random_grid = {
    'n_estimators': n_estimators,
    'max_features': max_features,
    'max_depth': max_depth,
    'min_samples_split': min_samples_split,
    'min_samples_leaf': min_samples_leaf,
    'bootstrap': bootstrap
}

# In ra để kiểm tra
print("Không gian siêu tham số:")
print(random_grid)

# %%
from sklearn.model_selection import RandomizedSearchCV

# Tạo mô hình RandomForestRegressor (đã được khởi tạo trước đó)
rf = RandomForestRegressor(random_state=42)

# Tạo một đối tượng RandomizedSearchCV để tìm kiếm siêu tham số với 100 vòng lặp (n_iter)
rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=100,
                               cv=3, scoring='neg_mean_squared_error', verbose=0, random_state=42, n_jobs=-1)

# Huấn luyện mô hình RandomizedSearchCV với dữ liệu huấn luyện đã lọc
rf_random.fit(X_train, y_train)

# In kết quả tham số tốt nhất
print("Best parameters found: ", rf_random.best_params_)

# %%
# Step 7: Evaluate the model
# Khởi tạo mô hình RandomForestRegressor với các tham số tối ưu tìm được từ RandomizedSearchCV
# **best_params: Truyền các tham số tốt nhất (được tìm thấy trong quá trình tìm kiếm) vào mô hình RandomForestRegressor
model = RandomForestRegressor(**rf_random.best_params_, random_state=42, n_jobs=-1)

# Huấn luyện mô hình với dữ liệu huấn luyện đã lọc (X_train_filtered, y_train)
model.fit(X_train, y_train)








# %%
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error: {mae}")

# %%
# Lưu mô hình vào thư mục hiện tại
import joblib
joblib.dump(model, 'random_forest_model.pkl')

# %%
from sklearn.ensemble import GradientBoostingRegressor

gbm_model = GradientBoostingRegressor(
    n_estimators=1000,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    random_state=42
)

gbm_model.fit(X_train, y_train)
y_pred = gbm_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# In kết quả
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"R-squared (R²): {r2:.4f}")


# %%
feature_importances = model.feature_importances_
feature_names = X_train.columns
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
print(importance_df.sort_values(by='Importance', ascending=False))


# khuc nay la ham dung model de du doan ne , api dung ham nay nhaaaa

import pandas as pd
import joblib
import os
import pickle

def load_model_and_encoders(model_path, encoder_dir):
    """
    Tải mô hình và các LabelEncoder từ file.

    Parameters:
    - model_path: Đường dẫn tới file mô hình.
    - encoder_dir: Đường dẫn tới thư mục chứa các encoder.

    Returns:
    - model: Mô hình đã huấn luyện.
    - encoders: Dictionary chứa các encoder theo tên cột.
    """
    # Tải mô hình
    model = joblib.load(model_path)
    
    # Tải các encoder
    encoders = {}
    for filename in os.listdir(encoder_dir):
        if filename.startswith('label_encoder_') and filename.endswith('.pkl'):
            column_name = filename.split('label_encoder_')[1].split('.pkl')[0]
            with open(os.path.join(encoder_dir, filename), 'rb') as f:
                encoders[column_name] = pickle.load(f)
    
    return model, encoders

def preprocess_input(input_data, encoders, model):
    """
    Tiền xử lý dữ liệu đầu vào: mã hóa và sắp xếp đúng thứ tự cột.

    Parameters:
    - input_data: Dictionary chứa dữ liệu đầu vào.
    - encoders: Dictionary chứa các LabelEncoder.
    - model: Mô hình đã huấn luyện (để lấy thông tin cột).

    Returns:
    - processed_df: DataFrame đã qua tiền xử lý, sắp xếp theo thứ tự cột của mô hình.
    """
    # Tạo DataFrame từ input_data
    input_df = pd.DataFrame([input_data])
    
    # Mã hóa các cột phân loại và điền giá trị Unknown nếu có giá trị rỗng
    for column, encoder in encoders.items():
        if column in input_df.columns:
            # Kiểm tra và điền giá trị Unknown nếu có NaN
            if input_df[column].isnull().any():
                input_df[column].fillna('Unknown', inplace=True)
            
            value = input_df.at[0, column]
            
            # Mã hóa cột nếu là dữ liệu phân loại
            if value in encoder.classes_:
                input_df.at[0, column] = encoder.transform([value])[0]
            else:
                input_df.at[0, column] = 0  # Giá trị mới chưa từng xuất hiện
    
    # Điền giá trị thiếu bằng 0 cho các cột số học
    input_df.fillna(0, inplace=True)
    
    # Sắp xếp đúng thứ tự cột dựa trên cột mà mô hình cần
    required_columns = model.feature_names_in_
    processed_df = input_df.reindex(columns=required_columns, fill_value=0)
    
    # In ra DataFrame đã xử lý
    print(input_data)
    print(processed_df)

    return processed_df



def predict_house_price(input_data, model_path, encoder_dir):
    """
    Dự đoán giá nhà từ dữ liệu đầu vào.

    Parameters:
    - input_data: Dictionary chứa dữ liệu đầu vào.
    - model_path: Đường dẫn tới file mô hình.
    - encoder_dir: Đường dẫn tới thư mục chứa các encoder.

    Returns:
    - Giá nhà dự đoán.
    """
    # Tải mô hình và encoders
    model, encoders = load_model_and_encoders(model_path, encoder_dir)
    
    # Tiền xử lý dữ liệu
    processed_data = preprocess_input(input_data, encoders, model)
    
    # Dự đoán giá
    predicted_price = model.predict(processed_data)
    return predicted_price[0]
# Đường dẫn tới file mô hình và thư mục chứa các encoder
model_path = 'random_forest_model.pkl'
encoder_dir = 'saved_encoders'

# Dữ liệu đầu vào
input_data = {
    'Address': 'Dự án Vinhomes Grand Park, Phường Long Thạnh Mỹ, Quận 9, Hồ Chí Minh',
    'Area': 50,
    'Frontage': 4.5, 
    'Access Road': 6,
    'House direction': 'Unknown',
    'Balcony direction': 'Unknown',
    'Floors': 3,
    'Bedrooms': 1,
    'Bathrooms': 1,
    'Legal status': 'Sale contract',
    'Furniture state': 'Basic',
    'Project': 'Vinhomes Grand Park',
    'City': 'Hồ Chí Minh',
    'Ward': 'Long Thạnh Mỹ',
    'District': ' Quận 9',
    'Location': 'Unknown'
}

# Dự đoán giá nhà
predicted_price = predict_house_price(input_data, model_path, encoder_dir)
print(f"Giá nhà dự đoán: {predicted_price:.2f}")







