# 📊 AI Campaign Cost Prediction – 2023–24

A machine learning project to predict YouTube campaign costs using influencer metrics and video characteristics. This project involves data cleaning, feature engineering, model training, and deploying an API using FastAPI for real-time cost predictions.

---

## 📁 Dataset

- **Source:** Google Sheets (CSV export link)
- **Name:** `New_AI_Final Data`
- **Path:**  
  `MyDrive/Colab Notebooks/1. Project 1: AI CAMPAIGN COST 2023-24 PROJECT (Done)/AI Cost Data.gsheet`

**Data Fetching Strategy:**  
The dataset is hosted on Google Sheets and accessed programmatically by exporting it as a CSV file.
Explanation: Instead of manually downloading the file, a direct link is used to fetch the data in .csv format via the Google Sheets export URL. This method ensures the data is always up-to-date and can be integrated directly into data pipelines or notebooks.

---

## 🧠 Feature Descriptions

| Feature                              | Description |
|--------------------------------------|-------------|
| `brand_name`                         | Advertiser running the campaign (e.g., mStock) |
| `product_name`                       | Product/service promoted |
| `yt_username`                        | YouTube influencer's channel name |
| `overall_videos_viewcount_of_channel`| Total video views on the channel |
| `channels_subscriber`                | Total number of subscribers |
| `first_five_avg_video_of_channel`    | Avg. metrics from first 5 videos |
| `engagement_rate_of_channel`         | Engagement rate from audience |
| `video_views`                        | Views on the campaign video |
| `video_duration`                     | Video length (in seconds) |
| `category`                           | Content type (e.g., Finance, Tech) |
| `video_type`                         | Video format (e.g., Dedicated, Shorts) |

---

## 🧰 Tools & Libraries

- **pandas** 
- **numpy**
- **matplotlib**, **seaborn**
- **scikit-learn**, **XGBoost**
- **FastAPI** 
- **joblib**

---

## 🔍 Data Analysis Workflow

1. **Data Cleaning With Pandas**  

2. **EDA (Exploratory Data Analysis):**
   Co-Relation (HEATMAP):
   
   - video_cost vs video_type
   - video_cost vs category
   - video_cost vs engagement_rate_of_channel

4. **Outlier Detection with Scatter & Box Plots**  
  
5. **Brand-Specific Outlier Cleaning**  
   - Each brand had its own subset of outliers removed using IQR (Interquartile Range)
   - Notable Features
     
     ● Outlier Removal Strategy:
       Each category of campaign is analyzed and cleaned individually.
       A sequence of before-and-after visuals highlights the effect of outlier filtering.

6. **Feature Engineering Before Feeding to Model**  
After testing with many models, both RandomForestRegressor and XGBRegressor
provided good accuracy for predicting campaign costs.

🧬 Feature Encoding Techniques
  - Tree-based models such as RandomForestRegressor and XGBRegressor can handle both One-Hot Encoding and Label Encoding effectively.
  - Both encodings were tested independently on the same dataset using the same models, to normalize the comparison and identify the better fit for cost prediction.
  - Here strategy to try both encodings and compare results was absolutely correct and aligned with best practices in machine learning


## 🧠 ML Model Learning & Evaluation

Several regression models were applied and evaluated using multiple error metrics to determine their effectiveness in predicting campaign costs.

### 🔍 Models Tested

#### 🔹 Linear Regression
- **Result:** Poor accuracy on test data.

#### 🔹 Ensemble Models – Bagging
- **RandomForestRegressor**
  - Provided **stable and accurate results**
  - Tested with both **One-Hot** and **Label Encoded** datasets
- **ExtraTreesRegressor**
  - Exhibited similar behavior and performance to RandomForest

#### 🔹 Ensemble Models – Boosting
- **GradientBoostingRegressor**
  - **Result:** Poor accuracy on test data
- **XGBRegressor**
  - Slightly **lower performance than RandomForest**, but still strong
  - Evaluated with both encoding techniques

#### 🔹 Support Vector Regression (SVR)
- **Result:** Poor accuracy on test data

---

### ✅ Best Performing Model

**🎯 RandomForestRegressor** using **Label Encoding** delivered the **best overall performance** across all models and evaluation metrics.

### 📈 Model Performance & Encoding Strategy
- R2 Score, MAE, RMSE, MSE
<img width="1097" alt="image" src="https://github.com/user-attachments/assets/7730dbbe-13dc-4c55-98dc-5e782d94e404" />

---

### 🔄 Encoding Strategy Comparison

In this project, both **One-Hot Encoding** and **Label Encoding** strategies were tested to evaluate their impact on:

- Feature space size  
- Model training time  
- Prediction error metrics  
- Model generalization capability
  
---

### 💡 Why Label Encoding Was Preferred:

- **Label Encoding** offers a **compact feature set**, making it more efficient in terms of memory and performance.
- **One-Hot Encoding** increases the number of features significantly, which:
  - May lead to **overfitting** if the dataset is not large enough  
  - Increases model complexity and training time unnecessarily

As a result, Label Encoding was selected for final model deployment.

---

### 🏆 Final Result

- **Best Model:** `RandomForestRegressor`  
- **Best Encoding Strategy:** `Label Encoding`

---

### 🔬 6. Model Testing

In the final model testing phase, the trained **RandomForestRegressor** model was:

- **Loaded** from the serialized `.pkl` file
- **Provided with new input data**
- **Used to predict campaign costs**

✅ The predictions confirmed the model’s **effectiveness** and **readiness for deployment** in a real-time environment.

---


### 🌐 7. API Endpoint Details

- **Route:** `/predict`  
- **Method:** `GET`  
- **Test URL:** [http://localhost:8000/predict](http://localhost:8000/predict)

#### 📥 Input Parameters:
1. `brand_name`  
2. `product_name`  
3. `for_username`  
4. `overall_videos_viewcount_of_channel`  
5. `subscribers_channel`  
6. `first_five_avg_video_of_channel`  
7. `engagement_rate_of_channel`  
8. `video_views`  
9. `video_duration` (in `HH:MM:SS`)  
10. `category`  
11. `video_type`

#### 📤 Output:
- Returns the **predicted `video_cost`** (in `float` format)

---

### 📁 Project Structure
AI_Campaign_Cost_Prediction_Project/
├── app.py # FastAPI app with /predict endpoint
├── Dockerfile # For containerizing the API
├── requirements.txt # Python dependencies
├── RF_trained_model.pkl # Trained RandomForestRegressor model

---

### ⚙️ Key Features

- ✅ Trained model loaded from `RF_trained_model.pkl`  
- 🔄 Input preprocessing includes:
  - Duration conversion (`HH:MM:SS` → seconds)
  - Label encoding for categorical fields  
- ❗ Handles unseen category values using fallback encoding (`-1`)  
- 🧰 Built using `joblib`, `pandas`, and `FastAPI`

---

### ▶️ Run the API

#### 📦 Install Required Libraries

```bash
pip install -r requirements.txt

To start the FastAPI application locally, run the following command:

🚀 Launch the FastAPI Server
uvicorn app:app --reload
