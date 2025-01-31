# HousePricePL: Predicting Real Estate Prices in Poland

## Project Structure Overview
This project is structured into several key folders, each responsible for a specific stage in the real estate price prediction pipeline. Below is a breakdown of the structure:

### **Project Structure:**
```
1_data_scraping
├── otodom_houses_scraping
│   ├── otodom_scraper
│   │   ├── spiders
│   │   │   ├── otodom_spider.py  # Main Scrapy spider for scraping data from Otodom.
│   │   ├── items.py              # Definition of the data structure for scraped items.
│   │   ├── pipelines.py          # Pipelines for processing scraped data.
│   │   ├── settings.py           # Configuration for Scrapy settings.
│   └── results
│       └── otodom_houses.json    # JSON file with the scraped data.

2_clean_data
├── cleaning
│   ├── basic_cleaning.py         # Functions to handle basic data cleaning (e.g., handling missing values).
│   ├── clustering.py             # Clustering logic for data exploration.
│   ├── encoding.py               # Encoding categorical variables.
│   ├── io.py                     # I/O operations for loading and saving data.
│   ├── statistics.py             # Basic statistics for data analysis.
├── results
│   └── otodom_houses_cleaned.csv # Cleaned data output.
└── main.py                       # Main script to run the cleaning pipeline.
3_train
├── best_results
│   ├── HistGradientBoosting.pkl  # Best-performing model saved as a pickle file.
│   ├── scaler.pkl                # Scaler used during model training.
│   ├── model_params.json         # Parameters for models and training configurations.
│   ├── mae_comparison.png        # Mean Absolute Error comparison across models.
│   ├── rmse_comparison.png       # Root Mean Square Error comparison across models.
│   ├── actual_vs_predicted.png   # Visualization of actual vs. predicted prices.
│   └── results.json              # JSON file summarizing training results.
├── results
│   ├── t1, t2, t3...             # Subfolders containing individual training runs.
│   │   ├── models                # Saved models, scalers, and result metrics.
│   │   └── visualizations        # Training-related visual outputs (e.g., error plots).
└── main.py                       # Main script for training models.

4_streamlit_app
├── app.py                        # Streamlit app for user interaction and prediction.
├── util_functions                # Utility functions used in the app.
```

## **Key Stages:**

1. **Data Scraping** (Folder: `1_data_scraping`)
   - Scrapes property data from Otodom using **Scrapy** and **Selenium**.
   - Saves listings dynamically in the `results` subfolder as a CSV.
   - **Command to start scraping:**
     ```bash
     scrapy crawl otodom_spider
     ```

2. **Data Cleaning** (Folder: `2_clean_data`)
   - Cleans and processes the raw data for analysis.
   - The cleaned data is saved in the `results` folder as a CSV.
   - Directly reads data output from the scraping stage without needing a manual path.
   - **Command to clean data:**
     ```bash
     python main.py
     ```

3. **Model Training** (Folder: `3_train`)
   - Trains models using configurations specified in `model_params.json`.
   - Automatically reads the cleaned data.
   - After training, the best model and scaler are saved in the `best_results` subfolder.
   - Training metrics and visualizations are also saved.
   - **Command to train models:**
     ```bash
     python main.py
     ```

4. **Dashboard for Predictions** (Folder: `streamlit_app`)
   - Streamlit app to predict property prices and display training metrics.
   - Users can input property features to get predictions.
   - **Command to start the Streamlit app:**
     ```bash
     streamlit run app.py
     ```

## **Requirements**
To run this project, you will need:

- **Python 3.8+**
- Required libraries listed in `requirements.txt` (Install using `pip install -r requirements.txt`)

### **Main libraries used:**
- Scrapy
- Selenium
- Pandas
- Scikit-learn
- XGBoost
- Streamlit

## **Instructions for Running the Project:**

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-repo-name/HousePricePL.git
   cd HousePricePL
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run each stage sequentially:**
   - **Data Scraping:**
     ```bash
     cd 1_data_scraping
     scrapy crawl otodom_spider
     ```

   - **Data Cleaning:**
     ```bash
     cd ../2_clean_data
     python main.py
     ```

   - **Model Training:**
     ```bash
     cd ../3_train
     python main.py
     ```

4. **Run the Streamlit App:**
   ```bash
   cd ../streamlit_app
   streamlit run app.py
   ```

## **Dashboard Overview:**
- The **Streamlit app** allows users to:
  - Input property features and get price predictions.
  - Visualize model performance metrics and insights.
