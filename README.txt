# Clothing Review Web App

A full-stack web application for browsing and reviewing women's clothing,
built with Flask and powered by a machine learning sentiment model.

## Features
- Browse clothing products by department and category
- Keyword search with partial match and stemming support
- View all customer reviews per product
- Submit new reviews with ML-generated recommendation prediction
- Override ML prediction manually if desired
- New reviews persist across sessions

## Tech Stack
- **Backend:** Python, Flask
- **ML:** Scikit-learn (Logistic Regression, Count Vectorizer), NLTK
- **Frontend:** HTML, CSS, jQuery
- **Data:** Women's E-Commerce Clothing Reviews (Kaggle)

## Project Structure

### `data-preprocessing/`
Contains the data cleaning and ML model training pipeline:
- `task1.ipynb` / `task1.py` — data cleaning and preprocessing of the raw dataset, generating `processed.csv` with a cleaned `Processed Review Text` column
- `task2_3.ipynb` / `task2_3.py` — ML model training using Logistic Regression with Count Vectorizer, producing the trained model files used in the web app
- `assignment3.csv` — original raw dataset
- `processed.csv` — cleaned dataset output

### `web-app/`
Contains the Flask web application:
- `app.py` — main Flask application
- `logistic_regression_model.pkl` — trained ML model
- `count_vectorizer.pkl` — trained vectorizer
- `templates/` — HTML pages for browsing, searching, and reviewing
- `static/` — CSS, jQuery, and product images
- `assignment3_II.csv` — extended dataset with product title and description columns

> Note: `processed.csv` was generated from the data cleaning pipeline
> in `data-preprocessing/`. `assignment3_II.csv` is an extended version
> of the dataset with additional product title and description columns,
> used by the web app for browsing and display purposes.

## How to Run
1. Navigate to the web-app folder:
```
   cd web-app
```
2. Install dependencies:
```
   pip install -r requirements.txt
```
3. Run the app:
```
   python app.py
```
4. Open `http://127.0.0.1:5000` in your browser

## Credits
- Dataset: Women's E-Commerce Clothing Reviews by Nicapotato on Kaggle
- Product images sourced from Pinterest