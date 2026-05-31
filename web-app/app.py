# imports
from flask import Flask, render_template, request, jsonify
import pandas as pd       
import pickle             
import re                
from nltk.stem import PorterStemmer  

# initializing flask
app = Flask(__name__)  

# load dataset
df = pd.read_csv('assignment3_II.csv')
df = df.reset_index() # add 'index' col to identify reviews (for url)

# load stopwords
with open('stopwords_en.txt', 'r') as f:
    stopwords = set(f.read().splitlines())

# load vocab dict to int indices
vocab_dict = {}
with open('vocab.txt', 'r') as f:
    for line in f.read().splitlines():
        word, idx = line.split(':')
        vocab_dict[word] = int(idx)

# load the trained model 
with open('logistic_regression_model.pkl', 'rb') as f:
    model = pickle.load(f)

# load count vectorizer for converting review text to feature vectors
with open('count_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

new_reviews = {}

# getting depart name for the tabs
departments = df['Department Name'].dropna().unique().tolist()

# mapping to show product images
title_images = {
    'Elegant A-Line Dress': 'pic1.jpeg',
    'Petite Floral Midi Dress': 'pic3.jpeg',
    'Petite High-Waisted Trousers': 'pic4.jpeg',
    'Tailored Wide-Leg Pants': 'pic5.jpeg',
    'Pleated Midi Skirt': 'pic6.jpeg',
    'Petite Pencil Skirt': 'pic7.jpeg',
    'Classic Straight-Leg Jeans': 'pic8.jpeg',
    'Petite Skinny Jeans': 'pic9.jpeg',
    'Casual Bermuda Shorts': 'pic10.jpeg',
    'Casual Jogger Pants': 'pic11.jpeg',
    'Silk Button-Up Blouse': 'pic12.jpeg',
    'Petite Satin Blouse': 'pic13.jpeg',
    'Petite Cable Knit Sweater': 'pic14.jpeg',
    'Classic Ribbed Knit Top': 'pic2.jpeg',
    'Cashmere Turtleneck Sweater': 'pic15.jpeg',
    'Petite Chunky Knit Sweater': 'pic16.jpeg',
    'Lightweight Fine Gauge Sweater': 'pic17.jpeg',
    'Petite Fine Gauge Cardigan': 'pic18.jpeg',
    'Lace Trimmed Intimates Set': 'pic19.jpeg',
    'Soft Lounge Set': 'pic20.jpeg',
    'Petite Lounge Pants': 'pic21.jpeg',
    'Soft Sleepwear Set': 'pic22.jpeg',
    'Vintage-Inspired Swimwear': 'pic23.jpeg',
    'Sheer Legwear Tights': 'pic24.jpeg',
    'Layering Bodysuit': 'pic25.jpeg',
    'Lace Trim Chemise': 'pic26.jpeg',
    'Classic Wool Overcoat': 'pic27.jpeg',
    'Petite Puffer Jacket': 'pic28.jpeg',
    'Tailored Blazer Jacket': 'pic29.jpeg',
    'Petite Quilted Jacket': 'pic30.jpeg',
    'Boho-Chic Maxi Dress': 'pic31.jpeg',
    'Petite Trendy Wrap Dress': 'pic32.jpeg'
}

class_images = {
    'Dresses': 'pic1.jpeg',
    'Pants': 'pic5.jpeg',
    'Skirts': 'pic6.jpeg',
    'Jeans': 'pic8.jpeg',
    'Shorts': 'pic10.jpeg',
    'Casual bottoms': 'pic11.jpeg',
    'Blouses': 'pic12.jpeg',
    'Knits': 'pic14.jpeg',
    'Sweaters': 'pic15.jpeg',
    'Fine gauge': 'pic18.jpeg',
    'Intimates': 'pic19.jpeg',
    'Lounge': 'pic20.jpeg',
    'Sleep': 'pic22.jpeg',
    'Swim': 'pic23.jpeg',
    'Legwear': 'pic24.jpeg',
    'Layering': 'pic25.jpeg',
    'Chemises': 'pic26.jpeg',
    'Outerwear': 'pic28.jpeg',
    'Jackets': 'pic29.jpeg',
    'Trend': 'pic31.jpeg',
}

# to show the nav tabs on every page
@app.context_processor
def inject_departments():
    return dict(departments=departments)


# preprocessing 
pattern = r"[a-zA-Z]+(?:[-'][a-zA-Z]+)?"
stemmer = PorterStemmer()

def preprocess(text):
    tokens = re.findall(pattern, text.lower())
    tokens = [t for t in tokens if len(t) > 1 and t not in stopwords]
    return ' '.join(tokens)

def predict_recommendation(text):
    processed = preprocess(text)
    vector = vectorizer.transform([processed])
    prediction = model.predict(vector)[0]
    return int(prediction)

# search function
def search_items(query, df):
    if not query:
        # return all if no query
        return df.drop_duplicates(subset=['Clothing ID']).to_dict('records')

    query_lower = query.lower()
    query_words = query_lower.split()
    stemmed_query_words = [stemmer.stem(w) for w in query_words]

    results = []
    seen_ids = set() # tracking to avoid duplicates

    for _, row in df.iterrows():
        clothing_id = row.get('Clothing ID')
        if clothing_id in seen_ids:
            continue
        
        # fields to include as searchable
        searchable = ' '.join([
            str(row.get('Class Name', '')),
            str(row.get('Department Name', '')),
            str(row.get('Division Name', '')),
            str(row.get('Clothes Title', '')),
            str(row.get('Clothes Description', '')),
            str(row.get('Title', '')),
            str(row.get('Review Text', ''))
        ]).lower()

        stemmed_fields = [stemmer.stem(w) for w in searchable.split()]

        # match if: substring found anywhere or stemmed word matches
        if (any(q in searchable for q in query_words) or
                any(q in stemmed_fields for q in stemmed_query_words)):
            results.append(row.to_dict())
            seen_ids.add(clothing_id)

    return results


# routes
@app.route('/')
def index():
    # show the most recent products (10)
    recent_items = df.drop_duplicates(subset=['Clothing ID']).tail(8).to_dict('records')
    return render_template('home.html', recent_items=recent_items)


@app.route('/browse/department/<dept_name>')
def browse_department(dept_name):
    # list all the class names in the selected department
    classes = df[df['Department Name'] == dept_name]['Class Name'].dropna().unique().tolist()
    if not classes:
        return "Department not found", 404
    return render_template('browse_department.html', dept_name=dept_name, classes=classes, class_images=class_images)


@app.route('/browse/department/<dept_name>/class/<class_name>')
def browse_class(dept_name, class_name):
    # list all the clothes title in the selected class and department
    filtered = df[(df['Department Name'] == dept_name) & (df['Class Name'] == class_name)]
    titles = filtered['Clothes Title'].dropna().unique().tolist()
    if not titles:
        return "Class not found", 404
    return render_template('browse_class.html', dept_name=dept_name, class_name=class_name, titles=titles, title_images=title_images)


@app.route('/browse/department/<dept_name>/class/<class_name>/title/<path:clothes_title>')
def browse_title(dept_name, class_name, clothes_title):
    # show the product page
    filtered = df[
        (df['Department Name'] == dept_name) &
        (df['Class Name'] == class_name) &
        (df['Clothes Title'] == clothes_title)
    ]
    if filtered.empty:
        return "Title not found", 404
    product = filtered.iloc[0].to_dict() # show product info
    clothing_id = int(product['Clothing ID'])
    reviews = filtered.to_dict('records') # show all reviews for this product
    return render_template('product.html', product=product, reviews=reviews, clothing_id=clothing_id)


@app.route('/classify', methods=['POST'])
# function to call API endpoint to generate recommended IND
def classify():
    data = request.get_json() 
    text = (data.get('title', '') + ' ' + data.get('review_text', '')).strip() # use both for better accuracy
    prediction = predict_recommendation(text)
    return jsonify({'prediction': prediction})


@app.route('/search')
# function to handle the search bar
def search():
    query = request.args.get('q', '').strip() # q is the query param from GET req
    results = search_items(query, df)
    return render_template('search.html', results=results, query=query)


@app.route('/product/<int:clothing_id>')
def product_detail(clothing_id):
    product_rows = df[df['Clothing ID'] == clothing_id]
    if product_rows.empty:
        return "Product not found", 404
    product = product_rows.iloc[0].to_dict()
    clothes_title = product.get('Clothes Title', '')
    if clothes_title:
        # if the title exists, find all rows across the entire dataset with that same title
        # because the same product can appear under multiple Clothing IDs
        all_reviews = df[df['Clothes Title'] == clothes_title].to_dict('records')
    else:
        # if no title, just use the rows from the og Clothing ID
        all_reviews = product_rows.to_dict('records')
    # passing the product info, all reviews, and clothing_id so the "Add a Review" button links correctly
    return render_template('product.html', product=product,
                           reviews=all_reviews, clothing_id=clothing_id)


@app.route('/item/<int:item_index>')
# function to show the item detail page
def item_detail(item_index):
    row = df[df['index'] == item_index] # find the row with matching index
    if row.empty:
        return "Review not found", 404
    item = row.iloc[0].to_dict() 
    return render_template('item.html', item=item)


@app.route('/review/new/<int:clothing_id>', methods=['GET', 'POST'])
# function to show the blank review form
def review_new(clothing_id):
    # clothing ID is passed to know which product is being reviewed
    return render_template('review_form.html', clothing_id=clothing_id)


@app.route('/review/confirm', methods=['POST'])
def review_confirm():
    global df # so we can add new review

    # retrieve data from the form
    clothing_id = request.form.get('clothing_id', '')
    title = request.form.get('title', '')
    review_text = request.form.get('review_text', '')
    rating = request.form.get('rating', '')
    age = request.form.get('age', '')
    recommended = request.form.get('recommended', '0')

    # use the new row's index as the review id so it's always retrievable
    next_index = int(df['index'].dropna().max()) + 1 # skips any NaN values to avoid errors
    review_id = str(next_index) # for templates and dict keys

    # look up the rest of the product details from existing data
    product_rows = df[df['Clothing ID'] == int(clothing_id)]
    if not product_rows.empty:
        product = product_rows.iloc[0]
        division_name = product.get('Division Name', '')
        department_name = product.get('Department Name', '')
        class_name = product.get('Class Name', '')
        clothes_title = product.get('Clothes Title', '')
        clothes_desc = product.get('Clothes Description', '')
    else:
        division_name = department_name = class_name = clothes_title = clothes_desc = ''

    new_row = {
        'Clothing ID': int(clothing_id),
        'Age': int(age) if age else 0,
        'Title': title,
        'Review Text': review_text,
        'Rating': int(rating) if rating else 0,
        'Recommended IND': int(recommended),
        'Positive Feedback Count': 0,
        'Division Name': division_name,
        'Department Name': department_name,
        'Class Name': class_name,
        'Clothes Title': clothes_title,
        'Clothes Description': clothes_desc
    }

    new_df_row = pd.DataFrame([new_row])
    new_df_row['index'] = next_index # add index column (for internal tracking)
    df = pd.concat([df, new_df_row], ignore_index=True)

    # drop the index column before appending to CSV file
    new_df_row.drop(columns=['index']).to_csv('assignment3_II.csv', mode='a', header=False, index=False)

    # for the web template
    review = {
        'id': review_id, # for url
        'clothing_id': clothing_id, # for the back to product link
        'title': title,
        'review_text': review_text,
        'rating': rating,
        'age': age,
        'recommended': recommended
    }

    new_reviews[review_id] = review # store in dict so it's retrievable 

    return render_template('review_result.html', review=review)


@app.route('/review/<review_id>')
def review_view(review_id):
    # try from the new review dict first
    review = new_reviews.get(review_id)
    # if not found then find in the csv
    if not review:
        try:
            row = df[df['index'] == int(review_id)]
            if row.empty:
                # if not found then it doesn't exist
                return "Review not found", 404
            r = row.iloc[0].to_dict()
            # rebuild in the same format as new_reviews entries
            review = {
                'id': review_id,
                'clothing_id': str(r.get('Clothing ID', '')),
                'title': str(r.get('Title', '')),
                'review_text': str(r.get('Review Text', '')),
                'rating': str(r.get('Rating', '')),
                'age': str(r.get('Age', '')),
                'recommended': str(int(r.get('Recommended IND', 0)))
            }
        except:
            return "Review not found", 404
    return render_template('review_view.html', review=review)


if __name__ == '__main__':
    app.run(debug=True)