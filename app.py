from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import plotly.graph_objects as go

app = Flask(__name__)

# Load the dataset
data = pd.read_csv("news_dataset.csv", encoding='latin1')

# Check class distribution
print(data['label'].value_counts())

# Ensure correct label encoding
data['fake'] = data['label'].apply(lambda x: 1 if x == "Fake" else 0)

# Prepare data for training
X, y = data['statement'], data['fake']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and fit the TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
X_train_vectorized = vectorizer.fit_transform(X_train)

# Create and fit the classifier
clf = LinearSVC()
clf.fit(X_train_vectorized, y_train)

# Predict on training data
y_train_pred = clf.predict(X_train_vectorized)

# Compute evaluation metrics
accuracy = accuracy_score(y_train, y_train_pred)
precision = precision_score(y_train, y_train_pred)
recall = recall_score(y_train, y_train_pred)
f1 = f1_score(y_train, y_train_pred)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the statement from the form data
        statement = request.form.get('statement', '')
        if statement:
            # Transform the input text using the same vectorizer
            vectorized_statement = vectorizer.transform([statement])
            # Make the prediction
            prediction = clf.predict(vectorized_statement)[0]
            result = "FAKE" if prediction == 1 else "REAL"
            
            # Get the image URL corresponding to the statement
            try:
                image = data.loc[data['statement'] == statement, 'image'].iloc[0]  # Assuming 'image' is the column name
            except IndexError:
                # Handle the case where no image is found
                image = "No image found"
            
            return render_template('result.html', result=result, statement=statement, image=image)
        else:
            # Handle the case where the statement is not provided
            return render_template('result.html', result="ERROR", statement="Statement not provided")

@app.route('/data_analysis')
def data_analysis():
    # Subplot 1: Total Records
    total_records = len(data)
    fig1 = go.Figure(go.Indicator(mode="number", value=total_records, title="Total Records"))
    fig1.update_layout(title_text='Total Records', width=550)
    
    # Subplot 2: Fake vs. Real News Count (Pie Chart)
    fake_real_count = data['fake'].value_counts()
    fig2 = go.Figure(go.Pie(labels=fake_real_count.index, values=fake_real_count.values, hole=0.3, pull=[0, 0.1]))
    fig2.update_layout(title_text='Fake vs. Real News Count')
    
    # Subplot 3: Training Data Distribution (Bar Chart)
    training_data_distribution = pd.DataFrame({'Label': y_train.value_counts().index, 'Count': y_train.value_counts().values})
    fig3 = go.Figure(go.Bar(x=training_data_distribution['Label'], y=training_data_distribution['Count'], marker_color=['blue', 'red']))
    fig3.update_layout(title_text='Training Data Distribution')

    # Animated Visualization of Category Distribution
    category_distribution = data['category'].value_counts()
    fig4 = go.Figure(go.Bar(x=category_distribution.index, y=category_distribution.values))
    fig4.update_layout(title_text='Category Distribution Over Time', xaxis_title='Category', yaxis_title='Count')
    fig4.update_layout(updatemenus=[dict(type='buttons', showactive=False,
                                      buttons=[dict(label='Play',
                                                    method='animate',
                                                    args=[None, {'frame': {'duration': 500, 'redraw': True},
                                                                 'fromcurrent': True}]),
                                               dict(label='Pause',
                                                    method='animate',
                                                    args=[[None], {'frame': {'duration': 0, 'redraw': True},
                                                                   'mode': 'immediate',
                                                                   'transition': {'duration': 0}}])])])
    frames = [go.Frame(data=[go.Bar(x=category_distribution.index[:i+1], y=category_distribution.values[:i+1])],
                       name=str(i)) for i in range(len(category_distribution))]
    fig4.update(frames=frames)

    # Visualization of Fake vs. True News within each Category
    category_fake_true = data.groupby(['category', 'fake']).size().unstack(fill_value=0)
    fig5 = go.Figure(data=[
        go.Bar(name='True', x=category_fake_true.index, y=category_fake_true[0]),
        go.Bar(name='False', x=category_fake_true.index, y=category_fake_true[1])
    ])
    fig5.update_layout(title='Fake vs. True News within Each Category', xaxis_title='Category', yaxis_title='Count')
    fig5.update_layout(barmode='group')

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_train, y_train_pred)
    fig6 = go.Figure(data=go.Heatmap(z=conf_matrix, x=['REAL', 'FAKE'], y=['REAL', 'FAKE'], colorscale='Viridis'))
    fig6.update_layout(title='Confusion Matrix', xaxis_title='Predicted label', yaxis_title='True label')

    return render_template('data_analysis.html', 
                           plot_html1=fig1.to_html(full_html=False), 
                           plot_html2=fig2.to_html(full_html=False), 
                           plot_html3=fig3.to_html(full_html=False),
                           plot_html4=fig4.to_html(full_html=False),
                           plot_html5=fig5.to_html(full_html=False),
                           plot_html6=fig6.to_html(full_html=False),
                           accuracy=accuracy,
                           precision=precision,
                           recall=recall,
                           f1=f1)

if __name__ == '__main__':
    app.run(debug=True)
