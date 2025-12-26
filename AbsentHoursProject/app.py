from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans

#1)Creating Flask instance
app = Flask(__name__)

#2)Load model
with open("absent_hours_prediction.pkl", "rb") as f:
    model = pickle.load(f)

# Options for dropdowns (extracted from the original data during training)
gender_options = ['M', 'F']
city_options = ['Burnaby', 'Courtenay', 'Richmond', 'Victoria', 'New Westminster', 'Vancouver', 'Sechelt', 'Kamloops', 'North Vancouver', 'Vananda', 'West Vancouver', 'Nanaimo', 'Aldergrove', 'Kelowna', 'Trail', 'Penticton', 'Duncan', 'Crawford Bay', 'Port Hardy', 'Logan Lake', 'Abbotsford', 'Dawson Creek', 'Surrey', 'Squamish', 'Good Hope Lake', 'Sidney', 'Coquitlam', 'Chilliwack', 'Okanagan Mission', 'Ganges', 'Prince George', 'Atlin', 'Whistler', 'Spences Bridge', 'New Westminister', 'Gibsons', 'Vernon', 'Fauquier', 'Mackenzie', 'Gold Bridge', 'Fort Fraser', 'Nelson', 'Kaslo', 'Creston', 'Wynndel', 'Muncho Lake', 'Kitimat', 'Lac La Hache', 'Armstrong', 'Quesnel', 'Hixon', 'Ocean Falls', 'Vallican', 'North Pender Island', 'Montney', 'Burns Lake', 'Midway', 'Westwold', 'Terrace', 'Golden', 'Port Coquitlam', 'White Rock', 'Summerland', 'Langley', 'Huntingdon', 'Yale', 'Aiyansh', 'Haney', 'Bowen Island', 'Fort Langley', 'Clinton', 'Cassiar', 'Pender Harbour', 'Parksville', 'Mcleese Lake', 'Fort St John', 'Campbell River', 'Wells', 'Manning Park', 'Decker Lake', 'Salmon Arm', 'Sooke', 'Horsefly', 'Invermere', 'Topley', 'Field', 'Youbou', 'Sorrento', 'Merritt', 'Rossland', 'Castlegar', 'Williams Lake', 'Willow Point', 'Woss', 'Cobble Hill', 'Bob Quinn Lake', 'Hedley', 'Sardis', 'Sparwood', 'Port Mcneill', 'Salmon Valley', 'Agassiz', 'Avola', 'Bear Lake', 'Brackendale', 'Skookumchuck', 'Cranbrook', 'Britannia Beach', 'Comox', 'Flatrock', 'Princeton', 'Fort Nelson', 'Slocan', 'Lumby', 'Chase', 'Boston Bar', 'Port Alberni', 'Union Bay', 'Francois Lake', 'Chief Lake', 'Beaver Valley', 'Oyster River', "D'arcy", 'Winfield', 'Valemount', 'Port Alice', 'Lakeview Heights', 'Canal Flats', 'Tofino', 'Mayne Island', 'Bamfield', 'Chilako River', 'Grasmere', 'Clearwater', 'Osoyoos', 'Grassy Plains', 'Fulford Harbour', 'Keremeos', 'Fort St James', 'Pitt Meadows', 'Lakelse Lake', 'Genelle', 'Iskut', 'Qualicum Beach', 'Oyama', 'Lillooet', 'Ladysmith', 'Chemainus', 'Douglas Lake', 'Yarrow', 'Elkford', 'Telegraph Creek', 'Mission', 'Fruitvale', 'Vanderhoof', 'Powell River', 'Pemberton', 'Tumbler Ridge', 'Chilanko Forks', 'Ashcroft', 'Klemtu', 'Cumberland', 'Port Mellon', 'Cortes Island', 'Jaffray', 'Black Point', 'Forest Grove', 'Grand Forks', 'Hazelton', 'Chetwynd', 'Riske Creek', 'Nakusp', 'Rutland', 'Bouchie Lake', 'Greenwood', 'Westbank', 'Houston', 'Gabriola Island', 'Port Renfrew', 'Wildwood', 'Port Edward', 'Balfour', 'Pritchard', 'Nimpo Lake', 'Black Pool', 'Vavenby', 'Oliver', 'Hemlock Valley', 'Bella Bella', 'Alexis Creek', 'Parson', 'Fraser Lake', 'Seton Portage', 'Rock Creek', 'Kimberley', 'Mcbride', 'Peachland', 'Dragon Lake', 'Tappen', 'Taylor', 'Salmo', 'Hansard', 'Dease Lake', 'Giscome', 'Granisle', 'Ucluelet', 'Blue River', 'Kitwanga', 'Mica Creek', 'Fernie', 'Pouce Coupe', 'Rosedale', 'Bougie Creek', 'Bridge Lake', 'Lake Cowichan', 'Revelstoke', 'Tatla Lake', 'Enderby', 'Hope', 'Radium Hot Springs', 'Sayward', 'Lower Post', 'South Slocan', 'Yahk', 'Elko', 'Christina Lake', 'Cluculz Lake', 'Toad River', 'Celista', 'Smithers', 'Little Fort', 'Sandspit', 'Quadra Island', 'Blueberry', 'Okanagan Falls', 'Cache Creek', 'Barriere', 'Lytton', 'Sicamous', 'Likely', 'Fairmont Hot Springs', 'Alkali Lake', 'Spillimacheen']
city_options.sort()
jobtitle_options = ['Baker', 'Accounting Clerk', 'Accounts Payable Clerk', 'Accounts Receiveable Clerk', 'Auditor', 'Bakery Manager', 'Benefits Admin', 'Cashier', 'Compensation Analyst', 'HRIS Analyst', 'Investment Analyst', 'Labor Relations Analyst', 'Recruiter', 'Dairy Person', 'Systems Analyst', 'Trainer', 'Meat Cutter', 'CEO', 'VP Stores', 'Legal Counsel', 'VP Human Resources', 'VP Finance', 'Exec Assistant, VP Stores', 'Exec Assistant, Human Resources', 'Exec Assistant, Legal Counsel', 'CHief Information Officer', 'Store Manager', 'Meats Manager', 'Exec Assistant, Finance', 'Director, Recruitment', 'Director, Training', 'Director, Labor Relations', 'Director, HR Technology', 'Director, Employee Records', 'Director, Compensation', 'Corporate Lawyer', 'Produce Manager', 'Director, Accounts Receivable', 'Director, Accounts Payable', 'Director, Audit', 'Director, Accounting', 'Director, Investments', 'Processed Foods Manager', 'Customer Service Manager', 'Dairy Manager', 'Produce Clerk', 'Shelf Stocker']  # Based on dataset sample
jobtitle_options.sort()
department_options = ['Bakery', 'Accounting', 'Accounts Payable', 'Accounts Receiveable', 'Audit', 'Employee Records', 'Customer Service', 'Compensation', 'HR Technology', 'Investment', 'Labor Relations', 'Recruitment', 'Dairy', 'Information Technology', 'Training', 'Meats', 'Executive', 'Store Management', 'Legal', 'Produce', 'Processed Foods']
department_options.sort()
store_location_options = ['Burnaby', 'Nanaimo', 'Richmond', 'Victoria', 'New Westminster', 'Vancouver', 'West Vancouver', 'Kamloops', 'North Vancouver', 'Aldergrove', 'Kelowna', 'Trail', 'Quesnel', 'Cranbrook', 'Abbotsford', 'Dawson Creek', 'Surrey', 'Squamish', 'Terrace', 'Chilliwack', 'Prince George', 'New Westminister', 'Vernon', 'Nelson', 'Fort St John', 'Williams Lake', 'Ocean Falls', 'Port Coquitlam', 'White Rock', 'Langley', 'Haney', 'Princeton', 'Fort Nelson', 'Valemount', 'Pitt Meadows', 'Bella Bella', 'Cortes Island', 'Grand Forks', 'Dease Lake', 'Blue River']
store_location_options.sort()
division_options = ['Executive', 'FinanceAndAccounting', 'HumanResources', 'InfoTech', 'Legal', 'Stores']
division_options.sort()
business_unit_options = ['HeadOffice', 'Stores']

#3)Routing the Flask app
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        
        data = {
            'Gender': request.form['gender'],
            'City': request.form['city'],
            'JobTitle': request.form['jobtitle'],
            'DepartmentName': request.form['department'],
            'StoreLocation': request.form['storelocation'],
            'Division': request.form['division'],
            'BusinessUnit': request.form['businessunit'],
            'Age': float(request.form['age']),
            'LengthService': float(request.form['lengthservice'])
        }
        df = pd.DataFrame([data])
        prediction = model.predict(df)[0]

    return render_template("index3.html", 
                           prediction=prediction,
                           gender_options=gender_options,
                           city_options=city_options,
                           jobtitle_options=jobtitle_options,
                           department_options=department_options,
                           store_location_options=store_location_options,
                           division_options=division_options,
                           business_unit_options=business_unit_options)
    
@app.route("/categorize")
def categorize():
    df = pd.read_csv("data/MFGEmployees.csv").dropna().drop_duplicates()
    df = df[(df['Age'] > 18) & (df['Age'] < 65)]
    # Select features excluding 'AbsentHours', name-related fields
    features = df.drop(columns=['EmployeeNumber', 'Surname', 'GivenName', 'AbsentHours'], errors='ignore')

    num_cols = features.select_dtypes(include=np.number).columns
    cat_cols = features.select_dtypes(include='object').columns

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ])

    X_scaled = preprocessor.fit_transform(features)

    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X_scaled)

    # Assign category labels based on cluster behavior
    cluster_avg_absence = df.groupby('Cluster')['AbsentHours'].mean().sort_values()
    cluster_to_label = {
        cluster_id: label for cluster_id, label in zip(
            cluster_avg_absence.index,
            ['Reliable Contributor', 'Generally Dependable', 'Needs Flexibility Support']
        )
    }

    cluster_to_color = {
        'Reliable Contributor': 'text-green-600',
        'Generally Dependable': 'text-yellow-600',
        'Needs Flexibility Support': 'text-red-600'
    }

    df['AbsentHoursCategory'] = df['Cluster'].map(cluster_to_label)
    df = df.sort_values(by='AbsentHours', ascending=False)

    # Select columns to show in table
    display_cols = ['GivenName', 'Surname', 'Gender', 'City', 'JobTitle',
                    'DepartmentName', 'StoreLocation', 'Division', 'AbsentHours', 'AbsentHoursCategory']

    categorized_data = []
    for label in ['Reliable Contributor', 'Generally Dependable', 'Needs Flexibility Support']:
        group = df[df['AbsentHoursCategory'] == label].head(10)
        categorized_data.append((label, cluster_to_color[label], group[display_cols].to_dict(orient='records')))

    return render_template("categorize.html", categorized_data=categorized_data)


#4)Running the Flask app
if __name__ == '__main__':
    app.run(debug=True)

