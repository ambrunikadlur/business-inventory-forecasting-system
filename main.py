import asyncio
import os
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import base64
from flask import Flask, request, render_template, session
import firebase_admin
import random
from firebase_admin import credentials, firestore
from SendEmail import send_email
import time
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from SendEmail import send_email
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import asyncio
import keras
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import numpy as np


warnings.filterwarnings('ignore')
cred = credentials.Certificate("key.json")
firebase_admin.initialize_app(cred)
app = Flask(__name__)
app.secret_key = "MySecretKey@123"
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
"""
daynames=["Sunday","Monday","Tuesday",
          "Wednesday","Thursday","Friday","Saturday"]
timeslots=["Morning","Afternoon",
            "Evening","Midnight"]
areanames=["Silk Board","Electronic City",
                    "Whitefield", "Majestic", "KR Puram", "Hebbal"]
"""
async def get_plot(dates_test, y_test, predictions):
    plt.figure(figsize=(12, 6))
    plt.plot(dates_test, y_test, label='Actual Predicion')
    plt.plot(dates_test, predictions, label='Compertitor Prediction')
    plt.title('Actual vs Predicted Pricing')
    plt.xlabel('Date')
    plt.ylabel('Predicted Pricing')
    plt.legend()
    plt.show()
    await asyncio.sleep(1)
    return plt

@app.route('/usercheckbusiness', methods=['POST','GET'])
async def usercheckbusiness():
    try:
        msg,summary,filename="","",""
        if request.method == 'POST':
            id = str(round(time.time()))
            filename="Img_"+id+".jpg"
            columns=['Date','Store_ID','Product_ID','Category','Region','Inventory_Level',
                       'Units_Sold','Units_Ordered','Demand_Forecast','Price','Discount',
                       'Weather_Condition','Holiday_Promotion','Seasonality',
                       'Competitor_Pricing']
            #file_path='retail_store_inventory.csv'
            #df= pd.read_csv(file_path, usecols=columns)
            #print("Before Mapping : \n",df.head())
            storename = request.form['storename']
            productid = request.form['productid']
            category = request.form['category']
            region = request.form['region']
            inventorylevel = request.form['inventorylevel']
            unitssold = request.form['unitssold']
            unitsordered = request.form['unitsordered']
            demandforecast = request.form['demandforecast']
            price = request.form['price']
            discount = request.form['discount']
            weathercondition = request.form['weathercondition']
            holidaypromotion = request.form['holidaypromotion']
            seasonality = request.form['seasonality']
                        
            weatherconditions_mapping = generate_mappings(weatherconditions)
            seasons_mapping=generate_mappings(seasons)
            regions_mapping=generate_mappings(regions)
            categories_mapping=generate_mappings(categories)
            productids_mappings=generate_mappings(productids)
            storenames_mappings=generate_mappings(storenames)
            
            data = pd.read_csv('retail_store_inventory.csv', usecols=columns)
            print("Head : \n",data.head())
            #data = data[(data['Store_ID']==storename) & (data['Product_ID']==productid) &
            #            (data['Category']==category) & (data['Region']==region) &                        
            #            (data['Weather_Condition']==weathercondition) & (data['Seasonality']==seasonality)]
            data = data[(data['Store_ID']==storename) & (data['Product_ID']==productid) &
                        (data['Category']==category) & (data['Region']==region)]
            print("After : \n",data.head())
            data['Date'] = pd.to_datetime(data['Date'])
            data.set_index('Date', inplace=True)
            Competitor_Pricing = data['Competitor_Pricing'].astype(float).values.reshape(-1, 1)

            scaler = MinMaxScaler(feature_range=(0,1))
            scaled_data = scaler.fit_transform(Competitor_Pricing)

            print("Scaled Data : \n", scaled_data)

            window_size = 12
            X, y = [],[]
            target_dates = data.index[window_size:]

            for i in range(window_size, len(scaled_data)):
                X.append(scaled_data[i - window_size:i, 0])
                y.append(scaled_data[i, 0])

            X = np.array(X)
            y = np.array(y)
            
            X_train, X_test, y_train, y_test, dates_train, dates_test = train_test_split(
            X, y, target_dates, test_size=0.2, shuffle=False)
            
            print("X train : \n",X_train)

            X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
            X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
            
            model_path = "model.keras"
            model = keras.saving.load_model(model_path)
            #history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.1)

            predictions = model.predict(X_test)
            predictions = scaler.inverse_transform(predictions).flatten()
            y_test = scaler.inverse_transform(y_test.reshape(-1,1)).flatten()
            
            plt1 = await get_plot(dates_test, y_test, predictions)
            plt1.savefig(os.path.join(app.config['UPLOAD_FOLDER'],filename))

            rmse = np.sqrt(np.mean((y_test - predictions)**2))
            print(f'LSTM RMSE: {rmse:.2f}')
            
            #ARIMA Model
            # 1. Load and prepare your data (replace with your data loading)
            # Example: Create a sample time series
            # In a real scenario, you would use:
            file_path='retail_store_inventory.csv'
            df = pd.read_csv(file_path)
            df = df[(df['Store_ID']==storename) & (df['Product_ID']==productid) &
                        (df['Category']==category) & (df['Region']==region)]
            
             
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            
            # Sample data creation (replace with your actual data)
            data = df['Competitor_Pricing']
            df = pd.DataFrame(data)
            
            # 2. Split data into training and testing sets
            train_size = int(len(df) * 0.8)
            train, test = df[:train_size], df[train_size:]

            # 3. Define the ARIMA model parameters (p, d, q)
            # p: order of the AutoRegressive (AR) part
            # d: degree of differencing (I)
            # q: order of the Moving Average (MA) part
            # For this example, we use a simple (1, 1, 1) order
            p, d, q = 1, 1, 1
            
            # 4. Fit the model
            # The `ARIMA` function takes the training data and the order as input
            model = ARIMA(train['Competitor_Pricing'], order=(p, d, q))
            fitted_model = model.fit()
            print(fitted_model.summary()) # Print model summary
            
            summary=fitted_model.summary()
            
            # 5. Make predictions on the test set
            # Start and end indices for forecasting
            start_index = len(train)
            end_index = len(df) - 1
            
            # Forecast the test period
            predictions = fitted_model.predict(start=start_index, end=end_index, typ='levels')

            # 6. Evaluate the model (Optional, but recommended)
            # Calculate Mean Squared Error
            mse = mean_squared_error(test['Competitor_Pricing'], predictions)
            print(f'\nMean Squared Error (MSE): {mse}')
            print(f'Root Mean Squared Error (RMSE): {np.sqrt(mse)}')
            
            # 7. Plot the results (Optional)
            """
            plt.figure(figsize=(10, 6))
            plt.plot(train.index, train['Competitor_Pricing'], label='Training Data')
            plt.plot(test.index, test['Competitor_Pricing'], label='Actual Test Data')
            plt.plot(test.index, predictions, color='red', linestyle='--', label='ARIMA Forecast')
            plt.title('ARIMA Forecast vs Actuals')
            plt.xlabel('Time')
            plt.ylabel('Value')
            plt.legend()
            plt.show()
            """
            
            msg=f"Based on the Analysis Competitor Prediction Price is LSTM RMSE: {rmse:.2f} ARIMA RMSE : {np.sqrt(mse):.2f}"
            """
            print("After Mapping : \n",df.head())
            # Apply the mapping using the .map() method
            df['Weather_Condition'] = df['Weather_Condition'].map(weatherconditions_mapping)
            df['Seasonality'] = df['Seasonality'].map(seasons_mapping)
            df['Region'] = df['Region'].map(regions_mapping)
            df['Category'] = df['Category'].map(categories_mapping)
            df['Product_ID'] = df['Product_ID'].map(productids_mappings)
            df['Store_ID'] = df['Store_ID'].map(storenames_mappings)
            
            print("After Mapping : \n",df.head())
            print("DF Keys : \n",df.keys())

            print("X Keys : \n", df.iloc[:,0:12].keys())
            print("Y Keys : \n", df.iloc[:,13])

            X = df.iloc[:,0:13].values
            y = df.iloc[:,13].values
            
            # Convert continuous labels to discrete classes
            y = [int(label) for label in y]

            #print(X)
            #print(y)
            
            # 3. Split Data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

            # 4. Train Model
            model = RandomForestClassifier(n_estimators=100, random_state=42) # n_estimators is the number of trees
            model.fit(X_train, y_train)

            print("X Test : ",X_test[0], " Length : ", len(X_test[0]))

            # 5. Make Predictions
            y_pred = model.predict(X_test)
            print("Y Prediction : ", y_pred)
            
            user_test = (np.array([[storenames_mappings[storename],productids_mappings[productid],
                                categories_mapping[category], regions_mapping[region],
                                inventorylevel,unitssold,unitsordered,
                                demandforecast,price,discount,
                                weatherconditions_mapping[weathercondition],
                                holidaypromotion,seasons_mapping[seasonality]]]))
            print("User Test : \n",user_test,  " Length : ", len(user_test[0]))

            y_pred = model.predict(user_test)
            print("Prediction : ", y_pred[0])
            
            msg=f"Based on the Analysis Competitor Prediction Price is {y_pred[0]}"
            """
            userid = session['userid']
            #id = str(round(time.time()))
            if(holidaypromotion==0):
                holidaypromotion="Holiday"
            else:
                holidaypromotion="Promotion"
            json={"id":id, "UserId":userid,'StoreId':storename,
                  'ProductId':productid, 'Category':category,
                  'Region':region, "InventoryLevel":inventorylevel,
                  "UnitsSold":unitssold, "UnitsOrdered":unitsordered,
                  "DemandForecast":demandforecast, "Price":price,
                  "Discount":discount, "WeatherCondition":weathercondition,
                  "Holiday_Promotion":holidaypromotion,"Result":msg, 'FileName':filename}
            
            db = firestore.client()
            db_ref = db.collection('newreport')
            db_ref.document(id).set(json)            
        return render_template("usercheckbusiness.html", storenames=storenames,
                               productids=productids, categories=categories,
                               regions=regions, seasons=seasons,
                               weatherconditions=weatherconditions, msg=msg, summary=summary,
                               filename=filename)
    except Exception as e:
        print("Exception : ", e)
        return str(e)


@app.route('/userforgotpassword', methods=['POST','GET'])
def userforgotpassword():
    try:
        msg=""
        if(request.method=="POST"):
            uname = request.form['uname']            
            db = firestore.client()
            dbref = db.collection('newuser')
            userdata = dbref.get()
            data = []
            for doc in userdata:
                print(doc.to_dict())
                print(f'{doc.id} => {doc.to_dict()}')
                data.append(doc.to_dict())
            flag = False
            for temp in data:
                if uname == temp['UserName']:                
                    session['emailid'] = temp['EmailId']
                    emailid=temp['EmailId']
                    session['userid'] = temp['id']
                    session['username'] = temp['FirstName'] + " " + temp['LastName']
                    flag = True
                    break
            if (flag):
                return render_template("usersendotppage.html", emailid=emailid)
            else:
                msg = "UserName is Invalid"
        return render_template("userforgotpassword.html", msg=msg)
    except Exception as e:
        return str(e)

@app.route('/usersendotppage', methods=['POST','GET'])
def usersendotppage():
    try:
        msg=""
        if(request.method=="POST"):
            otp = str(random.randint(1000,9999))
            session["otp"]=otp
            toemail = request.form["email"]
            body = "OTP to change the password : " + otp
            subject = "Change Password"
            receipients = [toemail]
            send_email(subject, body, receipients)
            return render_template("userenterotppage.html", msg=msg)    
        return render_template("userforgotpassword.html", msg=msg)
    except Exception as e:
        return str(e)

@app.route('/userenterotppage', methods=['POST','GET'])
def userenterotppage():
    try:
        msg=""
        if(request.method=="POST"):
            storedotp = session["otp"]
            enteredotp = request.form["otp"]
            flag=False
            if(str(storedotp)==str(enteredotp)):
                flag=True
            if(flag):
                return render_template("userchangepasswordpage.html", msg=msg)
            else:
                msg="OTP is not matching"
        return render_template("userenterotppage.html", msg=msg)
    except Exception as e:
        return str(e)

@app.route('/userchangepasswordpage', methods=['POST','GET'])
def userchangepasswordpage():
    try:
        msg=""
        if(request.method=="POST"):
            userid = session['userid']
            pwd = request.form["pwd"]
            cpwd = request.form["cpwd"]            
            if(pwd==cpwd):            
                db = firestore.client()
                data_ref = db.collection(u'newuser').document(userid)
                encode = base64.b64encode(pwd.encode("utf-8"))
                data_ref.update({u'Password': encode})
                return render_template("userlogin.html", msg=msg)        
            else:
                msg="Password & Confirm Password are not Matching"
        return render_template("userchangepasswordpage.html", msg=msg)
    except Exception as e:
        return str(e)
    
async def get_plot1(calculated_df):
    plt.title('Vehicle Count Plot')
    sns.histplot(calculated_df['vehicle_count'])
    await asyncio.sleep(1)
    return plt

weatherconditions=["Rainy","Sunny","Cloudy","Snowy"]
seasons=["Autumn","Summer","Winter","Spring"]
regions=["North","South","West","East"]
categories=["Groceries","Toys","Electronics","Furniture","Clothing"]
productids=["P0001","P0002","P0003","P0004","P0005","P0006","P0007","P0008","P0009","P0010",
            "P0011","P0012","P0013","P0014","P0015","P0016","P0017","P0018","P0019","P0020"]
storenames=["S001","S002","S003","S004","S005"]

def generate_mappings(list_data):
    randomlist=[]
    count=0
    while(True):
        if(count==len(list_data)):
            break
        temp=random.randint(1, len(list_data)*100)
        if(temp not in randomlist):
            randomlist.append(temp)
            count+=1
    randomlist_mappings={}
    x=0
    for x in range(0,len(randomlist)):
        randomlist_mappings[list_data[x]] = randomlist[x]
    return randomlist_mappings

"""
@app.route('/usercheckbusiness', methods=['POST','GET'])
def usercheckbusiness():
    try:
        msg=""
        if request.method == 'POST':
            columns=['Date','Store_ID','Product_ID','Category','Region','Inventory_Level',
                       'Units_Sold','Units_Ordered','Demand_Forecast','Price','Discount',
                       'Weather_Condition','Holiday_Promotion','Seasonality',
                       'Competitor_Pricing']
            file_path='retail_store_inventory.csv'
            df= pd.read_csv(file_path, usecols=columns)
            print("Before Mapping : \n",df.head())
            storename = request.form['storename']
            productid = request.form['productid']
            category = request.form['category']
            region = request.form['region']
            inventorylevel = request.form['inventorylevel']
            unitssold = request.form['unitssold']
            unitsordered = request.form['unitsordered']
            demandforecast = request.form['demandforecast']
            price = request.form['price']
            discount = request.form['discount']
            weathercondition = request.form['weathercondition']
            holidaypromotion = request.form['holidaypromotion']
            seasonality = request.form['seasonality']
            # Define a custom mapping dictionary            
            #weatherconditions_mapping = {"Rainy":0, "Sunny":1,"Cloudy":2,"Snowy":3}   
            #df['Weather_Condition'] = df['Weather_Condition'].map(weatherconditions_mapping)            
            #seasons_mapping={"Autumn":0,"Summer":1,"Winter":2,"Spring":3}            
            #regions_mapping={"North":0,"South":1,"West":2,"East":3}            
            #categories_mapping={"Groceries":0,"Toys":1,"Electronics":2,"Furniture":3,"Clothing":4}
            #productids_mappings={"P0001":0,"P0002":1,"P0003":2,"P0004":3,"P0005":4,
            #            "P0006":5,"P0007":6,"P0008":7,"P0009":8,"P0010":9,
            #            "P0011":10,"P0012":11,"P0013":12,"P0014":13,"P0015":14,
            #            "P0016":16,"P0017":16,"P0018":17,"P0019":18,"P0020":19}
            #storenames_mappings={"S001":0,"S002":1,"S003":2,"S004":3,"S005":4}
            
            weatherconditions_mapping = generate_mappings(weatherconditions)
            seasons_mapping=generate_mappings(seasons)
            regions_mapping=generate_mappings(regions)
            categories_mapping=generate_mappings(categories)
            productids_mappings=generate_mappings(productids)
            storenames_mappings=generate_mappings(storenames)
            
            print("After Mapping : \n",df.head())
            # Apply the mapping using the .map() method
            df['Weather_Condition'] = df['Weather_Condition'].map(weatherconditions_mapping)
            df['Seasonality'] = df['Seasonality'].map(seasons_mapping)
            df['Region'] = df['Region'].map(regions_mapping)
            df['Category'] = df['Category'].map(categories_mapping)
            df['Product_ID'] = df['Product_ID'].map(productids_mappings)
            df['Store_ID'] = df['Store_ID'].map(storenames_mappings)
            
            print("After Mapping : \n",df.head())
            print("DF Keys : \n",df.keys())

            print("X Keys : \n", df.iloc[:,0:12].keys())
            print("Y Keys : \n", df.iloc[:,13])

            X = df.iloc[:,0:13].values
            y = df.iloc[:,13].values
            
            # Convert continuous labels to discrete classes
            y = [int(label) for label in y]

            #print(X)
            #print(y)
            
            # 3. Split Data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

            # 4. Train Model
            model = RandomForestClassifier(n_estimators=100, random_state=42) # n_estimators is the number of trees
            model.fit(X_train, y_train)

            print("X Test : ",X_test[0], " Length : ", len(X_test[0]))

            # 5. Make Predictions
            y_pred = model.predict(X_test)
            print("Y Prediction : ", y_pred)
            
            user_test = (np.array([[storenames_mappings[storename],productids_mappings[productid],
                                categories_mapping[category], regions_mapping[region],
                                inventorylevel,unitssold,unitsordered,
                                demandforecast,price,discount,
                                weatherconditions_mapping[weathercondition],
                                holidaypromotion,seasons_mapping[seasonality]]]))
            print("User Test : \n",user_test,  " Length : ", len(user_test[0]))

            y_pred = model.predict(user_test)
            print("Prediction : ", y_pred[0])
            
            msg=f"Based on the Analysis Competitor Prediction Price is {y_pred[0]}"
            
            userid = session['userid']
            id = str(round(time.time()))
            if(holidaypromotion==0):
                holidaypromotion="Holiday"
            else:
                holidaypromotion="Promotion"
            json={"id":id, "UserId":userid,'StoreId':storename,
                  'ProductId':productid, 'Category':category,
                  'Region':region, "InventoryLevel":inventorylevel,
                  "UnitsSold":unitssold, "UnitsOrdered":unitsordered,
                  "DemandForecast":demandforecast, "Price":price,
                  "Discount":discount, "WeatherCondition":weathercondition,
                  "Holiday_Promotion":holidaypromotion,"Result":msg}
            
            db = firestore.client()
            db_ref = db.collection('newreport')
            db_ref.document(id).set(json)
            
        return render_template("usercheckbusiness.html", storenames=storenames,
                               productids=productids, categories=categories,
                               regions=regions, seasons=seasons,
                               weatherconditions=weatherconditions, msg=msg)
    except Exception as e:
        return str(e)
"""
@app.route('/customerviewreports', methods=['POST','GET'])
def customerviewreports():
    try:
        db = firestore.client()
        data_ref = db.collection('newreport')
        newdata = data_ref.get()
        id = int(session['userid'])
        print('UserId : ', id)
        data = []
        for doc in newdata:
            temp = doc.to_dict()
            print("Temp : ", temp)
            if (int(temp['UserId']) == id):
                data.append(doc.to_dict())
        return render_template("customerviewreports.html", data=data)
    except Exception as e:
        return str(e)

@app.route('/')
@app.route('/index')
def index():
    try:
        return render_template("index.html")
    except Exception as e:
        return str(e)

@app.route('/usermainpage')
def usermainpage():
    try:
        return render_template("usermainpage.html")
    except Exception as e:
        return str(e)

@app.route('/logout')
def logoutpage():
    try:
        session['id']=None
        return render_template("index.html")
    except Exception as e:
        return str(e)

@app.route('/about')
def aboutpage():
    try:
        return render_template("about.html")
    except Exception as e:
        return str(e)

@app.route('/services')
def servicespage():
    try:
        return render_template("services.html")
    except Exception as e:
        return str(e)

@app.route('/gallery')
def gallerypage():
    try:
        return render_template("gallery.html")
    except Exception as e:
        return str(e)

@app.route('/adminlogin', methods=['GET','POST'])
def adminloginpage():
    msg=""
    if request.method == 'POST':
        uname = request.form['uname'].lower()
        pwd = request.form['pwd'].lower()
        print("Uname : ", uname, " Pwd : ", pwd)
        if uname == "admin" and pwd == "admin":
            return render_template("adminmainpage.html")
        else:
            msg = "UserName/Password is Invalid"
    return render_template("adminlogin.html", msg=msg)

@app.route('/userlogin', methods=['GET','POST'])
def userlogin():
    msg=""
    if request.method == 'POST':
        uname = request.form['uname']
        pwd = request.form['pwd']
        db = firestore.client()
        dbref = db.collection('newuser')
        userdata = dbref.get()
        data = []
        for doc in userdata:
            print(doc.to_dict())
            print(f'{doc.id} => {doc.to_dict()}')
            data.append(doc.to_dict())
        flag = False
        for temp in data:
            print("Pwd : ", temp['Password'])
            #decMessage = fernet.decrypt(temp['Password']).decode()
            decode = base64.b64decode(temp['Password']).decode("utf-8")
            if uname == temp['UserName'] and pwd == decode:
                session['userid'] = temp['id']
                flag = True
                break
        if (flag):
            return render_template("usermainpage.html")
        else:
            msg = "UserName/Password is Invalid"
    return render_template("userlogin.html", msg=msg)

@app.route('/userviewprofile', methods=['GET','POST'])
def customerviewprofile():
    try:
        id=session['userid']
        db = firestore.client()
        dbref = db.collection('newuser')
        data = dbref.document(id).get().to_dict()
        print("User Data ", data)
        return render_template("userviewprofile.html", data=data)
    except Exception as e:
        return str(e)

@app.route('/adminviewfullreport', methods=['GET','POST'])
def adminviewfullreport():
    try:
        id=request.args['id']
        db = firestore.client()
        dbref = db.collection('newreport')
        data = dbref.document(id).get().to_dict()
        print("Report Data ", data)
        return render_template("adminviewfullreport.html", data=data)
    except Exception as e:
        return str(e)

@app.route('/userviewfullreport', methods=['GET','POST'])
def userviewfullreport():
    try:
        id=request.args['id']
        db = firestore.client()
        dbref = db.collection('newreport')
        data = dbref.document(id).get().to_dict()
        print("Report Data ", data)
        return render_template("userviewfullreport.html", data=data)
    except Exception as e:
        return str(e)

@app.route('/newuser', methods=['POST','GET'])
def newuser():
    try:
        msg=""
        print("Add New User page")
        if request.method == 'POST':
            fname = request.form['fname']
            lname = request.form['lname']
            uname = request.form['uname']
            pwd = request.form['pwd']
            email = request.form['email']
            phnum = request.form['phnum']
            address = request.form['address']
            id = str(round(time.time()))
            #encMessage = fernet.encrypt(pwd.encode())
            
            db = firestore.client()
            dbref = db.collection('newuser')
            userdata = dbref.get()
            is_present=False
            for doc in userdata:
                temp=doc.to_dict()                
                if(temp['UserName']==uname or temp['PhoneNumber']==phnum or temp['EmailId']==email):
                    is_present=True
                    break
            if(not is_present):
                encode = base64.b64encode(pwd.encode("utf-8"))
                #print("str-byte : ", encode)
                json = {'id': id,
                        'FirstName': fname, 'LastName': lname,
                        'UserName': uname, 'Password': encode,
                        'EmailId': email, 'PhoneNumber': phnum,
                        'Address': address}
                db = firestore.client()
                newuser_ref = db.collection('newuser')
                newuser_ref.document(id).set(json)
                print("User User Success")
                msg = "New User Inserted Success"
            else:
                msg = "UserName/Emailid/PhoneNumber already exists"
        return render_template("newuser.html", msg=msg)
    except Exception as e:
        return str(e)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/contact', methods=['POST','GET'])
def contactpage():
    try:
        msg=""
        if request.method == 'POST':
            cname = str(request.form['cname'])# + " " + str(request.form['lname'])
            subject = request.form['subject']
            message = request.form['message']
            email = request.form['email']
            id = str(random.randint(1000, 9999))
            json = {'id': id,
                    'ContactName': cname, 'Subject': subject,
                    'Message': message,
                    'EmailId': email}
            db = firestore.client()
            newdb_ref = db.collection('newcontact')
            id = json['id']
            newdb_ref.document(id).set(json)
            body = "Thank you for contacting us, " + str(cname) + " We will keep in touch with in 24 Hrs"
            receipients = [email]
            send_email(subject,body,recipients=receipients)
            msg = "New Contact Added Success"
        return render_template("contact.html", msg=msg)
    except Exception as e:
        return str(e)

@app.route('/adminviewusers', methods=['POST','GET'])
def adminviewusers():
    try:
        db = firestore.client()
        newdata_ref = db.collection('newuser')
        newdata = newdata_ref.get()
        data=[]
        for doc in newdata:
            data.append(doc.to_dict())
        print("Users Data " , data)
        return render_template("adminviewusers.html", data=data)
    except Exception as e:
        return str(e)

@app.route('/adminviewcontacts', methods=['POST','GET'])
def adminviewcontacts():
    try:
        db = firestore.client()
        newdata_ref = db.collection('newcontact')
        newdata = newdata_ref.get()
        data=[]
        for doc in newdata:
            data.append(doc.to_dict())
        print("Contact Data " , data)
        return render_template("adminviewcontacts.html", data=data)
    except Exception as e:
        return str(e)

@app.route('/adminviewreports', methods=['POST','GET'])
def adminviewreports():
    try:        
        db = firestore.client()
        newdata_ref = db.collection('newreport')
        newdata = newdata_ref.get()
        data, graph_values=[],[]
        for doc in newdata:
            data.append(doc.to_dict())
            temp=doc.to_dict()
        return render_template("adminviewreports.html", data=data)
    except Exception as e:
        return str(e)

@app.route('/userviewreports', methods=['POST','GET'])
def userviewreports():
    try:
        db = firestore.client()
        userid = session['userid']
        newdata_ref = db.collection('newreport')
        newdata = newdata_ref.get()
        data,graph_values=[],[]
        for doc in newdata:
            temp=doc.to_dict()
            if(temp['UserId']==userid):
                data.append(doc.to_dict())
        return render_template("userviewreports.html", data=data)
    except Exception as e:
        return str(e)

@app.route('/adminmainpage')
def adminmainpage():
    try:
        return render_template("adminmainpage.html")
    except Exception as e:
        return str(e)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    app.debug = True
    app.run()