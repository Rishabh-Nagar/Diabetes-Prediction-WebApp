import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import matplotlib
from manage_db import *
import hashlib
import lime
import lime.lime_tabular
matplotlib.use('Agg')
import PIL
from PIL import Image
import sklearn
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score

# Password
def generate_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()


def verify_hashes(password, hashed_text):
    if generate_hashes(password) == hashed_text:
        return hashed_text
    return False

feature_names = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']


def load_model(model_file):
    loaded_model = joblib.load(open(os.path.join(model_file),"rb"))
    return loaded_model

def get_value(val,my_dict):
	for key,value in my_dict.items():
		if val == key:
			return value 

def get_key(val,my_dict):
	for key,value in my_dict.items():
		if val == key:
			return key


html_temp = """
		<div style="background-color:{};padding:10px;border-radius:10px">
		<h1 style="color:white;text-align:center;">Muliebrous Diabetes Prediction </h1>
		</div>
		"""

# Avatar Image using a url
avatar1 ="https://www.w3schools.com/howto/img_avatar1.png"
avatar2 ="https://www.w3schools.com/howto/img_avatar2.png"

prescriptive_message_temp ="""
	<div style="background-color:silver;overflow-x: auto; padding:10px;border-radius:5px;margin:10px;">
		<h3 style="text-align:justify;color:black;padding:10px">Recommended Life style modification</h3>
		<ul>
		<li style="text-align:justify;color:black;padding:10px">Exercise Daily</li>
		<li style="text-align:justify;color:black;padding:10px">Manage your carb intake</li>
		<li style="text-align:justify;color:black;padding:10px">Increase your fiber intake</li>
		<li style="text-align:justify;color:black;padding:10px">Drink water and stay hydrated</li>
		<li style="text-align:justify;color:black;padding:10px">Monitor your blood sugar levels</li>
        <li style="text-align:justify;color:black;padding:10px">Manage stress levels</li>
		<ul>
		<h3 style="text-align:justify;color:black;padding:10px">Medical Mgmt</h3>
		<ul>
		<li style="text-align:justify;color:black;padding:10px">Consult your doctor</li>
		<li style="text-align:justify;color:black;padding:10px">Take your interferons</li>
		<li style="text-align:justify;color:black;padding:10px">Go for checkups</li>
		<ul>
	</div>
	"""


descriptive_message_temp ="""
	<div style="background-color:grey;overflow-x: auto; padding:10px;border-radius:5px;margin:10px;">
		<h3 style="text-align:justify;color:pink;padding:10px">Definition</h3>
		<p>A Disease in which the body’s ability to produce or respond to the hormone insulin is impaired, resulting in abnormal metabolism of carbohydrates and elevated levels of glucose in the blood.</p>
	</div>
	"""

@st.cache
def load_image(img):
	im =Image.open(os.path.join(img))

	return im

def change_avatar(sex):
	if sex == "male":
		avatar_img = 'img_avatar.png'
	else:
		avatar_img = 'img_avatar2.png'
	return avatar_img


def main():
#st.title("Malebrous Diabetes Prediction App")
    st.markdown(html_temp.format('royalblue'),unsafe_allow_html=True)
    menu =["Home", "Login", "Sign Up"]
    submenu = ["Plot", "Prediction", "Metrics"]

    choice = st.sidebar.selectbox("Menu", menu)
    if choice == "Home":
        st.subheader("Home")
        st.text("What is Diabetes?")
        st.markdown(descriptive_message_temp.format('navyblue'), unsafe_allow_html=True)
        st.image(load_image('diabetes.jpeg'))

    elif choice == "Login":
        username = st.sidebar.text_input("Username")
        password = st.sidebar.text_input("Password", type = 'password')
        if st.sidebar.checkbox("Login"):
            create_usertable()
            hashed_pswd = generate_hashes(password)
            result = login_user(username, verify_hashes(password,hashed_pswd))
            #if password == "12345":
            st.success("Welcome {}".format(username))

            activity = st.selectbox("Actvity", submenu)
            if activity == "Plot":
                st.subheader("Data Vs Plot")
                df = pd.read_csv("data/diabetes.csv")
                st.dataframe(df)

                st.set_option('deprecation.showPyplotGlobalUse', False)
                st.write("Count Plot For Outcome")
                df['Outcome'].value_counts().plot(kind = 'bar')
                st.pyplot()

                if st.checkbox("Area Chart"):
                    all_columns = df.columns.to_list()
                    feat_choices = st.multiselect("Choose a Feature", all_columns)
                    new_df = df[feat_choices]
                    st.area_chart(new_df)


            elif activity == "Prediction":
                st.subheader("Prediction Analytics")
                age = st.number_input("Age", 10, 80)

                pregnancies = st.slider('No of Pregnancies', 0, 20)
                glucose = st.number_input('Glucose', 0,200, 120 )
                blood_pressure = st.number_input('Blood Pressure', 0,122, 70 )
                skinthickness = st.number_input('Skin Thickness', 0,100, 20 )
                insulin = st.number_input('Insulin', 0,846, 79 )
                bmi = st.number_input('BMI', 0.0, 67.0, 20.0)
                diabetes_ped_func = st.slider('Diabetes Pedigree Function', 0.0,3.0, 0.2 )
                feature_list = [pregnancies, glucose, blood_pressure, skinthickness, insulin, bmi, diabetes_ped_func, age]
                st.write(feature_list)
                pretty_result = {"Pregnancies" : pregnancies, "Glucose" : glucose, "Blood Pressure" : blood_pressure, "Skin Thickness" : skinthickness, "Insulin" : insulin, "BMI" : bmi, "Diabetes Pedigree Function" : diabetes_ped_func, "Age" : age}
                st.json(pretty_result)
                single_sample = np.array(feature_list).reshape(1,-1)
                st.write(single_sample)

                #models
                model_choice = st.selectbox("Select Model",["Logistic Regression","KNN","Decision Tree","Random Forest", "SVM", "Naive Bayes"])
                if st.button("Predict"):
                    if model_choice == "Logistic Regression":
                        loaded_model = load_model("models/classifier_log.pkl")
                        prediction = loaded_model.predict(single_sample)
                        pred_prob = loaded_model.predict_proba(single_sample)
                    
                    elif model_choice == "Random Forest":
                        loaded_model = load_model("models/classifier_random.pkl")
                        prediction = loaded_model.predict(single_sample)
                        pred_prob = loaded_model.predict_proba(single_sample)

                    elif model_choice == "Decision Tree":
                        loaded_model = load_model("models/classifier_tree.pkl")
                        prediction = loaded_model.predict(single_sample)
                        pred_prob = loaded_model.predict_proba(single_sample)

                    elif model_choice == "SVM":
                        loaded_model = load_model("models/classifier_svm.pkl")
                        prediction = loaded_model.predict(single_sample)
                        pred_prob = loaded_model.predict_proba(single_sample)

                    elif model_choice == "KNN":
                        loaded_model = load_model("models/classifier_knn.pkl")
                        prediction = loaded_model.predict(single_sample)
                        pred_prob = loaded_model.predict_proba(single_sample)

                    else:
                        loaded_model = load_model("models/classifier_naive.pkl")
                        prediction = loaded_model.predict(single_sample)
                        pred_prob = loaded_model.predict_proba(single_sample)

                    st.write(prediction)
                    if prediction == 1:
                        st.warning("You have a new best friend. You have Diabetes.")
                        pred_probability_score = {"Not Diabetic":pred_prob[0][0]*100,"Diabetic":pred_prob[0][1]*100}
                        st.subheader("Prediction Probability Score using {}".format(model_choice))
                        st.json(pred_probability_score)
                        st.subheader("Prescriptive Analytics")
                        st.markdown(prescriptive_message_temp,unsafe_allow_html=True)
							
                    else:
                        st.success("You don't look like you have Diabetes.")
                        pred_probability_score = {"Not Diabetic":pred_prob[0][0]*100,"Diabetic":pred_prob[0][1]*100}
                        st.subheader("Prediction Probability Score using {}".format(model_choice))
                        st.json(pred_probability_score)


                if st.checkbox("Interpret"):
                    if model_choice == "Logistic Regression":
                        loaded_model = load_model("models/classifier_log.pkl")
                        

                        df = pd.read_csv("data/diabetes.csv")
                        x = df[['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']]
                        feature_names = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']
                        class_names = ['Not Diabetic(0)','Diabetic(1)']
                        explainer = lime.lime_tabular.LimeTabularExplainer(x.values,feature_names=feature_names, class_names=class_names,discretize_continuous=True)
                        # The Explainer Instance
                        exp = explainer.explain_instance(np.array(feature_list), loaded_model.predict_proba,num_features=9, top_labels=1)
                        exp.show_in_notebook(show_table=True, show_all=False)
                        # exp.save_to_file('lime_oi.html')
                        st.write(exp.as_list())
                        new_exp = exp.as_list()
                        label_limits = [i[0] for i in new_exp]
                        # st.write(label_limits)
                        label_scores = [i[1] for i in new_exp]
                        plt.barh(label_limits,label_scores)
                        st.pyplot()
                        plt.figure(figsize=(20,10))
                        fig = exp.as_pyplot_figure()
                        st.pyplot()

            else:
                st.title("Predicting Diabetes Web App")
                st.sidebar.title("Model Selection Panel")
                st.markdown("Affected by Diabetes or not ?")
                st.sidebar.markdown("Choose your model and its parameters")
                #@st.cache(allow_output_mutation=True)
                @st.cache(persist=True)
                def load_data():# Function to load our dataset
                    data = pd.read_csv("data/diabetes.csv")
                    return data
            
                def split(df):# Split the data to ‘train and test’ sets
                    req_cols = ['Pregnancies', 'Insulin', 'BMI', 'Age','Glucose','BloodPressure','DiabetesPedigreeFunction']
                    x = df[req_cols] # Features for our algorithm
                    y = df.Outcome
                    x = df.drop(columns=['Outcome'])
                    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
                    return x_train, x_test, y_train, y_test
            
                def plot_metrics(metrics_list):
                    if 'Confusion Matrix' in metrics_list:
                        st.subheader("Confusion Matrix")
                        plot_confusion_matrix(model, x_test, y_test, display_labels=class_names)
                        st.pyplot()
                    if 'ROC Curve' in metrics_list:
                        st.subheader('ROC Curve')
                        plot_roc_curve(model, x_test, y_test)
                        st.pyplot()

                    if 'Precision-Recall Curve' in metrics_list:
                        st.subheader('Precision-Recall Curve')
                        plot_precision_recall_curve(model, x_test, y_test)
                        st.pyplot() 

                df=load_data() 
                class_names = ['Diabetec', 'Non-Diabetic']
                x_train, x_test, y_train, y_test = split(df)



                st.sidebar.subheader("Select your Classifier")
                classifier = st.sidebar.selectbox("Classifier", ("Decision Tree", "Logistic Regression", "Random Forest"))
                if classifier == 'Decision Tree':
                    st.sidebar.subheader("Model parameters")
                    #choose parameters

                    criterion= st.sidebar.radio("Criterion(measures the quality of split)", ("gini", "entropy"), key='criterion')
                    splitter = st.sidebar.radio("Splitter (How to split at each node?", ("best", "random"), key='splitter')

                    metrics = st.sidebar.multiselect("Select your metrics : ", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

                    if st.sidebar.button("Classify", key='classify'):
                        st.subheader("Decision Tree Results")
                        model = DecisionTreeClassifier(criterion=criterion, splitter=splitter)
                        model.fit(x_train, y_train)
                        accuracy = model.score(x_test, y_test)
                        y_pred = model.predict(x_test)
                        st.write("Accuracy: ", accuracy.round(2)*100,"%")
                        st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
                        st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
                        plot_metrics(metrics)

                if classifier == 'Logistic Regression':
                    st.sidebar.subheader("Model Parameters")
                    C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key='C_LR')
                    max_iter = st.sidebar.slider("Maximum number of iterations", 100, 500, key='max_iter')
                    metrics = st.sidebar.multiselect("Select your metrics?", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))
                    if st.sidebar.button("Classify", key='classify'):
                        st.subheader("Logistic Regression Results")
                        model = LogisticRegression(C=C, penalty='l2', max_iter=max_iter)
                        model.fit(x_train, y_train)
                        accuracy = model.score(x_test, y_test)
                        y_pred = model.predict(x_test)
                        st.write("Accuracy: ", accuracy.round(2)*100,"%")
                        st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
                        st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
                        plot_metrics(metrics)

                if classifier == 'Random Forest':
                    st.sidebar.subheader("Model Hyperparameters")
                    n_estimators = st.sidebar.number_input("The number of trees in the forest", 100, 5000, step=10, key='n_estimators')
                    max_depth = st.sidebar.number_input("The maximum depth of the tree", 1, 20, step=1, key='n_estimators')
                    bootstrap = st.sidebar.radio("Bootstrap samples when building trees", ('True', 'False'), key='bootstrap')
                    metrics = st.sidebar.multiselect("What metrics to plot?", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))
                    if st.sidebar.button("Classify", key='classify'):
                        st.subheader("Random Forest Results")
                        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, bootstrap=bootstrap, n_jobs=-1)
                        model.fit(x_train, y_train)
                        accuracy = model.score(x_test, y_test)
                        y_pred = model.predict(x_test)
                        st.write("Accuracy: ", accuracy.round(2)*100,"%")
                        st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
                        st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
                        plot_metrics(metrics)

                if st.sidebar.checkbox("Show raw data", False):
                    st.subheader("Diabetes Raw Dataset")
                    st.write(df) 
    
 



        else:
            st.warning("Incorrect Username/Password")

    elif choice == "Sign Up":
        new_username = st.text_input("User Name")
        new_password = st.text_input("Password", type = "password")

        confirm_password = st.text_input("Confirm Password", type = "password")
        if new_password == confirm_password:
            st.success("Password Confirmed")

        else:
            st.warning("Password is not the same")

        if st.button("Submit"):
            create_usertable()
            hashed_new_password = generate_hashes(new_password)
            add_userdata(new_username, hashed_new_password)
            st.success("You Have Successfully created a new account")
            st.info("Login to Get Started!!")
            



if __name__ == "__main__":
    main()