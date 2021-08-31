## Summary of Project
- This project contains data sets containing real messages that were sent during disaster events and disaster categories. This project will based on the real messages and corresponding disaster categories build and export a machine learning model to classify these events so that you can send the messages to an appropriate disaster relief agency.

- This project includes a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display three visualizations of the data, including "Distribution of Message Genres", "Distribution of Response Categories for all message genre", and "Distribution of Response Categories for genre direct". 

## Repository Structure: 
- app
| - template
| |- master.html  # main page of web app
| |- go.html  # classification result page of web app
| - run.py  # Flask file that runs app

- data
|- disaster_categories.csv  # data to process, this file contains all the disaster categories
|- disaster_messages.csv  # data to process, this file contains real messages that were sent during disaster events
|- process_data.py # python script to merge and clean disaster_categories.csv and disaster_messages.csv
|- InsertDatabaseName.db   # database to save clean data to

- models
|- train_classifier.py # python script to train and export a classification model 
|- classifier.pkl  # saved model 

- README.md 

## Run python scripts and app 
After downloading the project from GitHub to your local computer, open Anaconda Prompt, and navigate into the project directory. 
##### Step 1: 
If you are using Windows (just change backslash to slash if in MacOS), then type 
`python data\process_data.py data\disaster_messages.csv data\disaster_categories.csv data\DisasterResponse.db`. Hit enter.
This step will clean data and save cleaned data to DisasterResponse.db database 
##### Step 2: 
Type `python models\train_classifier.py data\DisasterResponse.db models\classifier.pkl`. Hit enter. 
This step will read cleaned data from DisasterResponse.db, build a classifier model, export classifier performance score, and save built model to classifier.pkl 
##### Step 3 (run app):
Navigate to app directory
Type `python run.py`. Hit enter.
The app will start running. Once the line "Running on http://0.0.0.0:3001/ (Press CTRO+C to quit)" shows up, open a new brower page and type in "http://localhost:3001" in the address line to see the app webpage. 