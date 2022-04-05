import imp
from flask import Flask,request, url_for, redirect, render_template, jsonify
import pandas as pd
import pickle
import numpy as np
from os.path import exists
# Initalise the Flask app
app = Flask(__name__)

@app.route('/')
def home():
    if(exists(model)){
        
    }
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
