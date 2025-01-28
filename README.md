# Heart Disease Prediction with UI  

## Overview  
This project is a full-stack web application that leverages machine learning to predict the risk of heart disease. It uses a web-based frontend with a TensorFlow.js-powered machine learning model to provide real-time predictions based on user input.  

## Features  
- Heart disease prediction using an Artificial Neural Network (ANN).  
- Web-based interface built with HTML, CSS, JavaScript, and Bootstrap.  
- Client-side prediction using TensorFlow.js for fast, real-time analysis.  
- Scalable and modular architecture for future expansion.  

## Project Architecture  
- **Frontend:** HTML5, CSS3, JavaScript, Bootstrap  
- **Backend (optional):** Node.js for user management and database integration  
- **Machine Learning Model:** TensorFlow.js with a pre-trained ANN model converted from Keras  

## System Requirements  
- Modern web browser (Chrome, Firefox, Edge)  
- Node.js (optional for backend features)  
- TensorFlow.js for machine learning operations  

## Installation and Setup  
1. Clone this repository:  
   ```sh
   git clone https://github.com/Hari-haranK/Heart-Disease-Prediction-with-UI.git
   cd Heart-Disease-Prediction-with-UI
   ```  
2. Install dependencies (for backend integration):  
   ```sh
   npm install
   ```  
3. Start the development server:  
   ```sh
   npm start
   ```  
4. Open `index.html` in a web browser for frontend-only execution.  

## Usage  
1. Enter clinical data such as age, blood pressure, cholesterol, and other parameters.  
2. Click the "Analyze" button to get a prediction.  
3. The result displays the likelihood of heart disease risk.  

## Algorithm Used  
- Artificial Neural Network (ANN)  
- Input layer with 13 features  
- Two hidden layers with ReLU activation and dropout regularization  
- Output layer with sigmoid activation for binary classification  

## Future Enhancements  
- Integration with databases for storing user inputs and results.  
- Authentication system for user accounts.  
- Expanding the model to predict other diseases.  
- Improving accuracy with advanced deep learning techniques.  
