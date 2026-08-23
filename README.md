# 🛡️ Email & SMS Spam Detector using Naive Bayes

An end-to-end Machine Learning web application built with **Python**, **Scikit-Learn**, and **Streamlit** that classifies text messages and emails as **Spam** or **Ham (Safe)** in real-time with confidence scores.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.2%2B-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

---

## 🌟 Key Features

- ⚡ **Real-Time Classification:** Instant prediction of whether an email or SMS message is Spam or Safe (Ham).
- 📊 **Confidence Metrics:** Displays exact statistical percentage confidence for predictions.
- 🌙 **Dark & Light Mode:** Toggleable theme mode via the sidebar settings for a modern UI experience.
- 💡 **One-Click Examples:** Built-in preset test buttons to quickly test Spam and Safe email samples.
- 📦 **Pre-trained Models:** Includes pre-packaged `vectorizer.pkl` and `spam_model.pkl` so users can launch the web app immediately without re-training.

---

## 📂 Project Structure

```text
├── app.py                                         # Interactive Streamlit Web Interface
├── Spam_Classification_using_Naive_Bayes.ipynb    # Training notebook & EDA
├── emails.csv                                     # Dataset containing 5,700+ email messages
├── vectorizer.pkl                                 # Pre-trained CountVectorizer vocabulary
├── spam_model.pkl                                 # Pre-trained Multinomial Naive Bayes classifier
├── requirements.txt                               # Python dependencies list
└── README.md                                      # Project documentation
```

---

## 🛠️ How it Works

1. **Text Vectorization:** Raw input message text is transformed into a bag-of-words count matrix using `CountVectorizer`.
2. **Naive Bayes Classification:** The vectorized text is passed to a `Multinomial Naive Bayes` classifier trained on labeled email datasets.
3. **Probability Calculation:** Bayes' theorem calculates the posterior probability of the message being **Spam** vs **Ham**.
4. **UI Output:** Streamlit displays visual alerts (`Alert: SPAM` or `Clear: HAM`) along with calculated confidence metrics.

---

## 🚀 Local Installation & Quick Start

Follow these simple steps to run this application on your local machine:

### 1. Clone the Repository
```bash
git clone https://github.com/kumarjesh/Naive_Bayes_Classifier.git
cd Naive_Bayes_Classifier
```

### 2. Create and Activate a Virtual Environment
- **Windows:**
  ```powershell
  python -m venv venv
  .\venv\Scripts\activate
  ```
- **macOS / Linux:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

### 3. Install Required Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit Web Application
```bash
streamlit run app.py
```

The app will open automatically in your web browser at **`http://localhost:8501`**!

---

## 🌐 Deployment Instructions

### Option 1: Streamlit Community Cloud (Recommended - Free & 1-Click)
1. Push your repository to **GitHub**.
2. Visit [share.streamlit.io](https://share.streamlit.io/) and sign in with your GitHub account.
3. Click **New app**, select your repository, set the main file path to `app.py`, and click **Deploy**.

### Option 2: Deploying to Vercel / Render / Hugging Face
- **Hugging Face Spaces:** Create a new Space with the **Streamlit** SDK and push this repo.
- **Render:** Deploy as a **Web Service** with start command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`.

---

## 🛠️ Technologies Used

- **Language:** Python 3.9+
- **Machine Learning:** Scikit-Learn (`MultinomialNB`, `CountVectorizer`)
- **Web Framework:** Streamlit
- **Data Handling:** Pandas, NumPy, Joblib
- **PDF Export:** ReportLab

---

## 🤝 Connect with Me

If you found this project helpful, give it a ⭐ on GitHub and connect with me on LinkedIn!

- **GitHub:** [@kumarjesh](https://github.com/kumarjesh)
- **LinkedIn:** [LinkedIn Profile](https://linkedin.com/in/)

---
*Built with ❤️ using Python & Streamlit.*
