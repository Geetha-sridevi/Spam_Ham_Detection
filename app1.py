
import streamlit as st
import pickle

# Load the saved model
model = pickle.load(open("model101.pkl", "rb"))

# Load the CountVectorizer
cv = pickle.load(open("vectorizer101.pkl", "rb"))

# Streamlit UI
st.title("Spam / Ham Detection")

st.write("Enter a message to check whether it is Spam or Ham.")

# Input message
message = st.text_area("Enter Your Message")

# Prediction button
if st.button("Predict"):

    if message.strip() == "":
        st.warning("Please enter a message")

    else:
        # Convert text into numerical format
        message_vector = cv.transform([message])

        # Predict using Naive Bayes
        prediction = model.predict(message_vector)

        # Display result
        if prediction[0] == "spam":
            st.error("This message is SPAM")

        else:
            st.success("This message is HAM")
