import nltk
import string
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

# ------------------ FAQs ------------------
# E-Commerce FAQs
faqs = {
    "How can I create an account?": "You can create an account by clicking on the Sign Up button and filling in your details.",
    "How do I reset my password?": "Click on Forgot Password and follow the instructions sent to your email.",
    "What payment methods are accepted?": "We accept credit cards, debit cards, PayPal, and cash on delivery.",
    "Is cash on delivery available?": "Yes, cash on delivery is available in selected locations.",
    "How can I track my order?": "You can track your order from the Orders section in your account.",
    "How long does delivery take?": "Delivery usually takes 3 to 7 business days.",
    "Do you offer international shipping?": "Currently, we only ship within the country.",
    "Can I cancel my order?": "Yes, you can cancel your order before it is shipped.",
    "How do I return a product?": "You can request a return from the Orders page within 7 days of delivery.",
    "What is your return policy?": "Products can be returned within 7 days if they are unused and in original condition.",
    "When will I receive my refund?": "Refunds are processed within 5 to 10 business days.",
    "Are there any delivery charges?": "Delivery charges depend on your location and order value.",
    "Do you offer discounts or promo codes?": "Yes, we regularly offer discounts and promotional codes.",
    "How can I apply a promo code?": "You can apply a promo code during checkout.",
    "What should I do if I receive a damaged product?": "Please contact customer support immediately with photos of the damage.",
    "Can I change my delivery address after ordering?": "Address changes are possible before the order is shipped.",
    "Do you have a mobile app?": "Yes, our mobile app is available on Android and iOS.",
    "How can I contact customer support?": "You can contact customer support via email, phone, or live chat.",
    "What are your customer support hours?": "Customer support is available from 9 AM to 6 PM.",
    "Is my personal information secure?": "Yes, we use encryption and secure servers to protect your data.",
    "Can I order without creating an account?": "Yes, guest checkout is available.",
    "What happens if my order is delayed?": "You will be notified in case of any delay.",
    "Can I reorder a previous purchase?": "Yes, you can reorder from your order history.",
    "Do you provide invoices?": "Yes, invoices are available in your order details.",
    "How can I give feedback?": "You can submit feedback through the feedback section on our website."
}


# ------------------ NLP Functions ------------------
def preprocess(text):
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [
        word for word in tokens
        if word not in stopwords.words('english')
        and word not in string.punctuation
    ]
    return " ".join(tokens)

questions = list(faqs.keys())
processed_questions = [preprocess(q) for q in questions]

vectorizer = TfidfVectorizer()
faq_vectors = vectorizer.fit_transform(processed_questions)

def chatbot_response(user_input):
    user_input = preprocess(user_input)
    user_vector = vectorizer.transform([user_input])

    similarity = cosine_similarity(user_vector, faq_vectors)
    best_match = np.argmax(similarity)

    if similarity[0][best_match] < 0.2:
        return "Sorry, I couldn't understand your question."

    return faqs[questions[best_match]]
