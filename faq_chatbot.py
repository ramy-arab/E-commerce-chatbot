from chatbot_logic import chatbot_response

print("🤖 FAQ Chatbot (Type 'exit' to quit)")
while True:
    user_query = input("You: ")
    if user_query.lower() == "exit":
        print("Chatbot: Goodbye!")
        break
    response = chatbot_response(user_query)
    print("Chatbot:", response)
