from fastapi import FastAPI, Body
from models import InputModel
from final_custom_chatbot import Assistant

app = FastAPI()


@app.post("/user-input")
async def receive_user_input(question: InputModel):
    # Store the question in a variable
    user_question = question.question

    # Process the question
    assistant = Assistant()
    answer = assistant.main(user_question)

    return {"answer": answer}