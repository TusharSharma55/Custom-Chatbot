import uvicorn
from fastapi import FastAPI, Body
from models import InputModel
from fastapi.middleware.cors import CORSMiddleware

from final_custom_chatbot import Assistant
from chatbot2 import Assistant2

app = FastAPI(title="Custom Chatbot")
assistant = Assistant()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/user-input")
def receive_user_input(question: InputModel):
    # Store the question in a variable
    user_question = question.question

    answer = assistant.main(user_question)

    return {"answer": answer}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=50388)
    # uvicorn.run(app, host="127.0.0.1", port=50388)
