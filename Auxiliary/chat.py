from time import sleep

from Auxiliary.utils import *
from Auxiliary.llm import LLMModel

print("Инициализация модели...")
llm = LLMModel()
print("Модель инициализирована!\n")

temp_messages = {}

# Custom functions for buttons
def delete(_, message_tg: telebot.types.Message):
    Message.botDeleteMessage(message_tg)
    # ничего не возращаем, чтобы дальше шло как с обычными кнопками


def clear(_, message_tg: telebot.types.Message):
    bot.clear_step_handler_by_chat_id(
        message_tg.chat.id)  # просто очищаем step_handler
    # ничего не возращаем, чтобы дальше шло как с обычными кнопками


def delete_clear(*args):
    clear(*args)
    delete(*args)
    # ничего не возращаем, чтобы дальше шло как с обычными кнопками


# Custom functions for messages

# # Question

# # # Ask
def ask_question(message_tg: telebot.types.Message):
    clear(None, message_tg)

    botMessage = message_question_ask.line(message_tg, deleting_message=False)
    bot.register_next_step_handler(botMessage, answer_question(botMessage))
    return True

def ask_question_again(message_tg: telebot.types.Message):
    clear(None, message_tg)

    botMessage = message_question_ask.line(message_tg, deleting_message=False)
    bot.register_next_step_handler(botMessage, answer_question(botMessage))
    return True


# # # Answer
def answer_question(botMessage: telebot.types.Message):
    def wrapper(message_tg: telebot.types.Message):
        nonlocal botMessage
        Message.userSendLogger(message_tg)

        Message.botDeleteMessage(botMessage)
        botMessage = message_question_answer_processing.line(botMessage, deleting_message=False)

        # ML
        try:
            question = message_tg.text
            answer = llm(question)
            embeding = llm.get_embedding(question)

            if not llm.is_question_inappropriate(question):
                operations.record_QnA(question, answer, embeding)
                Message(answer, ((Button("✉️ Отправить на почту ✉️",
                                         f"{question}_{answer.replace('_', '-')}_send"),),)).line(botMessage)
            else:
                custom_close = Button("✖️ Закрыть ✖️", f"close_{message_tg.chat.id}_{message_tg.id}_custom")
                temp_messages[f"{message_tg.chat.id}_{message_tg.id}"] = message_tg
                Message(answer, ((button.question_again,), (custom_close,),)).line(botMessage)

        except Exception as exception:
            print(f"Ошибка: {exception}")

            message_question_answer_error.line(botMessage)

    return wrapper


# # # Send
def send_answer(botMessage: telebot.types.Message, data: list, message_history: telebot.types.Message):
    def wrapper(message_tg: telebot.types.Message):
        nonlocal botMessage, data, message_history
        Message.userSendLogger(message_tg)
        Message.botDeleteMessage(message_tg)

        email = message_tg.text
        question, answer = data

        try:
            html_content = f"""
                <html>
                    <body style="font-family: Arial, sans-serif; line-height: 1.6;">
                        <p>Уважаемый пользователь,</p>
                        <p>Благодарим вас за ваш запрос. Мы получили ваш вопрос:</p>
                        <br>
    
                        <h3 style="color: #333;">Вопрос:</h3>
                        <blockquote style="border-left: 4px solid #ddd; margin-left: 10px; padding-left: 10px;">
                            <strong>{question}</strong>
                        </blockquote>
    
                        <br>
    
                        <h3 style="color: #333;">Ответ:</h3>
                        <blockquote style="border-left: 4px solid #4CAF50; margin-left: 10px; padding-left: 10px; color: #333;">
                            <strong>{answer}</strong>
                        </blockquote>
    
                        <br>
                        <p>С уважением,</p>
                        <p>Команда службы поддержки РЖД</p>
                    </body>
                </html>
            """
            botMessage = message_email_processing.line(botMessage)
            send_email(email, "Ответ от службы поддержки РЖД", html_content)

            botMessage = message_email_success.line(botMessage)
            sleep(3)
            Message.botDeleteMessage(botMessage)
        except Exception as exception:
            print(f"Ошибка: {exception}")

            message_email_error.line(botMessage)
        else:
            Message(f"{answer}\n\n"
                    f"<i>Ответ был отправлен на email: <u>{email}</u></i>").line(message_history)

    return wrapper


# Buttons
button = Button('', '')

# Question
Button("📝 Задать заново 📝", "question_again")
Button("❔ Задать вопрос ❔", "ask_question")

# Cancel / close
Button("✖️ Отменить ✖️", "cancel", func=delete_clear)
Button("✖️ Закрыть ✖️", "close", func=delete)

# Messages

# Start
message_start = Message("<b>Привет <USERNAME>, это бот Паравозик 🚂!</b>\n\n"
                        "<i>Используй <u>команду в меню</u> или <u>кнопку под сообщением</u> чтобы задать вопрос</i>\n"
                        "<i><b>Ответ на вопрос вы можете получить на почту</b></i>\n", ((button.ask_question,),))

# Question

# # Ask question
message_question_ask = Message("<b>Напишите пожалуйста вопрос одним сообщением:</b>",
                               ((button.cancel,),), button.ask_question,
                               func=ask_question)

message_question_ask_again = Message("<b>Напишите пожалуйста вопрос связанный с <u>«РЖД»</u> одним сообщением:</b>",
                                     ((button.cancel,),), button.question_again,
                                     func=ask_question_again)

# # Processing
message_question_answer_processing = Message("<b>Готовим ответ...</b>")

# # Error
message_question_answer_error = Message("<b>Ошибка генерации.</b>", ((button.close,),))

# Email

# # Get
message_email_get = Message("<b>Напишите пожалуйста <u>email</u> пользователя</b>",
                            ((button.cancel,),))

# # Sending
message_email_processing = Message("<b>Отправляем...</b>")

# # Error
message_email_error = Message("<b>Email не найден!</b>",
                              ((button.close,),))

# # Success
message_email_success = Message(f"<b>Сообщение отправлено!</b>")
