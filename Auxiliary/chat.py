from Auxiliary.utils import *


# Custom functions for buttons
def delete(_, message_tg):
    Message.botDeleteMessage(message_tg)
    # ничего не возращаем, чтобы дальше шло как с обычными кнопками


def clear(_, message_tg):
    bot.clear_step_handler_by_chat_id(
        message_tg.chat.id)  # просто очищаем step_handler
    # ничего не возращаем, чтобы дальше шло как с обычными кнопками


def delete_clear(*args):
    clear(*args)
    delete(*args)
    # ничего не возращаем, чтобы дальше шло как с обычными кнопками


# Custom functions for messages
def ask_question(message_tg):
    clear(None, message_tg)
    Message.botDeleteMessage(message_tg)

    botMessage = message_question_ask.line(message_tg, deleting_message=False)
    bot.register_next_step_handler(botMessage, answer_question(botMessage))
    return True


def answer_question(botMessage):
    def wrapper(message_tg):
        nonlocal botMessage
        Message.userSendLogger(message_tg)
        Message.botDeleteMessage(message_tg)

        botMessage = message_question_answer_processing.line(botMessage)

        # ML
        try:
            is_question = True

            if is_question:
                question = message_tg.text
                answer = 'ANSWER_MODEL'

                Message(get_text_question_answer(question, answer),
                        ((Button("Отправить", f"{question}_{answer.replace('_', '-')}_send"),),)).line(botMessage)

        except Exception as exception:
            print(f"Ошибка: {exception}")

            message_question_answer_error.line(botMessage)

    return wrapper

def send_answer(botMessage, data, message_history):
    def wrapper(message_tg):
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

            message = Message(f"<b>Сообщение на почту {email} отправлено!</b>", ((button.close,),))
            message.line(botMessage)
        except Exception as exception:
            print(f"Ошибка: {exception}")

            message_email_error.line(botMessage)
        else:
            Message(f"{get_text_question_answer(*data)}\n\n"
                              f"<i>Ответ был отправлен на email: <u>{email}</u></i>",
                              ((button.close,),)).line(message_history)

    return wrapper


# Buttons
button = Button('', '')

# Cancel / close
Button("✖️ Отменить ✖️", "cancel_question", func=delete_clear)
Button("✖️ Закрыть ✖️", "close", func=delete)

# Messages

# Start
message_start = Message("<b>Привет, это бот Паравозик!</b>\n\n"
                        "<i>Ты можешь задать мне вопрос, для этого <b>используй команду в меню</b> (слева снизу)</i>\n"
                        "<i>Ответ на вопрос вы можете отправить на почту!</i>")

# Question

# # Ask question
message_question_ask = Message("<b>Напишите пожалуйста вопрос одним сообщением:</b>",
                               ((button.cancel_question,),),
                               func=ask_question)

# # Processing
message_question_answer_processing = Message("<b>Готовим ответ...</b>")

# # Error
message_question_answer_error = Message("<b>Ошибка генерации.</b>", ((button.close,),))

# Email

# # Get
message_email_get = Message("<b>Напишите пожалуйста <u>email</u> пользователя</b>",
                            ((button.cancel_question,),))

# # Sending
message_email_processing = Message("<b>Отправляем...</b>")

# # Error
message_email_error = Message("<b>Email не найден!</b>",
                              ((button.close,),))
