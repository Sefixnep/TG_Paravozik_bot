from Auxiliary.chat import *


@bot.message_handler(commands=["start"])
def start(message_tg: telebot.types.Message):
    Message.userSendLogger(message_tg)
    message_start.line(message_tg)


@bot.message_handler(commands=["question"])
def question(message_tg: telebot.types.Message):
    Message.userSendLogger(message_tg)
    ask_question(message_tg)


@bot.callback_query_handler(func=lambda call: True)
def callback_reception(call: telebot.types.CallbackQuery):
    if call.data not in button.callback_data:  # Если кнопка не найдена (скорее всего из-за перезапуска системы)
        start(call.message)
        return None

    data = button.callback_data[call.data]
    commands = ['send']

    to_message = None
    from_button = button.get_instance(call.data)

    if from_button:
        to_message = from_button(call.message)

    for command in commands:  # Для кастомной обработки
        if data.split('_')[-1] == command:
            command_data = data.split('_')[:-1]  # Данные передавающиеся кнопкой

            if command == 'send':
                clear(None, call.message)
                botMessage = message_email_get.line(call.message, deleting_message=False)
                bot.register_next_step_handler(botMessage, send_answer(botMessage, command_data, call.message))

            break
    else:
        if to_message is not None and to_message(
                call.message) is None:  # Вызываем функцию, если там нет return, то делаем old_line
            to_message.line(call.message)  # Выводить сообщение к которому ведет кнопка

    bot.answer_callback_query(callback_query_id=call.id, show_alert=False)


@bot.message_handler(content_types=['text'])
def watch(message_tg: telebot.types.Message):
    Message.userSendLogger(message_tg)


if __name__ == '__main__':
    print(f"Version: {config.version}")
    print(f"link: https://t.me/{config.Bot}")
    logger.info(f'{config.Bot} start')

bot.infinity_polling()  # logger_level=None
