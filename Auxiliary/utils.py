from random import randint
from typing import Callable, BinaryIO
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import smtplib
import telebot
from loguru import logger

from Auxiliary import config
from Auxiliary.DataBase import operations

bot = telebot.TeleBot(config.BOT_TOKEN, parse_mode='html')


class Message:
    def __init__(self, text: str, buttons: tuple | None = None, *from_buttons, photo: BinaryIO | None = None,
                 func: Callable = lambda *args: None):
        self.__text = text  # Текст сообщения
        self.__photo = photo  # В виде open(path, 'rb')
        self.__buttons = buttons  # Двумерный кортеж с кнопками в виде InlineKeyboardButton
        self.__board_tg = None  # Клавиатура кнопок под сообщением: InlineKeyboardMarkup
        if buttons:
            self.__board_tg = telebot.types.InlineKeyboardMarkup()
            for row in (map(lambda x: x.button_tg, buttons1D) for buttons1D in buttons):
                self.__board_tg.row(*row)
        for from_button in from_buttons:  # Кнопки, которые ведут к этому сообщению
            from_button.to_messages += (self,)
        self.__func = func  # Функция, которая должна происходить при вызове сообщения

    def __repr__(self):
        return self.__text

    def __call__(self, *args):
        return self.__func(*args)

    def __getitem__(self, item: tuple):
        return self.__buttons[item[0]][item[1]]

    def line(self, message_tg: telebot.types.Message, deleting_message: bool = True):
        if self.__photo is not None:
            if deleting_message:
                self.botDeleteMessage(message_tg)
            return self.__botSendMessage(message_tg)
        else:
            if deleting_message:
                return self.__botEditMessage(message_tg)
            else:
                return self.__botSendMessage(message_tg)

    @staticmethod
    def __trueText(text: str, message_tg: telebot.types.Message):
        decryption = {
            "<ID>": str(message_tg.chat.id),
            "<USERNAME>": str(message_tg.chat.username) if message_tg.chat.username else "User",
        }

        for key, value in decryption.items():
            text = text.replace(key, value)

        return text

    @staticmethod
    def userSendLogger(message_tg: telebot.types.Message, text: str | None = None):
        if text is None:
            if '\n' in message_tg.text:
                logger.info(f'{message_tg.from_user.username} ({message_tg.chat.id}): \n{message_tg.text}')
            else:
                logger.info(f'{message_tg.from_user.username} ({message_tg.chat.id}): {message_tg.text}')
        else:
            if '\n' in text:
                logger.info(f'{message_tg.chat.username} ({message_tg.chat.id}): \n{text}')
            else:
                logger.info(f'{message_tg.chat.username} ({message_tg.chat.id}): {text}')

    def __botSendMessage(self, message_tg: telebot.types.Message, parse_mode: str = 'HTML', indent: int = 3):
        text = self.__trueText(self.__text, message_tg)
        botMessage = bot.send_message(chat_id=message_tg.chat.id, text=text,
                                      reply_markup=self.__board_tg, parse_mode=parse_mode) \
            if self.__photo is None else bot.send_photo(
            chat_id=message_tg.chat.id, photo=self.__photo, caption=text,
            reply_markup=self.__board_tg, parse_mode=parse_mode)

        if self.__board_tg is None:
            if '\n' in text:
                logger.info(f"{config.Bot} ({botMessage.chat.username}, {message_tg.chat.id}):\n{text}\n")
            else:
                logger.info(f"{config.Bot} ({botMessage.chat.username}, {message_tg.chat.id}): {text}")
        else:
            reply_markup_text = ''
            for reply_markup1 in botMessage.json['reply_markup']['inline_keyboard']:

                for reply_markup2 in reply_markup1:
                    reply_markup_text += f'[{reply_markup2["text"]}]' + (' ' * indent)
                reply_markup_text = reply_markup_text[:-indent]

                reply_markup_text += '\n'
            reply_markup_text = reply_markup_text[:-1]
            logger.info(
                f"{config.Bot} ({botMessage.chat.username}, {message_tg.chat.id}):\n{text}\n{reply_markup_text}\n")
        return botMessage

    def __botEditMessage(self, message_tg: telebot.types.Message, parse_mode: str = 'HTML', indent: int = 3):
        text = self.__trueText(self.__text, message_tg)
        try:
            botMessage = bot.edit_message_text(chat_id=message_tg.chat.id, message_id=message_tg.id, text=text,
                                               reply_markup=self.__board_tg,
                                               parse_mode=parse_mode)
        except:
            botMessage = bot.send_message(chat_id=message_tg.chat.id, text=text,
                                          reply_markup=self.__board_tg, parse_mode=parse_mode)
            self.botDeleteMessage(message_tg)

        if self.__board_tg is None:
            if '\n' in text:
                logger.info(f"{config.Bot} ({botMessage.chat.username}, {message_tg.chat.id}):\n{text}\n")
            else:
                logger.info(f"{config.Bot} ({botMessage.chat.username}, {message_tg.chat.id}): {text}")
        else:
            reply_markup_text = ''
            for reply_markup1 in botMessage.json['reply_markup']['inline_keyboard']:

                for reply_markup2 in reply_markup1:
                    reply_markup_text += f'[{reply_markup2["text"]}]' + (' ' * indent)
                reply_markup_text = reply_markup_text[:-indent]

                reply_markup_text += '\n'
            reply_markup_text = reply_markup_text[:-1]
            logger.info(
                f"{config.Bot} ({botMessage.chat.username}, {message_tg.chat.id}):\n{text}\n{reply_markup_text}\n")
        return botMessage

    @staticmethod
    def botDeleteMessage(message_tg: telebot.types.Message):
        try:
            bot.delete_message(message_tg.chat.id, message_tg.id)
        except:
            pass


class Button:
    instances = list()  # Список со всеми объектами класса
    callback_data = dict()  # Словарь для хранения callback_data: data

    def __init__(self, text: str, data: str, *to_messages: Message, is_link: bool = False,
                 func: Callable = lambda to_messages, message_tg: None):
        self.text = text  # текст кнопки
        if is_link:  # Если кнопка - ссылка
            self.button_tg = telebot.types.InlineKeyboardButton(
                self.text, url=data)  # кнопка в виде объекта InlineKeyboardButton
        else:
            instance = self.__getattr__(data)

            if instance is not None:
                callback = instance.callback
                self.instances.remove(instance)
            else:
                callback = self.create_callback(data)
                self.callback_data[callback] = data
            self.instances.append(self)

            self.callback = callback  # Скрытые (уникальные) данные, несущиеся кнопкой
            self.button_tg = telebot.types.InlineKeyboardButton(
                self.text, callback_data=self.callback)  # кнопка в виде объекта InlineKeyboardButton
            self.to_messages = to_messages  # Сообщения, к которым ведёт кнопка
            self.__func = func  # Функция отбора сообщения из to_messages на основе предыдущего сообщения /
            # вспомогательное

    def __call__(self, message_tg: telebot.types.Message,
                 userSendLogger: bool = True) -> Message:  # При вызове кновки отдаем сообщение к которому будем идти
        if userSendLogger:
            Message.userSendLogger(message_tg, f'[{self.text}]')

        temp = self.__func(self.to_messages, message_tg)
        if temp is not None:
            return temp

        if self.to_messages:
            return self.to_messages[0]

    def __repr__(self):
        return self.callback, self.callback_data[self.callback]

    def __getattr__(self, data: str):  # by data
        for instance in self.instances:
            if self.callback_data[instance.callback] == data:
                return instance

    def get_instance(self, callback: str):  # by callback
        for instance in self.instances:
            if instance.callback == callback:
                return instance

    @classmethod
    def create_callback(cls, data: str):
        callback = operations.get_callback(data)

        if len(cls.callback_data) > 10 ** config.length_callback:
            raise ValueError("Callback length is too short")

        if callback is not None and len(callback) == config.length_callback:
            return callback

        callback = ''.join(str(randint(0, 9)) for _ in range(config.length_callback))
        while callback in cls.callback_data:
            callback = ''.join(str(randint(0, 9)) for _ in range(config.length_callback))
        operations.record_callback_data(callback, data)

        return callback


# Функция для отправки HTML письма
def send_email(to_email: str, subject: str, html_content: str):
    msg = MIMEMultipart()
    msg['From'] = config.EMAIL_USER
    msg['To'] = to_email
    msg['Subject'] = subject

    # HTML-содержимое письма
    msg.attach(MIMEText(html_content, 'html'))

    # Подключение к SMTP-серверу и отправка письма
    server = smtplib.SMTP_SSL(config.SMTP_SERVER, config.SMTP_PORT)
    server.login(config.EMAIL_USER, config.EMAIL_PASSWORD)
    server.sendmail(config.EMAIL_USER, to_email, msg.as_string())
    server.quit()
