import sqlite3
from Auxiliary import config


def creating_tables():
    # Подключение к базе данных
    connection = sqlite3.connect(config.Paths.DataBase)
    cursor = connection.cursor()

    # Создание таблицы "callback_data", если она не существует
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS "callback_data" (
            "callback" TEXT NOT NULL UNIQUE,
            "data" TEXT NOT NULL PRIMARY KEY UNIQUE
        );
        """)

    # Сохранение изменений
    connection.commit()

    # Закрытие соединения
    connection.close()


# Callback_data
def get_callback(data: str):
    # Подключение к базе данных
    connection = sqlite3.connect(config.Paths.DataBase)
    cursor = connection.cursor()

    # Находим нужный callback по data
    cursor.execute("SELECT callback FROM callback_data WHERE data = ?", (data,))

    callback = cursor.fetchone()

    # Закрытие соединения
    connection.close()

    return callback[0] if callback is not None else None


def record_callback_data(callback: str | int, data: str):
    # Подключение к базе данных
    connection = sqlite3.connect(config.Paths.DataBase)
    cursor = connection.cursor()

    # Находим нужный callback по data
    temp = get_callback(data)

    # Запись данных в таблицу callback_data
    if temp is None:
        cursor.execute("""
            INSERT INTO "callback_data" (
              "callback",
              "data"
            )
            VALUES (?, ?)
            """, (callback, data))
    else:
        cursor.execute("UPDATE callback_data SET callback = ? WHERE data = ?",
                       (callback, data))

    # Сохранение изменений
    connection.commit()

    # Закрытие соединения
    connection.close()


if __name__ != '__main__':
    creating_tables()
