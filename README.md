# TG_Paravozik_bot

# Создание QnA чат-бота для консультации сотрудников РЖД по различным вопросам

# Заказчик
<IMG SRC="https://github.com/NSO-Clio/Automatic-processing-of-work-records/assets/124351915/b3cc89e4-caf9-4c88-b756-58b88335a10e" width="50%" height="50%">


ОАО "РЖД", сделанно в рамках хакатона Цифровой Прорыв сезон: Искусственный Интеллект


# Проблема
> Ежегодно в компанию ОАО “РЖД” трудоустраиваются более 100 000 новых сотрудников. Компания
обладает обширным набором социальным пакетом для сотрудников, регулирующимся портфелем
нормативных документов. В момент приема работник обязан ознакомиться сразу с большим
количеством документов. При этом специалист по управлению персоналом из-за перегруженности
вопросами делопроизводства зачастую не имеет времени объяснить работнику все его
возможности и перечень льгот. Это приводит к тому, что хорошо продуманный и финансируемый
социальный пакет не является эффективным инструментом удержания работников - ведь им
просто не пользуются, потому что не знают как. Информирование работников происходит
спонтанно, часто с искажением, что приводит в недопониманию и увеличивает путь работника к
оформлению той или иной льготы.

# Решение
- Наш итоговый продукт- QnA телеграм бот, которому можно задать вопросы по различным темам: нормативные документы, льготы и так далее. <br>
- Ещё наш бот сохраняет заданные ему вопросы и ответы для дальнейшего использования. <br>

# Структура работы нашего решения
![image](https://github.com/user-attachments/assets/ac07203b-cc45-4654-8112-e37e4f3685c9)



### Масштабируемость
> Нашим продуктом может пользоваться несколько людей одновременно, что позволяет использовать его любому работнику в любом месте страны, где есть интернет

### Адаптивность
> Наше решение-кроссплатформенное, его можно использовать и на телефоне, и на ноутбуке, и на компьютере, продукт адаптируется к возможным ошибкам, если они произойдут, то пользователю покажут где, чтобы он мог попробовать снова

# Команда


**Вершинин Михаил**
> ML-инженер
- Почта: m_ver08@mail.ru
- telegram: @Radsdafar08

**Родионова Кристина**
> TG_bot-разработчик
- Почта: -
- telegram: @kristi3d2

**Парамонов Матвей**
> Дизайнер, аналитик
- Почта: matveyparamonov08@gmail.com
- telegram: @matveyparamonov

# Аннотация к файлам
- Readme.txt - файл с описание проекта
- convert_QnA.ipynb - ноутбук с кодом, заполняющим БД вопроосов и ответов начальными данными
- main.py - главный файл, инициализирующий тг-бота
- requirements.txt - файл с библиотеками и их версиями

- Папка Auxiliary:
- chat.py - файл, содержащий структуру чата с тг-ботом
- config.py - файл, с системными переменными
- utils.py - файл, содержащий классы объектно ориентированного кода для создания тг-бота

- Папка Auxiliary/data:
- LearningResources.pdf - ВАЖНЫЙ ФАЙЛ, который должен содержать нормативный документ, разбитый на пункты (структурированный, аналогично файлу "Коллективный договор"). (НАЗВАНИЕ ПРИ ЗАМЕНЕ ФАЙЛА НЕОБХОДИМО ПОМЕНЯТЬ НА "LearningResources.pdf")
- QnA.csv - файл, содержащий начальные данные по вопросам и ответам (на основе данного .xlsx файла), эти данные будут загружены в БД

- Папка Auxiliary/DataBase:
- DataBase.db - База Данных, содержащая вопросы, их ембединги и их ответы
- control.py - файл, содержащий системные переменные про БД
- operations.py - файл, содержащий функции для операции с БД

# Запуск бота
- Запустите файл main.py
# API
- Мы убрали свои ключи для together ai из config, чтобы их не испортили
- Также если необходимо использовать решение локально, то необходимо скачать веса к модели (требует большого количества вычислительных мощностей)
