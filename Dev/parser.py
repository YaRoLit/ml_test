from xmlrpc.client import DateTime
from telethon.sync import TelegramClient
 
from telethon.tl.functions.messages import GetDialogsRequest
from telethon.tl.types import InputPeerEmpty
from telethon.tl.functions.messages import GetHistoryRequest
from telethon.tl.types import PeerChannel
 
import csv


# ВНИМАНИЕ: В этом блоке необходимо указать персональные данные для активации телеграм клиента
api_id = 
api_hash = ''       
phone = ''
#-------------------------------------------------------------------------------------
client = TelegramClient(phone, api_id, api_hash)
client.start()
#-------------------------------------------------------------------------------------


def get_participants(link):
    '''
    Функция принимает в качестве аргумента название телеграм-канала в формате 'https://t.me/your_channel
    осуществляет поиск в данном канале всех участников
    возвращает список участников (user.id, user.username, user.first_name, user.last_name)
    '''
    if not link:
        raise('Empty link')

    all_participants = []
    all_participants = client.get_participants(link)
    users = []

    for user in all_participants:
        one_user = (user.id, user.username, user.first_name, user.last_name)
        users.append(one_user)

    return users


def get_messages(link, msg_limit=10000):
    '''
    Функция принимает в качестве аргумента название телеграм-канала в формате 'https://t.me/your_channel и глубину парсинга (количество считываемых сообщений, по умолчанию 10000)
    осуществляет поиск в данном канале сообщений от всех участников
    возвращает список сообщений (message.date, message_user.id, message.text)
    '''
    if not link:
        raise('Empty link')

    all_messages = []

    for message in client.get_messages(link, msg_limit):
        one_mes = (message.date, message.sender_id, message.text)
        all_messages.append(one_mes)

    return all_messages


def save_results(arr, filename='output.csv'):
    '''
    Функция принимает в качестве аргументов массив (список) для записи в csv файл и имя файла (по умолчанию - output.csv)
    ничего не возвращает, пишет в указанный файл
    '''
    print("Сохраняем данные в файл...")
    with open(filename, "w", encoding='UTF-16') as f:
            writer = csv.writer(f, delimiter="\t", lineterminator="\n")
            writer.writerows(arr)  
    print('Данные сохранены успешно.')



#channel_username = 'https://t.me/skazochnyy_les'# your channel

#save_results(get_participants(channel_username))
#save_results(get_messages(channel_username), 'chat.csv')

#print(get_messages(channel_username))
