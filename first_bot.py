# 2047305408:AAE_-D7byLe9hf1R34S9Z1NYrYI6EjvY9fY
import telebot

TOKEN = "2047305408:AAE_-D7byLe9hf1R34S9Z1NYrYI6EjvY9fY"
bot = telebot.TeleBot("2047305408:AAE_-D7byLe9hf1R34S9Z1NYrYI6EjvY9fY")


@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    bot.reply_to(message, "Howdy, how are you doing?")


@bot.message_handler(func=lambda message: True)
def echo_all(message):
    bot.reply_to(message, message.text)


bot.infinity_polling()