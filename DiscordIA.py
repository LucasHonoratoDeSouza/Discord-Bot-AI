import discord
from discord.ext import commands
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics.pairwise import cosine_similarity
import unicodedata
import asyncio



edit_mode = False
# Token do seu bot
TOKEN = 'seu-token'

intents = discord.Intents.default()
intents.messages = True

client = discord.Client(intents=intents)

dataset = pd.read_csv("dataset.csv")

X = dataset["pergunta"]
y = dataset["resposta"]

vectorizer = TfidfVectorizer()
X_vectorized = vectorizer.fit_transform(X)

model = MultinomialNB()
model.fit(X_vectorized, y)

def remover_acentos(texto):
    return ''.join(char for char in unicodedata.normalize('NFD', texto) if unicodedata.category(char) != 'Mn')

def encontrar_melhor_resposta(pergunta):
    pergunta_normalizada = remover_acentos(pergunta.lower())
    pergunta_vectorized = vectorizer.transform([pergunta_normalizada])
    
    similarities = cosine_similarity(pergunta_vectorized, X_vectorized)
    max_similarity_index = similarities.argmax()
    max_similarity = similarities[0, max_similarity_index]
    
    limite_similaridade = 0.7  # Ajuste o limite de similaridade aqui
    
    if max_similarity > limite_similaridade:
        return dataset.loc[max_similarity_index, 'resposta']
    else:
        return "Desculpe, não encontrei uma resposta adequada."


def ensinar(pergunta, resposta):
    global dataset
    nova_linha = pd.DataFrame({"pergunta": [pergunta], "resposta": [resposta]})
    dataset = pd.concat([dataset, nova_linha], ignore_index=True)
    dataset.to_csv("dataset.csv", index=False)

@client.event
async def on_ready():
    print(f'Conectado como {client.user}')

edit_mode = False

@client.event
async def on_message(message):
    global edit_mode
    if isinstance(message.channel, discord.DMChannel) and message.author != client.user:
        pergunta = message.content
        #print(message.author, ": ", pergunta)
        resposta = encontrar_melhor_resposta(pergunta)

        if resposta == "Desculpe, não encontrei uma resposta adequada.":
            if edit_mode == False:
                edit_mode = True
                await message.author.send(resposta)
                await message.author.send("Por favor, ensine-me a resposta:")
                def check(m):
                    return m.author == message.author and isinstance(m.channel, discord.DMChannel)
                try:
                    ensinar_resposta = await client.wait_for('message', check=check, timeout=10)
                    ensinar(pergunta, ensinar_resposta.content)
                    await message.author.send("obrigado por me ensinar! (seu ensinamento ficará disponivel quando o bot reiniciar)")
                    edit_mode = False
                    
                except asyncio.TimeoutError:
                    await message.author.send("Tempo esgotado. Por favor, tente novamente.")
                    edit_mode = False
            
        else:
            await message.author.send(resposta)

client.run(TOKEN)
