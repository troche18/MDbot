import json
import discord
import random
from config.config import DISCORD_BOT_TOKEN, TARGET_USER_ID, OWNER_ID, MESSAGE_FETCH_LIMIT, ROLE_ID
from markov_chain import load_messages, filter_message_content, train_markov_chain, generate_sentence_with_word, load_model_from_file, save_model_to_file
from tqdm import tqdm
from tqdm.asyncio import tqdm

intents = discord.Intents.all()
client = discord.Client(intents=intents)


@client.event
async def on_ready():
    print(f'Logged in as {client.user}')


@client.event
async def on_message(message):
    if message.author == client.user:
        return

    if client.user in message.mentions or int(ROLE_ID) in [i.id for i in message.role_mentions] or random.randint(0, 100) == 0:
        print("メンションが含まれています。")
        if "学習して" in message.content and message.author.id == int(OWNER_ID):
            print("学習してメンションがありました。")
            await train_model(message)
        else:
            await generate_sentence_response(message)


async def train_model(message):
    await message.channel.send("マルコフ連鎖モデルの学習を開始します...")
    messages = await fetch_user_messages(message.channel, TARGET_USER_ID, MESSAGE_FETCH_LIMIT)
    save_messages(messages, 'data/messages.json')
    user_messages = filter_message_content(messages)
    client.markov_model = train_markov_chain(user_messages)
    save_model_to_file(client.markov_model, 'data/markov_model.json')
    
    await message.channel.send("マルコフ連鎖モデルの学習が完了しました。")


async def fetch_user_messages(channel, user_id, limit):
    """特定のユーザーのメッセージを取得"""
    print("メッセージを取得しています...")
    messages = load_messages('data/messages.json')
    
    async for message in tqdm(channel.history(limit=limit), desc="Fetching messages"):
        v = {
                'content': message.content,
                'author_id': message.author.id,
                'timestamp': message.created_at.isoformat()
            }
        if message.author.id == int(user_id) and v not in messages:
            messages.append(v)
    return messages


def save_messages(messages, file_path):
    """メッセージをJSONファイルに保存"""
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(messages, file, ensure_ascii=False, indent=4)


async def generate_sentence_response(message):
    print("マルコフ連鎖モデルから文章を生成します...")
    
    client.markov_model = load_model_from_file('data/markov_model.json')
    
    if not hasattr(client, 'markov_model') or client.markov_model is None:
        await message.channel.send("モデルが学習されていません。まずは学習してとメンションしてください。")
        return
    
    generated_sentence = generate_sentence_with_word(client.markov_model, message.content)
    
    if generated_sentence:
        await message.channel.send(generated_sentence.replace(" ", ""))
    else:
        await message.channel.send("文章を生成できませんでした。")

client.run(DISCORD_BOT_TOKEN)
