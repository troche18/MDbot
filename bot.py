import discord
from discord.ext import commands
from discord import app_commands
from config.config import DISCORD_BOT_TOKEN, CHANNEL_ID, TARGET_USER_ID
from markov_chain import load_messages, filter_user_messages, train_markov_chain, generate_sentence

intents = discord.Intents.default()
intents.messages = True

class MyBot(commands.Bot):
    def __init__(self):
        super().__init__(command_prefix='!', intents=intents)
        self.tree = app_commands.CommandTree(self)
        self.markov_model = None

    async def setup_hook(self):
        await self.tree.sync()

bot = MyBot()

@bot.event
async def on_ready():
    print(f'Logged in as {bot.user}')

@bot.tree.command(name="train", description="Train the Markov chain model with user messages")
async def train(interaction: discord.Interaction):
    if interaction.channel_id != int(CHANNEL_ID):
        await interaction.response.send_message("このコマンドは指定されたチャンネルでのみ使用できます。", ephemeral=True)
        return

    # メッセージをロード
    messages = load_messages('data/messages.json')
    
    # 特定のユーザーのメッセージをフィルタリング
    user_messages = filter_user_messages(messages, TARGET_USER_ID)
    
    # マルコフ連鎖モデルを学習
    bot.markov_model = train_markov_chain(user_messages)
    
    await interaction.response.send_message("マルコフ連鎖モデルの学習が完了しました。")

@bot.tree.command(name="generate", description="Generate a sentence using the trained Markov chain model")
async def generate(interaction: discord.Interaction):
    if interaction.channel_id != int(CHANNEL_ID):
        await interaction.response.send_message("このコマンドは指定されたチャンネルでのみ使用できます。", ephemeral=True)
        return

    if bot.markov_model is None:
        await interaction.response.send_message("モデルが学習されていません。まずは/trainコマンドを使用してください。", ephemeral=True)
        return
    
    # 文章を生成
    generated_sentence = generate_sentence(bot.markov_model)
    
    if generated_sentence:
        await interaction.response.send_message(generated_sentence)
    else:
        await interaction.response.send_message("文章を生成できませんでした。")

bot.run(DISCORD_BOT_TOKEN)
