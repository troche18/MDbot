import json
import markovify
from config.config import MARKOV_STATE_SIZE, TARGET_USER_ID

def load_messages(file_path):
    """メッセージをJSONファイルから読み込む"""
    with open(file_path, 'r', encoding='utf-8') as file:
        messages = json.load(file)
    return messages

def filter_user_messages(messages, user_id):
    """特定のユーザーのメッセージのみをフィルタリング"""
    return [msg['content'] for msg in messages if msg['author_id'] == user_id]

def train_markov_chain(messages):
    """マルコフ連鎖モデルを学習"""
    text = " ".join(messages)
    return markovify.Text(text, state_size=MARKOV_STATE_SIZE)

def generate_sentence(model):
    """マルコフ連鎖モデルから文章を生成"""
    return model.make_sentence()

if __name__ == "__main__":
    # メッセージをロード
    messages = load_messages('data/messages.json')
    
    # 特定のユーザーのメッセージをフィルタリング
    user_messages = filter_user_messages(messages, TARGET_USER_ID)
    
    # マルコフ連鎖モデルを学習
    markov_model = train_markov_chain(user_messages)
    
    # 文章を生成
    generated_sentence = generate_sentence(markov_model)
    print(generated_sentence)
