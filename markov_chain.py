import json
import markovify
import MeCab
import re
from config.config import MARKOV_STATE_SIZE


def load_messages(file_path):
    """メッセージをJSONファイルから読み込む"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            if not content:
                return []
            messages = json.loads(content)
        return messages
    except FileNotFoundError:
        print(f"ファイルが見つかりません: {file_path}")
        return []
    except json.JSONDecodeError:
        print(f"JSONのデコードに失敗しました: {file_path}")
        return []


def filter_message_content(messages):
    """メッセージのみをフィルタリング"""
    return [msg['content'] for msg in messages]


def train_markov_chain(messages):
    """マルコフ連鎖モデルを学習"""
    if not messages:
        print("メッセージが空です。")
        return None
    
    texts = []
    for i in messages:
        texts.append(MeCab.Tagger('-Owakati').parse(i))
        
    parsed_text = "\n".join([i for i in format_text("\n".join(texts)).split('\n') if len(i) > 1])
    
    if not parsed_text.strip():
        print("MeCabの解析結果が空です。")
        return None
    return markovify.NewlineText(parsed_text, well_formed=False, state_size=int(MARKOV_STATE_SIZE))


def format_text(t):
    t = t.replace('　', ' ')  # Full width spaces
    t = re.sub(r'([。．！？…]+)', r'\1\n', t)  # \n after ！？
    t = re.sub(r'(.+。) (.+。)', r'\1 \2\n', t)
    t = re.sub(r'\n +', '\n', t)  # Spaces
    t = re.sub(r'([。．！？…])\n」', r'\1」 \n', t)  # \n before 」
    t = re.sub(r'\n +', '\n', t)  # Spaces
    t = re.sub(r'\n+', r'\n', t).rstrip('\n')  # Empty lines
    t = re.sub(r'\n +', '\n', t)  # Spaces
    return t


def generate_sentence(model):
    """マルコフ連鎖モデルから文章を生成"""
    return model.make_short_sentence(max_chars=140, min_chars=10, tries=50)


def generate_sentence_with_word(model, sentence):
    """文章に含まれる単語から始まるマルコフ連鎖モデルから文章を生成"""
    if sentence:
        words = MeCab.Tagger('-Owakati').parse(sentence).split()
        for word in words:
            if "<@" in word or ">" in word:
                continue
            try:
                return model.make_sentence_with_start(beginning=word, max_chars=140, min_chars=10, tries=50)
            except: 
                pass
    return generate_sentence(model)


def debug_model(model):
    """マルコフモデルのデバッグ情報を出力"""
    if model is None:
        print("モデルが生成されていません。")
        return
    
    # チェーンの状態を出力
    print("モデルの状態遷移:")
    for state, transitions in model.chain.model.items():
        print(f"状態: {state} -> 遷移: {transitions}")

    # モデルの全体的な情報を出力
    print("\nモデルの全体情報:")
    print(f"状態数: {len(model.chain.model)}")


def save_model_to_file(model, file_path):
    """マルコフモデルをファイルに保存"""
    if model is None:
        print("保存するモデルがありません。")
        return
    
    model_json = model.to_json()
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(model_json)
    print(f"モデルが{file_path}に保存されました。")


def load_model_from_file(file_path):
    """ファイルからマルコフモデルを読み込む"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            model_json = file.read()
        model = markovify.Text.from_json(model_json)
        print(f"モデルが{file_path}から読み込まれました。")
        return model
    except FileNotFoundError:
        print(f"ファイルが見つかりません: {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"JSONのデコードに失敗しました: {file_path}")
        return None


if __name__ == "__main__":
    messages = filter_message_content(load_messages('data/messages.json'))
    model = train_markov_chain(messages)
    save_model_to_file(model, 'data/markov_model.json')
    model = load_model_from_file('data/markov_model.json')
    # debug_model(model)
    print(generate_sentence_with_word(model, "テスト文章"))
