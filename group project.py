
# ==============================================================================
# 🐶 狗狗情绪翻译官 - 最终修复版 (修正翻译类别，并确保语音完整输出)
# ==============================================================================

# 1. 必要的库
import random
import librosa
import numpy as np
from transformers import pipeline
from IPython.display import Audio, display
from google.colab import files

print("🐶 正在更新翻译剧本...")

# 加载模型 (指向你刚才训练好的文件夹)
# 确保这里路径是对的
my_model_path = "./final_emotion_model"
classifier = pipeline("audio-classification", model=my_model_path)
tts = pipeline("text-to-speech", model="facebook/mms-tts-eng")

# --- 📝 关键修复：更新翻译字典，使其与模型预测的狗狗情绪标签一致 ---
translation_script = {
    # 1. 攻击 (Aggression) - 对应 'aggression'
    'aggression': [
        "😡 翻译: '离我远点！我在生气！(Get away! I'm angry!)'",
        "😡 翻译: '这是我的地盘！不准靠近！(My territory! Stay back!)'",
        "😡 翻译: '别惹我，小心我咬你！(Don't provoke me!)'"
    ],
    # 2. 联络/呼唤 (Contact) - 对应 'contact'
    'contact': [
        "👋 翻译: '哈喽？有人在吗？(Hello? Is anyone here?)'",
        "👋 翻译: '我在这儿！你们在哪呢？(I'm here! Where are you?)'",
        "👋 翻译: '主人，看我一眼嘛！(Master, look at me!)'"
    ],
    # 3. 玩耍 (Play) - 对应 'play'
    'play': [
        "😄 翻译: '快把球扔过来！我们来玩呀！(Throw the ball! Let's play!)'",
        "😄 翻译: '来追我呀！我跑得可快了！(Catch me if you can!)'",
        "😄 翻译: '我超开心的！想和你一起玩！(I'm super happy! Let's play!)'"
    ]
}

def translate_and_speak(audio_path):
    # 读取音频
    audio_array, sr = librosa.load(audio_path, sr=16000)

    # 预测
    predictions = classifier(audio_array)
    top_prediction = predictions[0]
    label = top_prediction['label']
    score = top_prediction['score']

    # 转换为小写，防止大小写不匹配
    label_key = label.lower()

    # 查字典
    # 如果字典里有，随机选一句；如果没有，显示默认提示
    texts = translation_script.get(label_key, [f"🤔 翻译: 我听到 '{label}' 的声音，但没有特定的翻译。(置信度: {score:.2%})"])
    result_text = random.choice(texts)

    # 语音合成
    # 使用完整的 result_text 进行语音合成，不再截断。
    text_to_read = result_text
    # 移除翻译内容中的表情符号，避免 TTS 报错或发音奇怪
    text_to_read = text_to_read.split("翻译:")[-1].strip() # 提取纯文本部分
    text_to_read = ''.join(c for c in text_to_read if c.isalnum() or c.isspace() or c in '!.?,') # 过滤特殊字符

    tts_output = tts(text_to_read)

    return label, score, result_text, tts_output

# --- 交互界面 ---
print("\n" + "="*50)
print("🎤 狗狗情绪翻译官已修复！现在能识别【狗狗情绪】并进行翻译了。") # 修正用户提示
print("⬇️ 请上传一个狗狗的叫声文件进行测试")
print("="*50)

uploaded = files.upload()

for filename in uploaded.keys():
    print(f"\n🔍 分析中: {filename} ...")
    try:
        emotion, conf, text, speech = translate_and_speak(filename)

        print("-" * 30)
        print(f"🐶 识别情绪: 【{emotion.upper()}】") # 打印出真正识别到的英文标签
        print(f"📊 置信度:   {conf:.2%}")
        print(f"📝 翻译内容: {text}")
        print("-" * 30)

        display(Audio(data=speech['audio'], rate=speech['sampling_rate']))

    except Exception as e:
        print(f"❌ 出错: {e}")
        # Cell 5.5: 评估模型在测试集上的表现 (分类报告与混淆矩阵)
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

print("⏳ 正在评估模型在测试集上的表现...")

# 获取测试集上的预测结果
predictions = trainer.predict(encoded_dataset["test"])
y_pred = np.argmax(predictions.predictions, axis=1) # 预测的类别ID
y_true = predictions.label_ids # 真实的类别ID

# 将数字标签ID转换回字符串标签名称，以便报告更易读
# id2label 是从 Cell 2 中获得的映射
y_pred_names = [id2label[str(label_id)] for label_id in y_pred]
y_true_names = [id2label[str(label_id)] for label_id in y_true]

# 获取所有可能的标签名称，并确保顺序与 id2label 对应
target_names = [id2label[str(i)] for i in sorted([int(k) for k in id2label.keys()])]

# 打印分类报告
print("\n--- 分类报告 (Classification Report) ---")
print(classification_report(y_true_names, y_pred_names, target_names=target_names))

# 绘制混淆矩阵
print("\n--- 混淆矩阵 (Confusion Matrix) ---")
cm = confusion_matrix(y_true_names, y_pred_names, labels=target_names)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)

fig, ax = plt.subplots(figsize=(8, 6))
disp.plot(cmap=plt.cm.Blues, ax=ax)
plt.title('Confusion Matrix for Dog Emotion Classification on Test Set')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

print("✅ 模型在测试集上的表现评估完成！")
