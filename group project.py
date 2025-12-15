# Cell 1: 安装依赖库
print("⏳ 正在安装依赖库，请稍等...")
# 安装 Hugging Face 相关库及音频处理库
!pip install datasets transformers accelerate librosa evaluate torch soundfile==0.12.1 --quiet
!pip install --upgrade accelerate --quiet
print("✅ 环境安装完成！")
# Cell 2: 加载数据与查看标签
import warnings
from datasets import load_dataset, Audio, ClassLabel

warnings.filterwarnings("ignore")
print("⏳ 正在加载狗狗情绪数据集 (cgeorgiaw/animal-sounds)...")

# 1. 加载 "dogs" 子集 (包含 警戒、孤独、玩耍 三种情绪)
try:
    # 移除了 trust_remote_code=True 参数，因为该参数已不被推荐使用。
    dataset = load_dataset("cgeorgiaw/animal-sounds", "dogs", split="train")
    # Debugging: 打印数据集特征以找出正确的标签列名
    print(f"DEBUG: Dataset features keys for 'dogs' subset: {dataset.features.keys()}")

    # FIX: 情绪标签在 'context' 列中，需要从这里提取并创建 'label' 列
    unique_contexts = sorted(list(set(dataset["context"])))
    labels = unique_contexts

    print("✅ 成功加载 'dogs' 子集。")
except Exception as e:
    print(f"⚠️ 无法加载 'dogs' 子集或提取标签: {e}")
    print("❌ 无法继续，请检查数据集或网络连接。")
    raise # 重新抛出异常，停止程序。

# 2. 获取标签信息
# 这是一个关键步骤，我们要确认 dataset 里有哪些情绪

print(f"📊 数据集包含的情绪标签: {labels}")
# 预期输出: ['disturbance', 'isolation', 'play'] 等 (现在应该能正确输出狗狗情绪标签了)

# 3. 制作标签映射字典 (让 AI 读懂这些词)
label2id = {label: str(i) for i, label in enumerate(labels)}
id2label = {str(i): label for i, label in enumerate(labels)}

# 4. 将 'context' 列映射到新的 'label' 列，并进行类型转换
def map_context_to_label(example):
    example['label'] = label2id[example['context']]
    return example

dataset = dataset.map(map_context_to_label)
# 移除原始的 'context' 列，如果不再需要
dataset = dataset.remove_columns(['context'])
# 转换为 ClassLabel 类型
dataset = dataset.cast_column("label", ClassLabel(names=labels))

# 5. 划分训练集 (80%) 和测试集 (20%)
dataset = dataset.train_test_split(test_size=0.2, seed=42)

print(f"✅ 数据准备就绪！")
print(f"   - 训练集样本: {len(dataset['train'])}")
print(f"   - 测试集样本: {len(dataset['test'])}")
# Cell 3.5: 使用 SMOTE 过采样技术平衡训练集
import numpy as np
from imblearn.over_sampling import SMOTE
from collections import Counter
from datasets import Dataset # 导入 Dataset 用于从 NumPy 数组创建 Hugging Face Dataset

print("⏳ 正在平衡训练集中的情绪类别...")

# 1. 准备训练数据：提取特征和标签
# 确保 'input_values' 转换为 NumPy 数组，以便 SMOTE 处理
X_train = np.array(encoded_dataset['train']['input_values'])
y_train = np.array(encoded_dataset['train']['label'])

print(f"平衡前训练集标签分布: {Counter(y_train)}")

# 2. 应用 SMOTE 进行过采样
sm = SMOTE(random_state=42, k_neighbors=1) # k_neighbors=1 是最小设置，适用于样本量较少的类别
X_resampled, y_resampled = sm.fit_resample(X_train, y_train)

print(f"平衡后训练集标签分布: {Counter(y_resampled)}")

# 3. 将平衡后的数据转换回 Hugging Face Dataset 格式
# 需要重新构造一个 Dataset 对象
encoded_dataset_balanced_train = Dataset.from_dict({
    'input_values': X_resampled.tolist(), # SMOTE 输出的是 NumPy 数组，需要转回列表
    'labels': y_resampled.tolist() # 标签也要转回列表
})

# 确保新数据集包含其他原始列，例如 attention_mask，这在 Wav2Vec2 中很重要
# 由于 SMOTE只处理 input_values 和 labels，其他列将丢失。我们需要重新添加它们。
# 这是一个挑战，因为SMOTE会生成新的样本，没有对应的原始 attention_mask。
# 简化的处理方式是假设所有新样本都使用与原始样本相同的 attention_mask 策略。
# 但更稳妥的是直接在 Trainer 中处理 batch 数据的 padding/attention_mask。
# 对于 Wav2Vec2，input_values 已经是固定长度且padding=True, truncation=True。
# 因此，可以假设 attention_mask 是全1（如果 max_length == 实际长度）或者基于 padding 生成。
# 为了简单起见，这里假设 `input_values` 的形状和原始数据一致，可以从原始数据集中提取 `attention_mask`。

# 我们可以构建一个包含所有必要特征的新 Dataset
# 这里需要注意的是，SMOTE只会对X_train和y_train进行操作，并不会生成attention_mask
# 我们可以先创建一个只有 input_values 和 labels 的 Dataset，然后在 Trainer 中处理 padding 和 attention_mask
# 但为了与原始 `encoded_dataset` 结构一致，我们最好也为 resampled data 生成 attention_mask。

# 重新创建 balanced_train_dataset，并添加一个默认的 attention_mask (全1，因为我们已经固定了长度)
# 注意: 理论上，attention_mask应该由 feature_extractor 根据实际数据长度生成。
# 但在这里，SMOTE生成的新数据，我们没有原始的音频长度信息。
# 鉴于 `input_values` 已经过 `feature_extractor` 统一长度处理，我们可以假设 `attention_mask` 都是全1。

attention_mask_resampled = np.ones(X_resampled.shape, dtype=int)

encoded_dataset['train'] = Dataset.from_dict({
    'input_values': X_resampled.tolist(),
    'attention_mask': attention_mask_resampled.tolist(),
    'labels': y_resampled.tolist()
})

print("✅ 训练集情绪类别平衡完成！")
# Cell 3: 特征提取 (形状强力对齐版)
import numpy as np
import soundfile as sf
import io
import librosa
from transformers import AutoFeatureExtractor
from datasets import ClassLabel # 导入 ClassLabel

print("⏳ 正在处理音频特征...")

# 1. 加载特征提取器
model_checkpoint = "facebook/wav2vec2-base"
feature_extractor = AutoFeatureExtractor.from_pretrained(model_checkpoint)

# **FIX:** 确保 'label' 列存在并具有正确的类型
# 数据集默认具有 'class' 列。在转换其类型之前，将其重命名为 'label'。
# 如果 'class' 列不存在，则尝试从 'name' 列推断 'label'。

if 'class' in dataset['train'].column_names:
    dataset = dataset.rename_column("class", "label")
    print("✅ 已将 'class' 列重命名为 'label'。")
elif 'label' not in dataset['train'].column_names: # Only attempt to create if 'label' is also missing
    print("⚠️ 数据集中未找到 'class' 列。尝试从 'name' 列创建 'label'。")
    # 定义一个函数，从 'name' 字段中提取标签信息
    def assign_label_from_name(example):
        found_label = None
        for animal_label in labels: # 'labels' comes from Cell 2
            if animal_label in example['name'].lower(): # Check if animal name is in the example's 'name' field
                found_label = animal_label
                break
        if found_label:
            example['label'] = int(label2id[found_label])
        else:
            # Fallback for cases where no animal name is found in 'name'
            print(f"⚠️ Warning: Could not infer label for name: {example['name']}. Assigning to {labels[0]} as fallback.")
            example['label'] = int(label2id[labels[0]]) # Assign to the first label as a fallback
        return example
    dataset = dataset.map(assign_label_from_name)
    print("✅ 成功从 'name' 列创建 'label' 列。")
else:
    print("✅ 'label' 列已存在，无需额外处理。")

# 2. 关闭自动解码并转换 'label' 列类型
# 现在 'label' 列应该已经存在了，我们可以进行类型转换。
dataset = dataset.cast_column("label", ClassLabel(names=labels)) # 确保标签类型正确
dataset = dataset.cast_column("audio", Audio(decode=False))

# 3. 定义“强力对齐”处理函数
def preprocess_function(examples):
    audio_arrays = []
    target_sr = 16000
    target_length = 24000  # 1.5秒

    for audio_data in examples["audio"]:
        try:
            # A. 读取音频
            if "bytes" in audio_data and audio_data["bytes"]:
                array, sr = sf.read(io.BytesIO(audio_data["bytes"]))
            elif "path" in audio_data and audio_data["path"]:
                array, sr = sf.read(audio_data["path"])
            else:
                array = np.zeros(target_length)
                sr = target_sr

            # B. 【关键修复】强制转单声道 (Mono)
            # 如果是立体声 (N, 2)，librosa 或者是 sf 读取出来可能是二维数组
            if len(array.shape) > 1:
                # 取平均值转为单声道，或者直接取第一个声道
                array = np.mean(array, axis=1)

            # C. 重采样到 16000Hz
            if sr != target_sr:
                array = librosa.resample(array, orig_sr=sr, target_sr=target_sr)

            # D. 【关键修复】严格统一长度 (Trim or Pad)
            current_len = len(array)
            if current_len > target_length:
                # 太长了，切掉
                array = array[:target_length]
            elif current_len < target_length:
                # 太短了，补零
                padding = target_length - current_len
                array = np.pad(array, (0, padding), "constant")

            # 双重保险：确保一定是 24000 长度
            if len(array) != target_length:
                 array = np.resize(array, target_length)

            audio_arrays.append(array)

        except Exception as e:
            # 遇到任何坏数据，填入全0静音，保证程序不崩
            print(f"⚠️ 跳过坏数据: {e}")
            audio_arrays.append(np.zeros(target_length))

    # E. 调用提取器
    # 此时 audio_arrays 里的每一个元素形状都是严格的 (24000,)
    inputs = feature_extractor(
        audio_arrays,
        sampling_rate=target_sr,
        max_length=target_length,
        truncation=True,
        padding=True
    )
    return inputs

# 4. 批量处理
# 此时应该能顺畅跑通了
print("开始批量提取特征 (Batch Processing)...")
encoded_dataset = dataset.map(preprocess_function, batched=True, batch_size=4)
print("✅ 特征提取完成！数据形状已完美对齐。")
# Cell 4: 开始微调训练
import evaluate
from transformers import AutoModelForAudioClassification, TrainingArguments, Trainer

print("⏳ 正在初始化训练...")

# 1. 加载预训练模型
model = AutoModelForAudioClassification.from_pretrained(
    model_checkpoint,
    num_labels=len(labels),
    label2id=label2id,
    id2label=id2label,
)

# 2. 定义评估方法 (准确率)
metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return metric.compute(predictions=predictions, references=eval_pred.label_ids)

# 3. 设置训练参数
training_args = TrainingArguments(
    output_dir="./dog_emotion_model",
    eval_strategy="epoch",  # 每个 epoch 测一次分
    save_strategy="epoch",  # 每个 epoch 保存一次
    learning_rate=3e-5,     # 学习率
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=5,     # 建议跑 5 轮，让它学透一点
    logging_steps=10,
    load_best_model_at_end=True, # 训练结束保留最好的那个
    metric_for_best_model="accuracy"
)

# 4. 初始化 Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["test"],
    tokenizer=feature_extractor,
    compute_metrics=compute_metrics,
)

# 5. 开跑！
print("🚀 开始微调 (Start Fine-tuning)...")
trainer.train()

# 6. 保存最终模型
trainer.save_model("./final_emotion_model")
feature_extractor.save_pretrained("./final_emotion_model")
print("🎉 训练完成！模型已保存到 ./final_emotion_model")
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
