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
