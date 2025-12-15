# Cell 1: 安装依赖库
print("⏳ 正在安装依赖库，请稍等...")
# 安装 Hugging Face 相关库及音频处理库
!pip install datasets transformers accelerate librosa evaluate torch soundfile==0.12.1 --quiet
!pip install --upgrade accelerate --quiet
print("✅ 环境安装完成！")
