### 部署模型

- https://huggingface.co/meta-llama/Llama-2-7b-chat-hf

```shell
# 克隆 Llama2 源码
$ git clone https://github.com/facebookresearch/llama.git

# 安装依赖
$ cd llama
$ pip install -e .

# 下载指定模型数据（大文件）
# https://huggingface.co/meta-llama/Llama-2-7b-chat-hf
$ GIT_LFS_SKIP_SMUDGE=1 git clone git@hf.co:meta-llama/Llama-2-7b-chat-hf
$ git lfs pull --include "*.safetensors"

# 单独下载 
$ git lfs pull --include "model-00001-of-00002.safetensors"
$ git lfs pull --include "model-00002-of-00002.safetensors"


# 修改使用本地模型
# TODO


# 创建运行环境（注意 python 版本，建议使用 3.10.x 版本）
$ conda create -n llama2 python=3.10.12
$ conda activate llama2

```


测试用例

```python

# Use a pipeline as a high-level helper
from transformers import pipeline
# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM


pipe = pipeline("text-generation", model="meta-llama/Llama-2-7b-chat-hf")

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

model
```


Web UI

- https://github.com/oobabooga/text-generation-webui

```shell

# https://github.com/oobabooga/text-generation-webui/issues/3246
$ git clone https://github.com/oobabooga/text-generation-webui.git
$ cd text-generation-webui
$ pip install -r requirements.txt

# https://github.com/ymcui/Chinese-LLaMA-Alpaca/wiki/%E4%BD%BF%E7%94%A8text-generation-webui%E6%90%AD%E5%BB%BA%E7%95%8C%E9%9D%A2
# 复制模型到目录 text-generation-webui/models/

# 启动服务 - 不设置模型也可以在UI上选择并加载
$ python server.py --model Llama-2-7b-chat-hf --listen-port 6006 --share
$ python server.py --listen-port 6006 --share

```