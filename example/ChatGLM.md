## 部署 ChatGLM 模型
- https://github.com/imClumsyPanda/langchain-ChatGLM
- https://github.com/imClumsyPanda/langchain-ChatGLM/blob/master/docs/INSTALL.md
- https://www.codewithgpu.com/i/imClumsyPanda/langchain-ChatGLM/langchain-ChatGLM
- https://www.autodl.com/docs/network_turbo/


1. 购买 AutoDL GPU节点，环境要求：Python 3.8.1 - 3.10，CUDA 11.7 环境下完成测试
2. 设置学术资源加速 `source /etc/network_turbo`
3. 部署 ChatGLM 模型

```
# https://github.com/imClumsyPanda/langchain-ChatGLM/blob/master/docs/INSTALL.md


# ==============================================================================
# 创建虚拟环境
# ==============================================================================
$ conda env list
base                  *  /root/miniconda3

$ conda create --name chatglm

$ conda env list
base                  *  /root/miniconda3
chatglm                  /root/miniconda3/envs/chatglm

$ source activate chatglm


# ==============================================================================
# 下载模型
# ==============================================================================
# https://github.com/git-lfs/git-lfs/blob/main/INSTALLING.md
$ (. /etc/lsb-release && curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | env os=ubuntu dist="${DISTRIB_CODENAME}" bash)
$ apt install git-lfs
$ git lfs install

# https://github.com/THUDM/ChatGLM2-6B
$ git clone https://huggingface.co/THUDM/chatglm2-6b

# 分屏查看下载进度（流量监控）
$ apt install bwm-ng
$ bmw-ng

# 下载后的模型目录
/root/autodl-tmp/model/chatglm2-6b


# ==============================================================================
# 部署模型
# ==============================================================================
# 拉取仓库
$ git clone https://github.com/imClumsyPanda/langchain-ChatGLM.git

# 进入目录
$ cd langchain-ChatGLM

# 安装依赖
$ pip install -r requirements.txt


# ==============================================================================
# 启动模型
# ==============================================================================
$ python webui.py --model-dir /root/autodl-tmp/model/ --model chatglm2-6b --no-remote-model
 
# https://www.codewithgpu.com/i/imClumsyPanda/langchain-ChatGLM/langchain-ChatGLM
# https://www.autodl.com/docs/ssh_proxy/



```## 部署 ChatGLM 模型
- https://github.com/imClumsyPanda/langchain-ChatGLM
- https://github.com/imClumsyPanda/langchain-ChatGLM/blob/master/docs/INSTALL.md
- https://www.codewithgpu.com/i/imClumsyPanda/langchain-ChatGLM/langchain-ChatGLM
- https://www.autodl.com/docs/network_turbo/
- https://github.com/NVIDIA/nvidia-container-toolkit

1. 购买 AutoDL GPU节点，环境要求：Python 3.8.1 - 3.10，CUDA 11.7 环境下完成测试
2. 设置学术资源加速 `source /etc/network_turbo`
3. 部署 ChatGLM 模型

```shell
# https://github.com/imClumsyPanda/langchain-ChatGLM/blob/master/docs/INSTALL.md


# ==============================================================================
# 创建虚拟环境
# ==============================================================================
$ conda env list
base                  *  /root/miniconda3

$ conda create --name chatglm

$ conda env list
base                  *  /root/miniconda3
chatglm                  /root/miniconda3/envs/chatglm

$ source activate chatglm


# ==============================================================================
# 下载模型
# ==============================================================================
# https://github.com/git-lfs/git-lfs/blob/main/INSTALLING.md
$ (. /etc/lsb-release && curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | env os=ubuntu dist="${DISTRIB_CODENAME}" bash)
$ apt install git-lfs
$ git lfs install

# https://github.com/THUDM/ChatGLM2-6B
$ git clone https://huggingface.co/THUDM/chatglm2-6b

# 分屏查看下载进度（流量监控）
$ apt install bwm-ng
$ bmw-ng

# 下载后的模型目录
/root/autodl-tmp/model/chatglm2-6b


# ==============================================================================
# 部署模型
# ==============================================================================
# 拉取仓库
$ git clone https://github.com/imClumsyPanda/langchain-ChatGLM.git

# 进入目录
$ cd langchain-ChatGLM

# 安装依赖
$ pip install -r requirements.txt

# 修改配置，使用本地模型，避免远程下载 
# configs/model_config.py

```

#### WebUI

```shell
# ==============================================================================
# 启动模型
# ==============================================================================
$ NUMEXPR_MAX_THREADS=1 python webui.py --no-remote-model --model-name chatglm2-6b
 
# https://www.codewithgpu.com/i/imClumsyPanda/langchain-ChatGLM/langchain-ChatGLM
# https://www.autodl.com/docs/ssh_proxy/

# 修改 webui.py 文件
# #launch() -> share=True
# #launch() -> server_port=6006
```

#### CLI

```shell
$ python cli.py --help

Usage: cli.py [OPTIONS] COMMAND [ARGS]...

Options:
  --version  Show the version and exit.
  --help     Show this message and exit.

Commands:
  embedding
  llm
  start

$ NUMEXPR_MAX_THREADS=1 python cli_demo.py --no-remote-model --model-name chatglm2-6b
```

#### API

```shell
# https://github.com/imClumsyPanda/langchain-ChatGLM/blob/master/docs/API.md
# https://github.com/imClumsyPanda/langchain-ChatGLM/blob/master/docs/fastchat.md

# 1. 配置 model，修改 configs/model_config.py，修改 fastchat-chatglm2-6b 配置项，修改 api_base_url 配置项为对外访问的地址
# 2. 通过访问 /docs 查看 API 文档


$ NUMEXPR_MAX_THREADS=1 python api.py --no-remote-model --model-name chatglm2-6b --port 6006

```

## 向量化

```
# https://huggingface.co/GanymedeNil/text2vec-large-chinese

# 分拆字符串参考 chains/local_doc_qa.py 文件
# 修改 ChineseTextSplitter 文件，使之符合自定义格式（默认实现可能不太符合需求）

```

## LangChain

```
# 参考实现
# https://github.com/imClumsyPanda/langchain-ChatGLM/blob/master/chains/local_doc_qa.py


```

## ChatGLM2-6B

> 使用原生模型服务

```shell
# 下载模型源码
$ git clone https://github.com/THUDM/ChatGLM2-6B.git

# 安装依赖
$ pip install -r requirements.txt

# 下载模型，并将 api / web / open_api 等路径调整为本地路径

# 其它启动方式
$ python cli_demo.py

# 测试方法参见 ChatGLM.http 文件
$ python api.py

# 使用 gradio 实现，可以通过命令行，也可以通过环境变量或代码指定端口
$ GRADIO_SERVER_PORT=6006 python web_demo.py

# 使用类 openai API启动模型 - 仅支持 Chat Completions API 
# 测试用例参见 ChatGLM.http 文件
$ python open_api.py

```

## 微调

### [P-tuning-v2](https://github.com/THUDM/P-tuning-v2)

```shell
# https://github.com/THUDM/ChatGLM2-6B/blob/main/ptuning/README.md

# 准备数据，将数据文件复制到指定目录
# 数据生成过程参考 ChatGLM.ipynb 文件

# 安装依赖
$ pip install rouge_chinese nltk jieba datasets

# P-Tuning-v2 方法会冻结全部的模型参数
# 修改 ChatGLM2-6B/ptuning/train.sh 和 ChatGLM2-6B/ptuning/evaluate.sh 脚本，准备微调
# PRE_SEQ_LEN - 设置 soft prompt 长度
# LR - 设置训练的学习率
# quantization_bit - 设置量化精度，默认为 FP16 精度

# 将准备好的微调数据放在指定位置，执行微调命令，注意：需要进入 ptuning 目录执行命令，否则可能会报找不到 main.py 文件的错误
$ cd ptuning
$ bash train.sh

# 验证模型
$ bash evaluate.sh

# 观察 GPU 使用情况
$ watch -n 2 nvidia-smi

# Q&A
# ImportError: Using the `Trainer` with `PyTorch` requires `accelerate>=0.20.1`: Please run `pip install transformers[torch]` or `pip install accelerate -U`
$ pip install transformers[torch]

```

脚本 train.sh 说明

```shell
PRE_SEQ_LEN=128
LR=2e-2
NUM_GPUS=1

torchrun --standalone --nnodes=1 --nproc-per-node=$NUM_GPUS main.py \
    --do_train \
    --train_file /root/autodl-tmp/ptunning/train.json \
    --validation_file /root/autodl-tmp/ptunning/dev.json \
    --preprocessing_num_workers 10 \
    --prompt_column content \
    --response_column summary \
    --overwrite_cache \
    --model_name_or_path /root/autodl-tmp/model/chatglm2-6b \
    --output_dir /root/autodl-tmp/output/adgen-chatglm2-6b-pt-$PRE_SEQ_LEN-$LR \
    --overwrite_output_dir \
    --max_source_length 64 \
    --max_target_length 128 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --predict_with_generate \
    --max_steps 3000 \
    --logging_steps 10 \
    # 机器学习中，指定在多少步之后保存模型的检查点，以便将来从这些检查点恢复训练，或者用来做模型评估
    --save_steps 1000 \
    # 学习率 (Learning Rate)
    --learning_rate $LR \
    # Soft Prompt 长度
    --pre_seq_len $PRE_SEQ_LEN \
    # 量化精度，不加该参数时为 FP16，int4是指4位整数（0~15, -7~8），FP16表示半精度浮点数
    # 使用"int4"可以减少模型的存储需求和计算复杂性，但可能会导致一些精度损失；而使用"fp16"则可以保持较高的精度，但需要更多的存储空间和计算资源
    --quantization_bit 4

```

脚本 evaluate.sh 说明

```shell
PRE_SEQ_LEN=128
CHECKPOINT=adgen-chatglm2-6b-pt-128-2e-2
STEP=3000
NUM_GPUS=1

torchrun --standalone --nnodes=1 --nproc-per-node=$NUM_GPUS main.py \
    --do_predict \
    --validation_file /root/autodl-tmp/ptunning/dev.json \
    --test_file /root/autodl-tmp/ptunning/dev.json \
    --overwrite_cache \
    --prompt_column content \
    --response_column summary \
    --model_name_or_path /root/autodl-tmp/model/chatglm2-6b \
    --ptuning_checkpoint /root/autodl-tmp/output/$CHECKPOINT/checkpoint-$STEP \
    --output_dir /root/autodl-tmp/output/$CHECKPOINT \
    --overwrite_output_dir \
    --max_source_length 64 \
    --max_target_length 64 \
    --per_device_eval_batch_size 1 \
    --predict_with_generate \
    --pre_seq_len $PRE_SEQ_LEN \
    --quantization_bit 4

```