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

```shell
# 安装依赖
$ pip install rouge_chinese nltk jieba datasets


```