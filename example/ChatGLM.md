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



```