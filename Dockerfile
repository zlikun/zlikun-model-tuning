# 设置基础镜像
FROM python:3-slim

# 设置工作目录
WORKDIR /app

# 复制应用程序代码到容器中
COPY . .

# 安装应用程序依赖项
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 暴露应用程序的端口
EXPOSE 5000

# 设置环境变量
ENV FLASK_APP=wsgi.py
ENV FLASK_RUN_HOST=0.0.0.0

# 启动应用程序
CMD ["flask", "run"]
