# From python:3.7

# COPY ./requirements.txt /webapp/requirements.txt

# WORKDIR /webapp

# RUN pip install -r requirements.txt

# COPY webapp/* /webapp

# ENTRYPOINT [ "python" ]

# CMD [ "app.py" ]

# ==============================
# 使用 python:3.10 映像
FROM python:3.10

# 安裝 Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -

# 將 Poetry 的 bin 目錄添加到 PATH
ENV PATH="/root/.local/bin:$PATH"

# 設置工作目錄
WORKDIR /webapp

# 複製 pyproject.toml 和 poetry.lock 文件到容器中
COPY ./pyproject.toml ./poetry.lock* /webapp/

# 安裝依賴項
RUN poetry install --no-root

# 複製 webapp 目錄中的所有文件到容器中
COPY webapp /webapp

# 設置 ENTRYPOINT 和 CMD
ENTRYPOINT [ "poetry", "run", "python" ]
CMD [ "app.py" ]