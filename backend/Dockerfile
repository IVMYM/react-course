# syntax=docker/dockerfile:1.4
# -------------- 0) 全局变量 --------------
ARG PY_VER=3.12
ARG PYTORCH_VER=2.2.2
ARG TORCH_URL=https://download.pytorch.org/whl/cpu

# -------------- 1) 系统依赖层（几乎不变，100% 缓存） --------------
FROM python:${PY_VER}-slim-bullseye AS system-base
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    sed -i 's|deb.debian.org|mirrors.ustc.edu.cn|g' /etc/apt/sources.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        gcc g++ libgomp1 libgl1-mesa-glx libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# -------------- 2) Python 依赖层（requirements.txt 变化才重建） --------------
FROM system-base AS python-builder
ARG PYTORCH_VER
ARG TORCH_URL
WORKDIR /wheels

# pip 国内镜像 & 超时
ENV PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ENV PIP_EXTRA_INDEX_URL="https://mirrors.aliyun.com/pypi/simple/ https://pypi.mirrors.ustc.edu.cn/simple/"
ENV PIP_TIMEOUT=120
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -U pip wheel

# 2.1 单独拉 torch CPU wheel（最耗时，放最前）
RUN --mount=type=cache,target=/root/.cache/pip \
    pip wheel --no-deps --wheel-dir /wheels \
        torch==${PYTORCH_VER} -f ${TORCH_URL}

# 2.2 其余依赖（自动复用上面 torch wheel）
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt

# -------------- 3) 最终运行时镜像（最小，无编译工具） --------------
FROM system-base AS runtime
WORKDIR /app

# 非 root 用户
RUN useradd -m -u 1000 -s /bin/bash appuser

# 安装编译好的 wheel（--no-index 禁止联网，纯本地）
COPY --from=python-builder /wheels /wheels
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir --no-index --find-links /wheels -r requirements.txt && \
    rm -rf /wheels

# 代码放最后，日常开发只改这一层
COPY . .
RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]