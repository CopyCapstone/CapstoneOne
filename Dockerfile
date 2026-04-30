FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /app

# Install OpenCV system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libxcb1 \
    && rm -rf /var/lib/apt/lists/*

# Step 1: Copy ONLY the requirements file first
COPY requirements.txt .

# Step 2: Install dependencies 
RUN pip install -r requirements.txt

# Step 3: Safely copy the rest of your app's code
# Docker will look at your .dockerignore and skip the unwanted files
COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]