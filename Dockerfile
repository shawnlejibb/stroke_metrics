# Sử dụng Python 3.10 làm image cơ bản
FROM python:3.10-slim

RUN apt-get update && apt-get install -y python3-opencv

# Thiết lập thư mục làm việc trong container
WORKDIR /app

# Sao chép file requirements.txt vào container
COPY requirements.txt .

# Cài đặt các package trong requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Sao chép toàn bộ mã nguồn của ứng dụng vào container
# COPY . .

# Expose port 8000 để server có thể truy cập từ bên ngoài
EXPOSE 8000

# Lệnh mặc định để chạy server FastAPI với uvicorn
# CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]