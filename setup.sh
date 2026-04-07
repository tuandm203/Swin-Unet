# 1. Cập nhật hệ thống & Cài Python venv
sudo apt update && sudo apt install python3-venv python3-pip -y

# 2. Tìm ổ đĩa lớn (Thường là /workspace) và vào đó
# Nếu máy bạn thuê gắn ổ 368GB ở chỗ khác, hãy đổi tên thư mục /workspace
cd /workspace

# 3. Tạo môi trường ảo SẠCH
python3 -m venv swin_env
source swin_env/bin/activate

# 4. Cài đặt PyTorch chuẩn CUDA 12.4 (Không bao giờ lỗi Driver nữa)
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 --no-cache-dir

# 5. Clone code và cài requirements
# Thay link repo của bạn vào đây
git clone https://github.com/tuandm203/Swin-Unet.git
cd Swin-Unet
pip install -r requirements.txt --no-cache-dir

# 6. Kiểm tra GPU
python -c "import torch; print('CUDA Ready:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0))"