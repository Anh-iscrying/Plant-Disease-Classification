import streamlit as st
import torch
import timm
from PIL import Image
import torchvision.transforms as transforms

# 1. Danh sách các loiaj bệnh
classes = ['Pepper__bell___Bacterial_spot',
            'Pepper__bell___healthy', 
            'Potato___Early_blight', 
            'Potato___Late_blight', 
            'Potato___healthy', 
            'Tomato_Bacterial_spot', 
            'Tomato_Early_blight', 
            'Tomato_Late_blight', 
            'Tomato_Leaf_Mold', 
            'Tomato_Septoria_leaf_spot', 
            'Tomato_Spider_mites_Two_spotted_spider_mite', 
            'Tomato__Target_Spot', 
            'Tomato__Tomato_YellowLeaf__Curl_Virus', 
            'Tomato__Tomato_mosaic_virus', 
            'Tomato_healthy']

# 2. Cấu hình giao diện
st.set_page_config(page_title="Nhận diện bệnh lá cây", layout="wide")
st.title("🌿 Hệ thống nhận diện bệnh trên lá cây")
st.write("Đồ án: Mạng Nơ-ron và Học sâu")

# Sidebar để chọn Model
st.sidebar.header("Cài đặt mô hình")
model_option = st.sidebar.selectbox(
    "Chọn kiến trúc mạng muốn dùng:",
    ("MobileNetV2 (Baseline)", "ResNet50 (Nâng cao 1)", "EfficientNet-B0 (Nâng cao 2)")
)

# 3. Hàm load model
@st.cache_resource
def load_model(choice):
    if choice == "MobileNetV2 (Baseline)":
        model = timm.create_model('mobilenetv2_100', pretrained=False)
        model.classifier = torch.nn.Linear(model.classifier.in_features, len(classes))
        model.load_state_dict(torch.load("mobilenet_model.pth", map_location='cpu'))
    elif choice == "ResNet50 (Nâng cao 1)":
        model = timm.create_model('resnet50', pretrained=False)
        model.fc = torch.nn.Linear(model.fc.in_features, len(classes))
        model.load_state_dict(torch.load("resnet50_model.pth", map_location='cpu'))
    else: # EfficientNet-B0
        model = timm.create_model('efficientnet_b0', pretrained=False)
        model.classifier = torch.nn.Linear(model.classifier.in_features, len(classes))
        model.load_state_dict(torch.load("efficientnet_model.pth", map_location='cpu'))
    
    model.eval()
    return model

# Nạp model
model = load_model(model_option)

# 4. Tiền xử lý ảnh (Dùng chuẩn ImageNet)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 5. Giao diện upload và dự đoán
uploaded_file = st.file_uploader("Chọn một ảnh lá cây cần kiểm tra...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns(2)
    image = Image.open(uploaded_file).convert("RGB")
    
    with col1:
        st.image(image, caption='Ảnh đã tải lên', use_container_width=True)
    
    # Thực hiện dự đoán
    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        confidence, pred_idx = torch.max(probabilities, 0)
        
    with col2:
        st.subheader("Kết quả dự đoán:")
        st.info(f"Mô hình: **{model_option}**")
        st.success(f"Kết quả: **{classes[pred_idx.item()]}**")
        st.warning(f"Độ tin cậy: **{confidence.item():.2%}**")
        st.progress(confidence.item())