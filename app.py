import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av

# ==========================================
# 1. CẤU HÌNH GIAO DIỆN PRO
# ==========================================
st.set_page_config(page_title="AI Biển Báo Giao Thông", page_icon="🚦", layout="wide")

st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    h1 {text-align: center; color: #1E88E5; font-family: 'Arial Black', sans-serif;}
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1>🚦 HỆ THỐNG NHẬN DIỆN BIỂN BÁO GIAO THÔNG</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #666; font-size: 18px;'>Tích hợp Trí tuệ Nhân tạo YOLOv8 - Đọc luật giao thông thời gian thực</p>", unsafe_allow_html=True)
st.markdown("---")
 st.markdown("---")
  
# 2. TẢI MÔ HÌNH & TỪ ĐIỂN
# ==========================================
@st.cache_resource
def load_model():
    return YOLO('best.pt')

model = load_model()

tu_dien_yeu_cau = {
    'P-101': 'Yeu cau: Duong cam tat ca phuong tien!',
    'P-102': 'Yeu cau: Duong cam di nguoc chieu!',
    'P-103a': 'Yeu cau: Cam o to di vao!',
    'P-103b': 'Yeu cau: Cam o to re phai!',
    'P-103c': 'Yeu cau: Cam o to re trai!',
    'P-104': 'Yeu cau: Cam mo to di vao!',
    'P-105': 'Yeu cau: Cam o to va mo to!',
    'P-106a': 'Yeu cau: Cam xe tai di vao!',
    'P-106b': 'Yeu cau: Cam xe tai theo tan quy dinh!',
    'P-107a': 'Yeu cau: Cam xe khach va xe tai!',
    'P-112': 'Yeu cau: Cam nguoi di bo!',
    'P-115': 'Yeu cau: Han che trong luong toan bo xe!',
    'P-117': 'Yeu cau: Han che chieu cao xe!',
    'P-123a': 'Yeu cau: Cam re trai!',
    'P-123b': 'Yeu cau: Cam re phai!',
    'P-124a': 'Yeu cau: Cam quay dau xe!',
    'P-124b': 'Yeu cau: Cam o to quay dau xe!',
    'P-124c': 'Yeu cau: Cam quay dau va re trai!',
    'P-125': 'Yeu cau: Cam vuot!',
    'P-126': 'Yeu cau: Cam o to tai vuot!',
    'P-127': 'Yeu cau: Khong vuot qua toc do toi da!',
    'P-128': 'Yeu cau: Cam su dung coi!',
    'P-130': 'Yeu cau: Cam dung xe va do xe!',
    'P-131a': 'Yeu cau: Cam do xe!',
    'P-137': 'Yeu cau: Cam re trai va re phai!',
    'W-201a': 'Chu y: Cho ngoat nguy hiem vong sang trai!',
    'W-201b': 'Chu y: Cho ngoat nguy hiem vong sang phai!',
    'W-201c': 'Chu y: Cho ngoat nguy hiem lien tiep!',
    'W-201d': 'Chu y: Cho ngoat nguy hiem lien tiep!',
    'W-202a': 'Chu y: Nhieu cho ngoat nguy hiem lien tiep!',
    'W-202b': 'Chu y: Nhieu cho ngoat nguy hiem lien tiep!',
    'W-203b': 'Chu y: Duong hep ben trai!',
    'W-203c': 'Chu y: Duong hep ben phai!',
    'W-205a': 'Chu y: Duong giao nhau cung cap!',
    'W-205b': 'Chu y: Duong giao nhau cung cap!',
    'W-205d': 'Chu y: Duong giao nhau cung cap!',
    'W-207a': 'Chu y: Giao nhau voi duong khong uu tien!',
    'W-207b': 'Chu y: Giao nhau voi duong khong uu tien (ben phai)!',
    'W-207c': 'Chu y: Giao nhau voi duong khong uu tien (ben trai)!',
    'W-208': 'Chu y: Giao nhau voi duong uu tien (Phai nhuong duong)!',
    'W-209': 'Chu y: Giao nhau co tin hieu den!',
    'W-210': 'Chu y: Giao nhau voi duong sat co rao chan!',
    'W-219': 'Chu y: Doc xuong nguy hiem!',
    'W-224': 'Chu y: Co nguoi di bo cat ngang qua duong!',
    'W-225': 'Chu y: Tre em qua duong!',
    'W-227': 'Chu y: Cong truong dang thi cong!',
    'W-233': 'Chu y: Nguy hiem khac!',
    'W-235': 'Chu y: Duong doi!',
    'W-245a': 'Chu y: Di cham, chu y quan sat!',
    'R-301c': 'Hieu lenh: Chi duoc re trai!',
    'R-301d': 'Hieu lenh: Chi duoc re phai!',
    'R-301e': 'Hieu lenh: Re trai hoac re phai!',
    'R-302a': 'Hieu lenh: Huong phai di vong sang phai!',
    'R-302b': 'Hieu lenh: Huong phai di vong sang trai!',
    'R-303': 'Hieu lenh: Noi giao nhau chay theo vong xuyen!',
    'R-407a': 'Hieu lenh: Duong 1 chieu!',
    'R-409': 'Hieu lenh: Cho quay dau xe!',
    'R-425': 'Hieu lenh: Duong danh cho o to!',
    'R-434': 'Hieu lenh: Tram xe buyt!',
    'DP-135': 'Chi dan: Het tat ca lenh cam!',
    'S-509a': 'Bien phu: Huong duong uu tien!'
}

# ==========================================
# 3. THANH BÊN (SIDEBAR) & CHỌN CHẾ ĐỘ
# ==========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3253/3253083.png", width=120)
    st.title("⚙️ Bảng Điều Khiển")
    
    st.markdown("**1. Chọn nguồn ảnh:**")
    lua_chon = st.radio("", ("🖼️ Phân tích Ảnh Tĩnh", "🎥 Camera Thời Gian Thực"))
    File "/mount/src/nhan-dien-bien-bao/app.py", line 25
   st.markdown("---")
   st.markdown("---")
 st.expander("ℹ️ Giới thiệu dự án"):
        st.write("Dự án sử dụng mô hình YOLOv8 để nhận diện biển báo và tự động hiển thị luật giao thông bằng tiếng Việt trực tiếp lên khung hình.<br>Thành viên:<br> Nguyễn Sỹ Tấn Tài <br> Lương Văn Trường <br> Trần Quốc Toàn", unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("**2. Bộ lọc hình ảnh:**")
    bat_lam_net = st.checkbox("✨ Tự động làm nét (Dùng cho ảnh mờ)", value=False)

# ==========================================
# 4. HÀM XỬ LÝ LÕI AI
# ==========================================
def lam_net_anh(image):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)

def phan_tich_va_ve(anh_bgr):
    ket_qua = model.predict(source=anh_bgr, conf=0.15, verbose=False)
    anh_hien_thi = ket_qua[0].plot()
    cac_bien = ket_qua[0].boxes.cls
    ds_nhan = model.names
    
    h, w = anh_hien_thi.shape[:2]
    toa_do_y = h - 30 
    toa_do_x = 30
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.6, w / 1200) 
    
    for nhan in cac_bien:
        ten_goc = ds_nhan[int(nhan)]
        cau_yeu_cau = f" -> {tu_dien_yeu_cau.get(ten_goc, f'Bien {ten_goc} (Chua dich)')}"
        (tw, th), baseline = cv2.getTextSize(cau_yeu_cau, font, font_scale, 2)
        cv2.rectangle(anh_hien_thi, (toa_do_x-10, toa_do_y-th-baseline-10), 
                      (toa_do_x+tw+10, toa_do_y+baseline+10), (0, 255, 255), -1)
        cv2.putText(anh_hien_thi, cau_yeu_cau, (toa_do_x, toa_do_y), font, font_scale, (0,0,0), 2)
        toa_do_y -= (th + baseline + 30)
        
    return anh_hien_thi

# ==========================================
# 5. GIAO DIỆN HIỂN THỊ
# ==========================================
if lua_chon == "🖼️ Phân tích Ảnh Tĩnh":
    uploaded_file = st.file_uploader("Kéo thả hoặc chọn ảnh từ máy tính...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        col1, col2 = st.columns(2) 
        image_pil = Image.open(uploaded_file)
        anh_goc_rgb = np.array(image_pil)
        
        if bat_lam_net:
            anh_goc_rgb = lam_net_anh(anh_goc_rgb)
            
        anh_goc_bgr = cv2.cvtColor(anh_goc_rgb, cv2.COLOR_RGB2BGR)
        
        with col1:
            st.info("📸 Ảnh Đầu Vào")
            st.image(anh_goc_rgb)

        with col2:
            st.success("🤖 AI Phân Tích")
            with st.spinner("AI đang tính toán..."):
                anh_ket_qua_bgr = phan_tich_va_ve(anh_goc_bgr)
                st.image(cv2.cvtColor(anh_ket_qua_bgr, cv2.COLOR_BGR2RGB))

elif lua_chon == "🎥 Camera Thời Gian Thực":
    st.info("💡 Bấm 'START' và cấp quyền truy cập Camera. Đưa biển báo vào khung hình, AI sẽ nhận diện liên tục!")
    
    # Cấu hình máy chủ trung gian để luồng video webRTC không bị chặn
    RTC_CONFIGURATION = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )
    
    # Hàm xử lý từng khung hình video theo thời gian thực
    def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
        img_bgr = frame.to_ndarray(format="bgr24")
        
        if bat_lam_net:
            img_bgr = lam_net_anh(img_bgr)
            
        img_ket_qua_bgr = phan_tich_va_ve(img_bgr)
        return av.VideoFrame.from_ndarray(img_ket_qua_bgr, format="bgr24")

    # Bật khung phát video stream
    webrtc_streamer(
        key="nhan-dien-realtime",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_frame_callback=video_frame_callback,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
