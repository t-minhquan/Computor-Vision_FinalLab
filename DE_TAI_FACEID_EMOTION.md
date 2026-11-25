# BÁO CÁO ĐỀ TÀI BÀI TẬP LỚN

- **Motivation/Background**: Biểu cảm khuôn mặt khó phân loại khi ánh sáng không đồng đều và bóng đổ gắt.
- **Purpose**: Đề xuất hệ thống nhận diện 7 cảm xúc khuôn mặt (FER).
- **Overview Pipeline**: Cắt mặt → Gamma → DoG → chuẩn hóa tương phản → Spatial-LBP → SVM.
- **Detailed Plan**: Thách thức, giải pháp, tiêu chí đánh giá, dữ liệu, thí nghiệm E1–E4 và tham khảo.
- **References**: [Kiến trúc pipeline](PIPELINE_ARCHITECTURE.md), [hướng dẫn BTL](TOM_TAT_HUONG_DAN_BTL.md), [quy ước Markdown](agents/rules/MD_CONVENTION.md).
- **Created**: 2026-11-23T23:07:16+07:00
- **Last Updated**: 2026-11-25T17:19:50+07:00

---

# MÔN HỌC: XỬ LÝ ẢNH VÀ THỊ GIÁC MÁY TÍNH (121036)
**Trường Đại học Giao thông Vận tải TP. Hồ Chí Minh (UTH)**  
**Đề tài:** Nhận diện cảm xúc khuôn mặt trong điều kiện ánh sáng không đồng đều

**Trạng thái:** Đề xuất nghiên cứu; chưa có kết quả thực nghiệm. Các tham số và quy mô bên dưới là cấu hình dự kiến cần kiểm chứng.

---

## I. TUYÊN BỐ THÁCH THỨC (BẮT BUỘC)

### A. Bối cảnh và Thách thức
* **1. Tác vụ giải quyết:**  
  Xây dựng hệ thống thị giác máy tính nhận diện cảm xúc từ biểu cảm khuôn mặt:
  - **FER (Facial Emotion Recognition):** Phân loại trạng thái cảm xúc khuôn mặt (7 lớp: Vui vẻ, Buồn bã, Giận dữ, Ngạc nhiên, Sợ hãi, Ghê tởm, Bình thường).
  - **Ứng dụng thực tế:** Demo phân loại biểu cảm khuôn mặt từ ảnh hoặc webcam khi ánh sáng thay đổi. Nhãn biểu cảm không dùng để kết luận mức độ chú ý hoặc trạng thái tâm lý của người được chụp.

* **2. Dữ liệu thực tế:**  
  - Dữ liệu dự kiến: *Real-world Affective Faces Database (RAF-DB)* và ảnh tự chụp bằng webcam/camera điện thoại tại môi trường lớp học có cửa sổ đón nắng xiên và đèn trần. Mỗi ảnh phải có nhãn một trong 7 cảm xúc.
  - Quy mô: Tách biệt rõ ràng tập phát triển (Development set - 70%) dùng để khảo sát tham số và tập đánh giá (Evaluation set - 30%) được phân tầng độc lập.
  - Chia tập trước khi tạo biến thể ánh sáng; giữ ảnh cùng người/phiên chụp trong cùng tập khi có thông tin nhóm. Tập phát triển dùng huấn luyện và chọn tham số qua validation; tập đánh giá chỉ dùng sau khi khóa cấu hình.

* **3. Thách thức chính ($C$):**  
  **Chiều khó: "Ánh sáng không đồng đều và bóng đổ gắt" (Non-uniform Illumination and Harsh Shadows)**.  
  - Trong điều kiện thực tế, nguồn sáng thường đến từ một phía (ánh sáng xiên từ cửa sổ, đèn rọi lệch góc), tạo ra sự chênh lệch cường độ sáng cực lớn giữa hai nửa khuôn mặt.
  - Xuất hiện các vùng bóng đổ sâu (shadows) tại hốc mắt, cánh mũi, cằm và vùng cháy sáng (specular highlights) tại trán, gò má.

* **4. Kiểu hạn chế ($F$ - Cách làm thông thường ngừng hoạt động như thế nào):**  
  - Khi đưa ảnh thô vào bộ mô tả gradient toàn cục HOG:
    - **Đối với FER:** Bóng đổ tạo ra các cạnh nhân tạo (false edges) với độ dốc gradient rất lớn tại ranh giới vùng tối. Bộ phân loại cảm xúc nhầm lẫn bóng đổ ở khóe miệng/hốc mắt với biểu cảm giận dữ (Anger) hoặc buồn bã (Sadness), khiến độ chính xác FER sụt giảm nghiêm trọng.

* **5. Giả định thu hẹp:**  
  - Để tập trung toàn bộ năng lực giải quyết thách thức ánh sáng $C$, bài toán cố định các chiều khó khác bằng giả định kỹ thuật hợp lý:
    - **Góc nhìn (Pose):** Cố định người dùng nhìn trực diện camera (góc quay yaw, pitch, roll trong khoảng dải hẹp $[-15^\circ, +15^\circ]$).
    - **Khoảng cách & Tỷ lệ:** Người dùng đứng cách camera ở khoảng cách tiêu chuẩn $0.5\text{m} - 1.2\text{m}$.
    - **Không bị che khuất (Occlusion):** Khuôn mặt không bị che bởi khẩu trang hoặc kính râm.
  - *Lý do chấp nhận giả định:* Demo thu nhận biểu cảm có hướng dẫn, người tham gia chủ động nhìn về camera. Báo cáo riêng những ảnh RAF-DB không đáp ứng giả định.

---

### B. Lý thuyết của Giải pháp Đề xuất ($P$)
* **1. Kỹ thuật đề xuất ($P$):**  
  Pipeline chuẩn hóa chiếu sáng đa tầng (**Multi-stage Illumination Normalization Pipeline**) kết hợp bộ mô tả bất biến cục bộ (**Local Invariant Descriptors - Uniform LBP**):
  - Chuỗi tiền xử lý giảm ảnh hưởng chiếu sáng (Illumination Normalization): **Gamma Correction ($\gamma$-compression)** $\rightarrow$ **Lọc thông dải Difference of Gaussians (DoG bandpass filtering)** $\rightarrow$ **Chuẩn hóa tương phản bền vững (Robust Contrast Normalization)**.
  - Trích chọn đặc trưng phân vùng bất biến biến đổi độ sáng đơn điệu: **Spatially Enhanced Uniform Local Binary Patterns (Spatial-LBP)**.
  - Phân loại: Support Vector Machine đa lớp (SVM kernel RBF).

* **2. Cơ chế giải quyết ($P$ giải quyết $C$ như thế nào):**  
  - Theo mô hình phản xạ quang học Lambertian, một ảnh số $I(x,y)$ được tạo thành từ tích của hai thành phần:
    $$I(x, y) = L(x, y) \times R(x, y)$$
    Trong đó $L(x, y)$ là thành phần chiếu sáng môi trường (Illuminance) và $R(x, y)$ là thành phần phản xạ đặc trưng của bề mặt khuôn mặt (Reflectance - chứa chi tiết biểu cảm).
  - Thành phần chiếu sáng $L(x,y)$ biến thiên chậm trong không gian (tần số thấp) và gây ra bóng đổ diện rộng. Ngược lại, thông tin nhận dạng $R(x,y)$ (kết cấu da, viền mắt, nếp nhăn biểu cảm) nằm ở dải tần số không gian trung bình và cao.
  - **Gamma Correction ($I' = I^\gamma$ với $\gamma < 1$):** Nén dải động phi tuyến, kéo sáng các vùng bóng đổ tối mà không làm cháy trắng các vùng sáng.
  - **Bộ lọc Difference of Gaussians (DoG):**
    $$DoG(x, y) = \frac{1}{2\pi\sigma_1^2} e^{-\frac{x^2+y^2}{2\sigma_1^2}} - \frac{1}{2\pi\sigma_2^2} e^{-\frac{x^2+y^2}{2\sigma_2^2}} \quad (\sigma_1 < \sigma_2)$$
    Tác động trực tiếp lên phổ tần số: Giảm thành phần biến thiên chậm và suy giảm nhiễu tần số cao. Ranh giới bóng đổ gắt có thể còn lại vì chứa cả tần số trung bình và cao.
  - **Toán tử Uniform LBP:** So sánh thứ tự cường độ tương đối giữa pixel tâm $g_c$ và các lân cận $g_p$:
    $$LBP_{P, R} = \sum_{p=0}^{P-1} s(g_p - g_c) 2^p, \quad s(x) = \begin{cases} 1 & x \ge 0 \\ 0 & x < 0 \end{cases}$$
    Phép so sánh LBP giữ nguyên dưới biến đổi đơn điệu tăng nghiêm ngặt áp dụng chung cho các pixel. Tính chất này không bảo đảm bất biến với bóng đổ cục bộ, nhiễu hoặc clipping.

* **3. Dự đoán điểm thất bại (Failure Point Prediction):**  
  - *Giả thuyết trước thực nghiệm:* Thiếu sáng, nhiễu và clipping làm mất chi tiết mắt, miệng và nếp nhăn; chuẩn hóa có thể khuếch đại nhiễu thay vì phục hồi biểu cảm. FER được dự đoán giảm Accuracy và Macro-F1 khi mức khó tăng. Chưa xác định ngưỡng thất bại; không suy ra lux hoặc SNR chỉ từ giá trị pixel.

---

### C. Tiêu chí Kiểm chứng
* **1. Thước đo định lượng:**
  - **FER:**
    - Accuracy tổng thể (%).
    - F1-Score vĩ mô (Macro F1-score) cho 7 lớp cảm xúc để tránh lệch lớp.
    - Thời gian trích xuất & suy luận trung bình mỗi khung hình (Inference latency, ms/frame).
* **2. Baseline so sánh:**
  - *Baseline FER:* Trích xuất HOG toàn cục + SVM trên ảnh xám thô.
  - Cấu hình baseline cố định: ảnh 128×128, HOG cell 16×16, block 32×32, stride 16×16, 9 hướng và L2-Hys, tắt gamma của OpenCV; 1.764 đặc trưng. SVM RBF với C=10, gamma=scale và class_weight=balanced, cùng chính sách cấu hình bộ phân loại của pipeline LBP.
  - [Mã HOG](src/feature_extraction/hog_descriptor.py), [train](src/training/train.py), [đánh giá](src/evaluation/evaluate.py) dùng chung quy trình dữ liệu và cắt mặt. Đây là so sánh toàn pipeline; ablation vẫn cần để đo riêng đóng góp chuẩn hóa ánh sáng.

---

## II. CÁC TRỤC ĐÁNH GIÁ VÀ ĐỊNH VỊ DỰ ÁN

| Trục | Trọng số | Định vị thực hiện trong Đề tài |
| :--- | :---: | :--- |
| **1. Mức ý nghĩa của thách thức** | **25%** | Kiểm chứng bằng **E1**: đo Accuracy và Macro-F1 của HOG + SVM trên từng tầng ánh sáng; phân tích 3 ca lỗi tiêu biểu nếu có. Không ấn định trước mức sụt giảm. |
| **2. Độ vững chắc & tính mới lý thuyết** | **35%** | Lập luận từ mô hình Lambertian $I = L \times R$: **Chương 2** (Gamma, DoG, chuẩn hóa tương phản), **Chương 3** (Spatial-LBP), **Chương 5** (SVM đa lớp). Đánh giá đóng góp của chuỗi xử lý qua khảo sát tham số và ablation, chưa khẳng định mức tính mới. |
| **3. Độ chặt chẽ thực nghiệm** | **30%** | Phân tầng dữ liệu độc lập thành 3 mức $C_1, C_2, C_3$ theo độ lệch chuẩn chiếu sáng $\sigma_{\text{lum}}$ và tỷ lệ diện tích bóng đổ. Thực hiện đầy đủ 4 thí nghiệm bắt buộc **E1, E2, E3, E4**. Phân tích định lượng và báo cáo chi tiết theo từng tầng. |
| **4. Chất lượng trình bày** | **10%** | Báo cáo cấu trúc 12 mục chuẩn mực, hình ảnh trực quan chất lượng cao, trích dẫn học thuật định dạng chuẩn IEEE. |

> [!WARNING]
> **Quy định nghiêm ngặt về mô hình dựng sẵn:**  
> Hệ thống sử dụng Haar Cascade như một **công cụ tiện ích** để cắt vùng mặt (crop ROI). Phần xử lý thách thức ánh sáng (Gamma, DoG, chuẩn hóa tương phản, Spatial-LBP) được cài đặt bằng OpenCV/NumPy/SciPy; SVM được huấn luyện trên tập phát triển bằng scikit-learn. Không sử dụng mô hình Deep Learning dựng sẵn làm thành phần khắc phục $C$.

---

## III. DỮ LIỆU ẢNH THỰC TẾ VÀ PHÂN TẦNG

### 1. Nguồn dữ liệu
- **Tập Real-world Affective Faces Database (RAF-DB):** Bộ dữ liệu cảm xúc thực tế chứa các ảnh có ánh sáng phức tạp tự nhiên.
- **Tập kiểm thử hiện trường (In-house Classroom Dataset):** Dự kiến 10 tình nguyện viên thể hiện 7 biểu cảm dưới 3 điều kiện: đèn chuẩn, ánh sáng nghiêng $45^\circ$, và ngược sáng. Ghi nhãn cảm xúc theo biểu cảm được hướng dẫn, kiểm tra thủ công và loại ảnh mơ hồ.

### 2. Tiêu chí phân tầng thách thức $C$
Mỗi ảnh khuôn mặt sau khi chuẩn hóa kích thước ($128 \times 128$) được tính hai chỉ số định lượng:
1. Độ lệch chuẩn cường độ xám: $\sigma_{\text{lum}} = \sqrt{\frac{1}{N}\sum (I_i - \bar{I})^2}$
2. Tỷ lệ diện tích bóng đổ tối: $R_{\text{shadow}} = \frac{N(I_i < 40)}{N_{\text{total}}}$

| Tầng | Tên tầng | Tiêu chuẩn định lượng | Biểu hiện trực quan | Mục tiêu số ảnh |
| :---: | :--- | :--- | :--- | :---: |
| **$C_1$** | Nhẹ (Uniform) | $\sigma_{\text{lum}} \le 25$ và $R_{\text{shadow}} \le 5\%$ | Ánh sáng phẳng, chiếu trực diện, không có bóng tối cục bộ | 400 |
| **$C_2$** | Trung bình (Side-light) | Không thuộc $C_1$ hoặc $C_3$ | Nguồn sáng nghiêng $45^\circ$, bóng đổ một bên sống mũi/má | 400 |
| **$C_3$** | Nặng (Harsh / Backlit) | $\sigma_{\text{lum}} > 50$ hoặc $R_{\text{shadow}} > 25\%$ | Ngược sáng cực mạnh hoặc bóng đổ che khuất trên 1/3 khuôn mặt | 400 |

Các ngưỡng là quy ước dự kiến, tính trên ảnh xám trước chuẩn hóa ánh sáng; chúng chỉ là chỉ báo gần đúng, cần đối chiếu trực quan. Báo cáo số ảnh thực thu theo tầng và lớp, không coi 400 ảnh/tầng là dữ liệu đã có.

---

## IV. BỐN THÍ NGHIỆM BẮT BUỘC (E1 – E4)

* **Thí nghiệm E1 (Chứng cứ thách thức):**
  - Đánh giá baseline HOG thô + SVM trên 3 tầng $C_1, C_2, C_3$.
  - Mục tiêu: Kiểm tra giả thuyết hiệu năng FER giảm khi $C$ tăng, báo cáo cả trường hợp giả thuyết không được xác nhận.
  - Trình bày 3 cặp ảnh nhận diện sai tiêu biểu cùng biểu đồ gradient giả do bóng đổ gây ra.

* **Thí nghiệm E2 (Khảo sát tham số tiền xử lý và trích đặc trưng):**
  - Khảo sát hệ số nén dải động Gamma $\gamma \in [0.1, 0.2, 0.5, 1.0]$.
  - Khảo sát độ lệch chuẩn Gaussian $\sigma_1 \in \{1.0, 1.5, 2.0\}$ và $\sigma_2 \in \{2.0, 4.0, 8.0\}$, chỉ xét $\sigma_1 < \sigma_2$ trong bộ lọc DoG.
  - Khảo sát cấu hình bán kính và số lân cận của LBP ($P=8, R=1$ so với $P=16, R=2$).
  - Xuất ảnh trung gian: Ảnh gốc $\rightarrow$ Gamma $\rightarrow$ DoG $\rightarrow$ chuẩn hóa tương phản $\rightarrow$ bản đồ LBP. Thay đổi một yếu tố mỗi lượt, giữ các yếu tố còn lại cố định; chọn tham số trên validation trong tập phát triển.

* **Thí nghiệm E3 (Ablation Study - Đánh giá loại bỏ thành phần):**
  - Tiến hành tắt lần lượt:
    1. Bỏ bước lọc DoG (giữ Gamma + chuẩn hóa tương phản + Spatial-LBP + SVM).
    2. Bỏ toàn bộ chuỗi tiền xử lý ánh sáng (chạy trực tiếp LBP trên ảnh thô).
    3. Bỏ đặc tính Spatial LBP (chỉ lấy histogram toàn cục).
  - So sánh Accuracy và Macro-F1 với pipeline đầy đủ trên cùng tập đánh giá, báo cáo chênh lệch thực đo cho từng biến thể. Giữ split và cấu hình SVM cố định, huấn luyện lại cho mỗi biểu diễn đặc trưng.

* **Thí nghiệm E4 (Kiểm tra điểm gãy - Failure Mode Analysis):**
  - Tăng độ khó nhân tạo bằng cách giảm cường độ sáng trung bình của ảnh xuống các mức cực hạn ($[20, 15, 10, 5, 2, 0]$ giá trị pixel) và tăng tỷ lệ bóng tối nhân tạo.
  - Chỉ tạo biến thể từ ảnh tập đánh giá, không thay đổi nhãn hoặc huấn luyện lại. Vẽ Accuracy và Macro-F1 theo mức suy giảm, phân tích các ca mất chi tiết và so với giả thuyết ở Mục I.B.3. Không đặt trước ngưỡng thất bại hoặc quy đổi pixel thành lux.

---

## V. TÀI LIỆU THAM KHẢO CHUẨN IEEE

1. [1] X. Tan and B. Triggs, "Enhanced local texture feature sets for face recognition under difficult lighting conditions," *IEEE Transactions on Image Processing*, vol. 19, no. 6, pp. 1635-1650, June 2010, doi: 10.1109/TIP.2010.2042645.
2. [2] T. Ahonen, A. Hadid, and M. Pietikainen, "Face description with local binary patterns: Application to face recognition," *IEEE Transactions on Pattern Analysis and Machine Intelligence*, vol. 28, no. 12, pp. 2037-2041, Dec. 2006, doi: 10.1109/TPAMI.2006.244.
3. [3] C. Shan, S. Gong, and P. W. McOwan, "Facial expression recognition based on Local Binary Patterns: A comprehensive study," *Image and Vision Computing*, vol. 27, no. 6, pp. 803-816, May 2009, doi: 10.1016/j.imavis.2008.08.005.
