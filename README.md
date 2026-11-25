# Ứng dụng nhận diện cảm xúc khuôn mặt

- **Motivation/Background**: Xây dựng backend cho đề tài FER dưới ánh sáng không đồng đều.
- **Purpose**: Hướng dẫn chạy giao diện thử ảnh, huấn luyện và gọi API nhận diện 7 cảm xúc.
- **Overview Pipeline**: JPEG/PNG → Haar Cascade → Gamma → DoG → chuẩn hóa tương phản → Spatial-LBP → SVM.
- **Detailed Plan**: Cài đặt, API, dữ liệu huấn luyện, kiểm thử và giới hạn hiện tại.
- **References**: [Đề tài](DE_TAI_FACEID_EMOTION.md), [kiến trúc](PIPELINE_ARCHITECTURE.md), [FastAPI upload](https://fastapi.tiangolo.com/tutorial/request-files/), [giới hạn request](https://fastapi.tiangolo.com/advanced/request-body-limits/).
- **Created**: 2026-11-23T23:23:17+07:00
- **Last Updated**: 2026-11-25T17:19:50+07:00

---

## 1. Cài đặt và chạy

Dùng Python 3.12, chạy lệnh từ thư mục gốc dự án trên PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements-dev.txt
.venv\Scripts\python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

Mở [giao diện thử ảnh](http://127.0.0.1:8000/) để chọn JPEG/PNG, xem trước và nhận diện cảm xúc. Frontend dùng HTML/CSS/JavaScript thuần, chạy cùng backend; không cần npm hoặc cổng riêng. Nếu server đang chạy từ trước khi cập nhật, dừng bằng Ctrl+C rồi chạy lại lệnh uvicorn trên.

Giao diện hiển thị trạng thái model, kết quả tiếng Việt, thời gian xử lý và vùng mặt. Khi chưa có model, bạn vẫn xem trước ảnh được nhưng nút nhận diện bị khóa; huấn luyện và khởi động lại backend rồi nhấn **Kiểm tra lại**. Backend thiếu model trả HTTP 503. [Swagger UI](http://127.0.0.1:8000/docs) vẫn dùng được để thử API. Chỉ cần [requirements.txt](requirements.txt) khi chạy ứng dụng, [requirements-dev.txt](requirements-dev.txt) thêm công cụ kiểm thử.

## 2. API

| Method | Endpoint | Kết quả |
| :--- | :--- | :--- |
| GET | `/` | Giao diện thử nhận diện từ ảnh |
| GET | `/health` | `status` và `model_ready` |
| GET | `/api/v1/emotions` | Danh sách 7 nhãn cảm xúc |
| POST | `/api/v1/predict` | Multipart có trường `file`; trả `emotion`, `face_box`, `latency_ms` |

```powershell
curl.exe -X POST http://127.0.0.1:8000/api/v1/predict -F "file=@data/sample_test/face.jpg"
```

Ví dụ cấu trúc phản hồi, không phải kết quả đã đo:

```json
{
  "emotion": "happiness",
  "face_box": {"x": 120, "y": 80, "width": 160, "height": 160},
  "latency_ms": 25.0
}
```

`face_box` dùng tọa độ pixel trên ảnh gốc; `latency_ms` đo giải mã ảnh, phát hiện mặt, trích đặc trưng và dự đoán, không gồm thời gian upload. Không xuất xác suất chưa được hiệu chuẩn.

**Khi thử ảnh FER2013 hoặc ảnh mặt đã cắt sẵn:** bật **Ảnh đã cắt mặt (FER2013)** trên giao diện. API tương ứng là `POST /api/v1/predict?cropped_faces=true`. Chế độ này dùng nguyên vùng ảnh, chuyển xám và resize giống lúc train; không chạy Haar hoặc xác minh ảnh có đúng một mặt. `face_box` khi đó là toàn ảnh. Không bật cho ảnh toàn cảnh hoặc ảnh có nhiều người. Ảnh upload thông thường vẫn dùng chế độ mặc định để phát hiện và cắt mặt; không tự động bỏ qua detector khi không tìm thấy mặt.

CLI cũng hỗ trợ ảnh đã cắt:

```powershell
.venv\Scripts\python -m src.pipeline --input data/sample_test/face.jpg --cropped-faces
```

- 413: ảnh lớn hơn 5 MiB hoặc request lớn hơn 5 MiB + 64 KiB cho multipart.
- 422: thiếu trường `file`, ảnh lỗi, không phải JPEG/PNG, trên 12 megapixel, không có mặt hoặc nhiều mặt.
- 503: chưa có model đã huấn luyện. Trạng thái này được kiểm tra trước khi giải mã ảnh.
- Request chỉ nhận một file, không có trường văn bản. Backend không lưu ảnh upload và tên file không dùng để tạo đường dẫn.

## 3. Chuẩn bị dữ liệu và huấn luyện

**Với FER2013 đã giải nén vào thư mục dữ liệu thô:** giữ nguyên hai thư mục `train` và `test`, chạy trực tiếp:

```powershell
.venv\Scripts\python -m src.training.train --data-dir data/raw/train --cropped-faces
```

Script chấp nhận tên lớp FER2013 `angry`, `happy`, `sad` và map về nhãn ứng dụng `anger`, `happiness`, `sadness`; các nhãn còn lại giữ nguyên. Cờ `--cropped-faces` dùng ảnh mặt đã cắt sẵn, chuyển xám và resize về 128×128, bỏ Haar trong lúc train. Không dùng cờ này cho ảnh toàn cảnh chưa cắt mặt. API dùng cùng bước chuẩn bị ảnh với train khi chọn chế độ ảnh đã cắt; chế độ ảnh upload thông thường vẫn dùng Haar.

Lệnh không sửa hoặc sao chép dữ liệu thô, chỉ huấn luyện từ phần train. Giữ phần test để đánh giá sau, không trộn vào train. Đây là cấu hình khởi đầu để chạy demo; chưa khảo sát tham số hoặc đo chất lượng. Nếu cần chọn tham số, tách validation từ phần train trước. Việc resize ảnh 48×48 không khôi phục chi tiết đã mất.

Thư mục [dữ liệu thô](data/raw/) giữ nguyên ảnh gốc. Chuẩn bị [tập train](data/processed/train/) với đúng 7 thư mục con:

```text
train/
├── neutral/
├── happiness/
├── sadness/
├── surprise/
├── anger/
├── disgust/
└── fear/
```

Mỗi lớp cần ít nhất hai ảnh JPEG/PNG có đúng một mặt phát hiện được. Đây chỉ là điều kiện chạy thuật toán, không đủ để bảo đảm chất lượng model. Nhãn lấy từ tên thư mục; ảnh lỗi hoặc không đúng số mặt khiến lệnh dừng và nêu tên ảnh để sửa, không âm thầm bỏ mẫu.

Chia người/phiên chụp trước khi tạo biến thể ánh sáng. Chỉ đưa phần huấn luyện của tập phát triển vào lệnh dưới; giữ [validation](data/processed/validation/) để chọn tham số và [evaluation](data/processed/evaluation/) để đánh giá cuối. Backend chưa tự chia tập hoặc tải RAF-DB.

```powershell
.venv\Scripts\python -m src.training.train --data-dir data/processed/train
.venv\Scripts\python -m src.pipeline --input data/sample_test/face.jpg
```

Model mặc định lưu cục bộ trong [thư mục model](models/), tên `fer.joblib`. Lệnh từ chối ghi đè model cũ. Muốn lưu lần chạy khác, dùng `--output models/fer_run2.joblib`. Chọn model khác cho API rồi khởi động lại:

```powershell
$env:FER_MODEL_PATH = (Resolve-Path models/fer_run2.joblib).Path
.venv\Scripts\python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

Chỉ nạp model joblib do mình huấn luyện hoặc tin cậy: định dạng này có thể thực thi mã khi đọc. API không nhận upload model. Model sai định dạng hoặc sai phiên bản đặc trưng khiến startup thất bại; đường dẫn `FER_MODEL_PATH` không tồn tại cũng báo lỗi.

Lệnh huấn luyện dùng SVM RBF với `C=10`, `gamma=scale`, trọng số lớp cân bằng; đây là cấu hình khởi đầu, chưa khảo sát E2. Model lưu phiên bản đặc trưng để kiểm tra tương thích khi nạp. Lệnh in số mẫu mỗi lớp, không tuyên bố độ chính xác từ dữ liệu train.

## 4. Đánh giá toàn bộ tập test

Dùng model đã train để chạy mọi ảnh trong thư mục test, không huấn luyện hoặc tinh chỉnh tham số:

```powershell
.venv\Scripts\python -m src.evaluation.evaluate --model models/fer.joblib --data-dir data/raw/test --cropped-faces
```

Cần giữ `--cropped-faces` với FER2013 để xử lý ảnh giống lúc train. Script [đánh giá](src/evaluation/evaluate.py) in tiến độ mỗi 50 ảnh và tạo thư mục kết quả riêng theo thời gian trong [experiments/results](experiments/results/). Có thể chỉ định thư mục mới bằng `--output-dir`; lệnh từ chối ghi đè kết quả cũ.

- `summary.json`: Accuracy, Macro-F1, Weighted-F1, số ảnh lỗi, thời gian xử lý và thông tin môi trường/model.
- `classification_report.csv`: Precision, Recall, F1 và số ảnh từng lớp.
- `confusion_matrix.csv`: Hàng là nhãn thật, cột là dự đoán; có thêm cột lỗi xử lý nếu cần.
- `predictions.csv`: Tên ảnh, nhãn thật/dự đoán, đúng/sai, độ trễ hoặc lỗi; ghi dần khi chạy.
- `REPORT.md`: Báo cáo đọc trực tiếp với bảng chỉ số và ma trận nhầm lẫn.

Accuracy tính trên **toàn bộ ảnh**, gồm cả ảnh không xử lý được như một dự đoán sai. Macro-F1 dùng đủ bảy lớp; ảnh lỗi vẫn tạo false negative cho lớp thật. Độ trễ chỉ tính các lần pipeline thành công, không gồm đọc file hoặc nạp model. Đây là đánh giá tổng thể trên thư mục được chỉ định, chưa phải baseline hoặc E1–E4 phân tầng ánh sáng. Không dùng ảnh train để kết luận khả năng tổng quát hóa.

## 5. Baseline HOG + SVM

Baseline dùng ảnh xám 128×128 trực tiếp, không chạy Gamma, DoG hoặc chuẩn hóa tương phản của phương pháp đề xuất. HOG dùng cell 16×16, block 32×32, stride 16×16, 9 hướng, L2-Hys và tắt gamma của OpenCV; tổng 1.764 đặc trưng. Cấu hình này được cố định trước khi đánh giá, không chọn theo điểm test. SVM giữ cùng chính sách với LBP: kernel RBF, `C=10`, `gamma=scale`, trọng số lớp cân bằng. [HOG OpenCV](https://docs.opencv.org/4.12.0/d5/d33/structcv_1_1HOGDescriptor.html).

**Train trên máy:**

```powershell
.venv\Scripts\python -m src.training.train --data-dir data/raw/train --cropped-faces --feature-type hog --output models/hog_baseline.joblib
```

**Đánh giá trên cùng tập test:**

```powershell
.venv\Scripts\python -m src.evaluation.evaluate --model models/hog_baseline.joblib --feature-type hog --data-dir data/raw/test --cropped-faces
```

Train mặc định với `--feature-type hog` cũng lưu tên model riêng `hog_baseline.joblib`; không ghi đè model LBP. Trích đặc trưng báo tiến độ mỗi 500 ảnh và SVM in log LIBSVM. Log không phải phần trăm fit hoặc khả năng resume. Train toàn bộ vẫn có thể lâu; chọn GPU không tự tăng tốc SVC.

**Train trên Colab:** mở [notebook baseline](notebooks/02_train_hog_baseline_colab.ipynb), upload gói baseline mới rồi chạy từng ô. Gói ZIP LBP cũ không chứa code HOG, nên cần gói mới; tạo lại bằng lệnh sau nếu cần:

```powershell
.venv\Scripts\python -m src.training.package_colab --output .tmp/fer2013_hog_colab.zip
```

Notebook dùng thư mục Colab riêng và lưu model hoàn tất vào `MyDrive/FER_Lab/HOG_Baseline`. Tải model về thư mục model local với tên `hog_baseline.joblib`, rồi dùng lệnh đánh giá trên. Nếu gói ZIP đã tồn tại, đổi tên `--output` để giữ bản cũ.

So sánh `summary.json` và báo cáo từng lớp của hai lần đánh giá: cùng danh sách ảnh, nhãn và chế độ `--cropped-faces`, không điều chỉnh tham số từ kết quả test. Báo cáo mới ghi cả loại/phiên bản đặc trưng và tham số SVM. Baseline HOG là mốc so sánh toàn pipeline; để khẳng định đóng góp riêng của chuẩn hóa ánh sáng vẫn cần ablation LBP có/không chuẩn hóa và E1–E4 theo tầng ánh sáng.

Baseline hiện phục vụ CLI train/đánh giá. Backend FE giữ model LBP; không đổi `FER_MODEL_PATH` sang HOG vì backend mặc định dùng đặc trưng LBP và sẽ từ chối model không khớp.

## 6. Huấn luyện trên Google Colab

Để máy local rảnh làm việc khác, dùng [notebook Colab](notebooks/01_train_fer_colab.ipynb). SVM hiện tại chạy CPU, chọn GPU không tự tăng tốc và Colab không bảo đảm nhanh hơn máy local.

1. Tạo gói code + ảnh train bằng lệnh bên dưới. File ZIP nằm trong thư mục tạm được Git bỏ qua, không chứa ảnh test, model hoặc thông tin đăng nhập.
2. Mở [Google Colab](https://colab.research.google.com/), upload notebook và chạy các ô theo thứ tự. Ở ô đầu, chọn file ZIP vừa tạo.
3. Cho phép notebook kết nối Drive. Model hoàn tất và log được lưu trong thư mục lần chạy riêng dưới `MyDrive/FER_Lab`.
4. Tải model từ ô cuối, đặt vào thư mục model local với tên `fer_colab.joblib`, rồi dùng biến `FER_MODEL_PATH` như bên dưới.

```powershell
.venv\Scripts\python -m src.training.package_colab
# Upload file .tmp/fer2013_colab.zip khi notebook yêu cầu
```

Gói chốt phiên bản NumPy, SciPy, scikit-learn, joblib, Pillow và OpenCV đang cài local. Notebook chạy được với kernel Colab Python 3.13: ô cài đặt dùng uv để tạo môi trường Python 3.12 riêng theo gói local, và mọi lệnh train/kiểm tra model đều chạy bằng môi trường đó. Không cần đổi kernel hoặc tự bỏ kiểm tra phiên bản. Cần mạng để tải Python và các thư viện; nếu cài thất bại thì dừng trước khi train. Môi trường thực tế được ghi vào `training_environment.json` cùng model trên Drive.

Notebook tự hỏi upload khi chưa thấy ZIP, dùng lại thư mục đã giải nén hợp lệ nếu chạy lại ô đầu. File ZIP cũ vẫn dùng được với notebook mới, không cần đóng gói lại. Chạy lại đóng gói với `--output .tmp/fer2013_colab_v2.zip` nếu muốn giữ gói cũ. [Quản lý Python bằng uv](https://docs.astral.sh/uv/guides/install-python/).

```powershell
$env:FER_MODEL_PATH = (Resolve-Path models/fer_colab.joblib).Path
.venv\Scripts\python -m uvicorn src.api.main:app --port 8000
```

Colab có giới hạn phiên và có thể ngắt kết nối. Lưu Drive giữ được model **đã train và ghi file xong**, không tạo khả năng resume giữa chừng cho SVM. Nếu fit bị dừng trước khi hoàn tất thì vẫn cần train lại. [Colab FAQ](https://research.google.com/colaboratory/faq.html), [tương thích model scikit-learn](https://scikit-learn.org/stable/model_persistence.html).

## 7. Cấu trúc và kiểm thử

- [API](src/api/main.py), [pipeline dùng chung](src/pipeline.py), [tiền xử lý](src/preprocessing/), [LBP](src/feature_extraction/lbp_descriptor.py), [SVM](src/emotion_recognition/svm_fer_classifier.py) và [huấn luyện](src/training/train.py).
- [Frontend](src/api/static/index.html) được phục vụ tại `/`, tài nguyên tĩnh tại `/static/`. [Kiểm thử JavaScript](tests/frontend.test.cjs) dùng Node có sẵn, chạy `node --test tests/frontend.test.cjs`; đây là kiểm thử logic với DOM giả lập, chưa thay thế kiểm thử trực quan trên trình duyệt.
- [Tests](tests/): kiểm tra phép biến đổi, 59 bins LBP, lỗi đầu vào, số mặt, giới hạn upload, trạng thái thiếu model, lưu/nạp SVM và train → API bằng dữ liệu tổng hợp.
- [Notebook Colab](notebooks/01_train_fer_colab.ipynb) gọi CLI huấn luyện; [experiments](experiments/) vẫn là khung, E1–E4 chưa triển khai.

```powershell
.venv\Scripts\python -m pytest tests/ -q
.venv\Scripts\python -m ruff check src tests
.venv\Scripts\python -m pip check
```

## 8. Trạng thái hiện tại

Đã có backend, giao diện thử ảnh, pipeline cổ điển và hỗ trợ chuyển huấn luyện sang Colab. FER2013 đã được người dùng đưa vào dữ liệu thô; chưa có kết quả đánh giá model thực tế. Tests dùng đặc trưng/ảnh tổng hợp và giả lập vùng mặt ở luồng thành công; không dùng chúng làm bằng chứng về độ chính xác FER. Chạy ứng dụng cục bộ để phát triển; chưa triển khai dịch vụ công khai. Nhãn chỉ mô tả biểu cảm quan sát được.
