# BÀI TẬP LỚN HỌC MÁY (MACHINE LEARNING ASSIGNMENT)
**Trường Đại học Bách Khoa - ĐHQG-HCM**  
**Khoa Khoa học và Kỹ thuật Máy tính**  
**Chủ đề:** Học máy với dữ liệu dạng bảng (Tabular Data)  
**Tập dữ liệu:** HR Analytics: Job Change of Data Scientists (`aug_train.csv`)  

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MinhEiden/ML-Assignment-/blob/main/notebooks/01_eda_exploration.ipynb)
![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![Status](https://img.shields.io/badge/Status-EDA%20Completed-brightgreen.svg)

---

## 📌 1. Giới thiệu Bài toán & Tập dữ liệu

Bài toán đặt ra là dự đoán xem một ứng viên sau khi hoàn thành các khóa đào tạo của công ty có muốn chuyển việc (tìm kiếm công việc mới) hay không (`target`: `0` - Không đổi việc, `1` - Muốn đổi việc).

- **Số lượng mẫu:** 19,158 dòng
- **Số lượng đặc trưng:** 14 cột (1 cột ID, 1 cột Target, 12 biến đầu vào)
- **Đặc điểm dữ liệu:**
  - Có tỷ lệ dữ liệu khuyết ở 8/12 đặc trưng (lên đến 32% ở `company_type`), đáp ứng yêu cầu thực hành kỹ thuật **Imputation**.
  - Đa dạng kiểu dữ liệu: Biến số, biến phân loại danh nghĩa (Nominal), biến thứ bậc (Ordinal) và biến phân loại nhiều giá trị (High-cardinality), đáp ứng yêu cầu thực hành kỹ thuật **Encoding**.
  - Dữ liệu bị mất cân bằng lớp (Class Imbalance: ~24.9% nhãn 1 vs ~75.1% nhãn 0).

---

## 📁 2. Cấu trúc Dự án (Project Structure)

Dự án được tổ chức theo chuẩn thực hành kỹ thuật học máy:

```text
ML-Assignment-/
├── data/
│   └── aug_train.csv             # Tập dữ liệu nguồn (19,158 dòng x 14 cột)
├── modules/
│   ├── __init__.py               # Package init & xuất khẩu hàm tiện ích
│   └── eda_viz.py                # Module trực quan hóa EDA theo từng nhóm đặc trưng
├── notebooks/
│   └── 01_eda_exploration.ipynb  # Google Colab / Jupyter Notebook phân tích EDA toàn diện
├── reports/
│   └── figures/                  # 17 biểu đồ phân tích xuất bản ở độ phân giải 300 DPI
│       ├── group1_target_distribution.png
│       ├── group2_num_city_development_index.png
│       ├── group2_num_training_hours.png
│       ├── group3_nominal_company_type.png
│       ├── group3_nominal_enrolled_university.png
│       ├── group3_nominal_gender.png
│       ├── group3_nominal_major_discipline.png
│       ├── group3_nominal_relevent_experience.png
│       ├── group4_ordinal_company_size.png
│       ├── group4_ordinal_education_level.png
│       ├── group4_ordinal_experience.png
│       ├── group4_ordinal_last_new_job.png
│       ├── group5_high_cardinality_city_top15.png
│       ├── group6_correlation_matrix.png
│       ├── group6_missing_values_bar.png
│       ├── group6_missingness_vs_target.png
│       └── group7_pca_scree_plot.png
├── .gitignore                    # Bỏ qua virtualenv, bytecode, file rác macOS/Windows
├── README.md                     # Tài liệu hướng dẫn & báo cáo tổng quan EDA
└── requirements.txt              # Danh sách thư viện phụ thuộc
```

---

## 📊 3. Báo cáo & Phát hiện Chính từ EDA (Exploratory Data Analysis)

Quá trình EDA được chia theo 7 nhóm đặc trưng theo đề cương khoa học:

| Nhóm | Đặc trưng | Kỹ thuật Trực quan | Phát hiện cốt lõi & Đóng góp kỹ thuật |
| :--- | :--- | :--- | :--- |
| **Nhóm 1: Biến mục tiêu** | `target` | Bar Chart + Donut Chart | Tỷ lệ `target=1` là **24.93%** (~1:3). Cần lưu ý đánh giá bằng chỉ số **PR-AUC, ROC-AUC, F1-score** thay vì chỉ dùng Accuracy. |
| **Nhóm 2: Biến số / Liên tục** | `city_development_index`, `training_hours` | Histogram (KDE) + Boxplot theo Target | `city_development_index` có tính phân tách nhãn cực mạnh: ứng viên ở thành phố chỉ số phát triển thấp có xu hướng đổi việc cao vượt trội. `training_hours` bị lệch phải (right-skewed). |
| **Nhóm 3: Biến danh nghĩa ít giá trị** | `gender`, `company_type`, `major_discipline`, `relevent_experience`, `enrolled_university` | Bar Chart (Số lượng) + Bar Chart (Tỷ lệ Target=1) | Ứng viên không có kinh nghiệm liên quan hoặc đăng ký học đại học chính quy (full-time) có tỷ lệ muốn đổi việc cao (~34-38%). |
| **Nhóm 4: Biến thứ bậc (Ordinal)** | `education_level`, `company_size`, `experience`, `last_new_job` | Ordered Bar Chart + Line Chart xu hướng | Biến `experience` (23 mốc) thể hiện xu hướng giảm đơn điệu (monotonic downward trend) rất rõ nét: kinh nghiệm càng cao thì tỷ lệ muốn đổi việc càng giảm (từ >35% xuống <15%). |
| **Nhóm 5: Biến danh nghĩa nhiều giá trị** | `city` (123 thành phố) | Top 15 Horizontal Bar + Bảng kiểm chứng 1-1 | **Phát hiện quan trọng:** Kiểm chứng thực nghiệm chứng minh 100% (123/123) thành phố có quan hệ 1-1 với `city_development_index`. Quyết định: **Loại bỏ cột `city`** để tránh tạo 123 cột one-hot gây nguyền rủa số chiều (curse of dimensionality). |
| **Nhóm 6: Dữ liệu khuyết & Tương quan** | Tất cả các cột có khuyết | Bar Chart tỷ lệ khuyết + So sánh MNAR + Spearman Heatmap (7x7) | **Chứng minh cơ chế MNAR:** Ứng viên khuyết `company_size` và `company_type` có tỷ lệ đổi việc lên tới **40.6%** và **38.3%** (gấp 2-3 lần nhóm có thông tin). Do đó **không được xóa dòng**, cần điền nhãn riêng `'Unknown'` kèm cờ `MissingIndicator`. |
| **Nhóm 7: Khảo sát PCA** | Toàn bộ không gian đặc trưng | Scree Plot & Đường phương sai tích lũy | Sau One-Hot Encoding có **62 chiều**. Để giữ **90% phương sai**, cần chọn $k = 47$ components. Để giữ **95% phương sai**, cần chọn $k = 51$ components. Đáp ứng trực tiếp yêu cầu mục 4.1 trong Đề bài. |

---

## 🛠️ 4. Đề xuất Chiến lược Tiền Xử Lý & Điền Khuyết (Imputation Strategy)

Dựa trên các bằng chứng thực nghiệm từ EDA, chiến lược tiền xử lý cho Pipeline được xây dựng như sau:

| Đặc trưng | % Khuyết | Cơ chế Khuyết | Chiến lược xử lý đề xuất cho Pipeline | Cơ sở khoa học |
| :--- | :--- | :--- | :--- | :--- |
| `company_size` | 30.82% | **MNAR** | Thêm category `'Unknown'` + `MissingIndicator` | Tỷ lệ Target ở nhóm khuyết lên tới 40.6% (rất cao). |
| `company_type` | 32.05% | **MNAR** | Thêm category `'Unknown'` + `MissingIndicator` | Tỷ lệ Target ở nhóm khuyết là 38.3%. |
| `gender` | 23.53% | **MNAR / MAR** | Thêm category `'Unknown'` / `'Other'` | Tỷ lệ Target ở nhóm khuyết đạt 30.5%. |
| `major_discipline` | 14.68% | **MAR** | Điền category `'No Major'` | Chủ yếu khuyết do học vấn High School / Primary School. |
| `education_level` | 2.40% | **MAR / MCAR** | Điền Mode (`'Graduate'`) hoặc Ordinal Imputer | Tỷ lệ khuyết nhỏ (2.4%), không làm lệch phân phối. |
| `last_new_job` | 2.21% | **MAR** | Điền Mode (`'1'`) | Tỷ lệ khuyết thấp, đa số ứng viên ở mốc 1 năm. |
| `enrolled_university` | 2.01% | **MAR** | Điền Mode (`'no_enrollment'`) | Tỷ lệ `no_enrollment` chiếm đa số áp đảo (>72%). |
| `experience` | 0.34% | **MCAR** | Điền Median (~9 năm) hoặc Mode (`'>20'`) | Khuyết ngẫu nhiên hoàn toàn và tỷ lệ rất nhỏ. |
| `city` | 0.00% | *Không khuyết* | **Loại bỏ (Drop)** | Đã mã hóa toàn vẹn vào `city_development_index`. |
| `enrollee_id` | 0.00% | *Không khuyết* | **Loại bỏ (Drop)** | Cột định danh, không chứa tín hiệu dự đoán. |

---

## 🚀 5. Hướng dẫn Chạy (Quick Start)

### 🔹 Cách 1: Chạy trực tiếp trên Google Colab (Khuyến nghị)
Nhấp chuột vào nút sau để mở trực tiếp notebook trên Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MinhEiden/ML-Assignment-/blob/main/notebooks/01_eda_exploration.ipynb)

> Notebook đã được tích hợp cơ chế **tự động clone repository, liên kết thư mục `modules/` và đọc dữ liệu `data/`** mà không yêu cầu mount Google Drive hay can thiệp thủ công. Chọn **Runtime -> Run all** để chạy toàn bộ từ đầu đến cuối thành công 100%.

### 🔹 Cách 2: Chạy trên môi trường Local

1. **Clone repository:**
   ```bash
   git clone https://github.com/MinhEiden/ML-Assignment-.git
   cd ML-Assignment-
   ```

2. **Cài đặt thư viện:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Khởi chạy Jupyter Notebook:**
   ```bash
   jupyter notebook notebooks/01_eda_exploration.ipynb
   ```
   Hoặc mở file notebook trong Visual Studio Code và chọn Python Kernel mong muốn.

---

## 📜 6. Đóng góp & Bản quyền
Dự án được thực hiện phục vụ Bài tập lớn môn **Học máy (Machine Learning)** - Khoa KH&KT Máy tính, Trường Đại học Bách Khoa, ĐHQG-HCM.
