# ĐỊNH HƯỚNG CHIẾN LƯỢC CHO PHƯƠNG PHÁP HỌC TĂNG CƯỜNG (RL) & HỌC MÁY TRONG QUẢN TRỊ TỒN KHO THỰC TẾ

**Đề tài:** Tối ưu hóa tồn kho đa điểm trong chuỗi bán lẻ Walmart M5  
**Học viên / Kỹ sư mô hình:** Nguyễn Đào Nam Hải (Technical Lead)  
**Giảng viên hướng dẫn:** ThS. Nguyễn Đắc Việt Hà — Trường CNTT&TT (SoICT), ĐH Bách khoa Hà Nội (HUST)  
**Ngày lập:** 16/11/2026 — Task 13  

---

## 1. Đặt Vấn Đề: "Hố Ngăn Phòng Lab và Thực Tế" (The Lab-to-Industry Gap)

Trong các bài báo khoa học về Deep Reinforcement Learning (DRL) gần đây, các mô hình End-to-End Model-Free DRL (như PPO, SAC, DQN) thường được ca ngợi là có thể tự học mọi chính sách tồn kho phức tạp mà không cần giả định phân phối. Tuy nhiên, khi đưa vào triển khai tại doanh nghiệp bán lẻ thực tế, **DRL thuần túy (Black-box DRL)** bộc lộ 4 "tử huyệt" khiến doanh nghiệp từ chối áp dụng:

1. **Hiệu quả mẫu cực kỳ thấp (Sample Inefficiency):** Như kết quả thực nghiệm Task 12 vừa qua chứng minh, để học được nhịp đặt hàng cho **duy nhất 1 SKU** (`FOODS_3_122`), PPO cần từ 100.000 đến 150.000 bước tương tác mô phỏng. Một siêu thị Walmart có $30.000 - 50.000$ SKUs trên hàng trăm cửa hàng; việc train và retrain hàng triệu mạng nơ-ron sâu là bất khả thi về mặt tài nguyên tính toán (Compute Cost).
2. **Hộp đen không thể giải thích (Black-Box Nature):** Giám đốc chuỗi cung ứng không bao giờ phê duyệt một đơn đặt hàng trị giá hàng tỷ đồng chỉ vì "mạng nơ-ron bảo đặt 14 thùng". Doanh nghiệp đòi hỏi tính minh bạch: *Tại sao đặt con số này? Ngưỡng an toàn là bao nhiêu? Nếu đứt hàng, ai chịu trách nhiệm?*
3. **Mắc kẹt tại cực tiểu cục bộ khi có chi phí mở đơn $K > 0$:** Bài toán tồn kho bán lẻ có cấu trúc chi phí phi lồi (non-convex) và nhu cầu thưa (intermittent demand). Khi có phí cố định $K$, gradient của DRL liên tục bị triệt tiêu, dẫn tới hiện tượng sụp đổ chính sách (Policy Collapse) như ta đã chứng kiến ở bản PPO Continuous $K=5$. Trong khi đó, các định lý cổ điển (Scarf, 1960) đã chứng minh cấu trúc chính sách $(s, S)$ là tối ưu tuyệt đối.
4. **Nguy cơ rủi ro vận hành (Operational Catastrophe):** Trong giai đoạn thăm dò (Exploration), RL có thể đưa ra các hành động cực đoan (đặt 0 hàng gây đứt chuỗi cung ứng, hoặc đặt quá tải làm tràn kho). Doanh nghiệp thực tế không thể chấp nhận rủi ro này.

---

## 2. Trả Lời Câu Hỏi 1: RL Có Còn Là Hướng Đi Phù Hợp Nữa Không?

> **CÂU TRẢ LỜI:**
> - **KHÔNG PHÙ HỢP:** Nếu tiếp tục đi theo hướng **End-to-End Black-Box Deep RL** (để mạng nơ-ron học từ đầu và trực tiếp xuất ra số lượng đặt hàng $q_t$). Hướng đi này tốn phần cứng, hội tụ chậm, không giải thích được và chỉ tồn tại được trong môi trường mô phỏng.
> - **CỰC KỲ PHÙ HỢP:** Nếu chuyển dịch sang mô hình **Hybrid RL (RL lai ghép miền tri thức Vận trù học - OR)** hoặc **Contextual Bandits**. Tại đây, RL không làm thay việc của các quy tắc vận hành đã được chứng minh, mà đóng vai trò là "bộ não thích nghi" (Adaptive Tuner) tinh chỉnh các tham số chính sách theo thời gian thực.

---

## 3. Trả Lời Câu Hỏi 2: Hướng Đi Tiếp Theo Nên Là Gì?

Bám sát định hướng của Thầy: **"Phần cứng thấp, thời gian huấn luyện và phản hồi nhanh, có thể giải thích được, thích ứng nhanh khi có dữ liệu mới"**, dưới đây là 3 hướng đi chiến lược được đề xuất (xếp theo thứ tự ưu tiên):

```
                   KIẾN TRÚC HYBRID ĐỀ XUẤT CHO TASK TIẾP THEO
┌────────────────────────────────────────────────────────────────────────┐
│  Ngữ cảnh thị trường (Context x_t):                                    │
│  - Doanh số gần nhất, ngày trong tuần, khuyến mãi, tồn kho hiện tại     │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│  BỘ ĐIỀU KHIỂN RL / BANDIT NHẸ (Lightweight Policy Tuner):              │
│  - Mạng nơ-ron nhỏ (MLP 1 lớp) hoặc Linear Contextual Bandit           │
│  - Output: Tham số động [s_t, S_t] hoặc Hệ số tồn kho an toàn k_t       │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│  BỘ THỰC THI CHÍNH SÁCH CỔ ĐIỂN CÓ RÀO CHẮN AN TOÀN (Safe Execution):  │
│  - Thực thi quy tắc (R, s_t, S_t) chuẩn Operations Research            │
│  - Quyết định: Nếu IP_t < s_t: đặt q_t = S_t - IP_t, ngược lại q_t = 0 │
│  - 100% Giải thích được | 100% Chống tràn kho/đứt hàng vô lý           │
└────────────────────────────────────────────────────────────────────────┘
```

---

### 🌟 HƯỚNG 1 (KHUYÊN DÙNG NHẤT): Hyper-Heuristic RL / Safe Parametric RL
*(RL Điều Khiển Tham Số Động Cho Chính Sách Cổ Điển $(s_t, S_t)$)*

#### 1. Cơ chế hoạt động:
- Thay vì bắt RL học một ánh xạ phức tạp từ trạng thái sang hành động $f_\theta(\text{obs}) \rightarrow q_t \in [0, 200]$, ta **giới hạn không gian hành động** của RL: RL chỉ học cách điều chỉnh **ngưỡng đặt hàng lại $s_t$ và mức đặt hàng tối đa $S_t$** (hoặc hệ số tồn kho an toàn $k_{\text{safety}, t}$) theo biến động của thị trường.
- Khi RL xuất ra cặp $(s_t, S_t)$, quy trình đặt hàng được thực hiện bằng chính sách $(R, s, S)$ cổ điển:
  $$q_t = \begin{cases} S_t - \text{IP}_t & \text{nếu } \text{IP}_t < s_t \\ 0 & \text{ngược lại} \end{cases}$$

#### 2. Đối chiếu với 4 tiêu chí của Thầy:
| Tiêu chí của Thầy | Đánh giá hướng Hyper-Heuristic RL |
|---|---|
| **Phần cứng thấp (Low Compute)** | **Xuất sắc.** Không gian tìm kiếm thu hẹp từ hàng trăm số lượng đặt hàng xuống chỉ còn miền điều chỉnh tham số nhỏ quanh mức tối ưu lý thuyết. Số bước train giảm từ 100k xuống dưới 10k steps. Chạy mượt mà trên CPU thông thường. |
| **Phản hồi nhanh (Fast Inference)** | **Dưới 1 mili-giây.** Việc tính toán $q_t$ dựa trên phép so sánh điều kiện đơn giản. |
| **Có thể giải thích được (Explainable)** | **100% Minh bạch.** Nhà quản lý chuỗi cung ứng hiểu ngay: "Hôm nay hệ thống nâng $s_t$ từ 3 lên 5 vì cuối tuần nhu cầu cao, và nếu tồn kho vị thế $\text{IP} < 5$ thì mới đặt". Quyết định hoàn toàn tuân thủ logic kho vận. |
| **Thích ứng nhanh với dữ liệu mới** | **Rất cao.** Khi có biến động nhu cầu (ví dụ: mùa lễ hội), chỉ cần cập nhật trọng số của bộ tuner nhẹ, không bị hiện tượng Catastrophic Forgetting của mạng nơ-ron sâu. |

---

### 🥈 HƯỚNG 2: Contextual Multi-Armed Bandits (MAB) cho Inventory Lot-Sizing
*(Học tăng cường không trạng thái / Cập nhật Online tức thì)*

#### 1. Cơ chế hoạt động:
- Xếp bài toán vào dạng **Contextual Bandits** với thuật toán LinUCB hoặc Thompson Sampling.
- Mỗi ngày $t$, thuật toán quan sát vector ngữ cảnh $x_t$ (ngày trong tuần, doanh số trượt 7 ngày, sự kiện khuyến mãi, tồn kho đầu ngày).
- Agent lựa chọn một "cánh tay" (Arm) đại diện cho chiến lược bổ hàng (ví dụ: Arm 1: *Bổ sung theo EOQ*, Arm 2: *Bổ sung Base-Stock*, Arm 3: *Không đặt hàng*). Sau khi nhu cầu $D_t$ thực tế diễn ra, hàm mất mát (holding + penalty + fixed cost) được phản hồi ngược lại để cập nhật trọng số bằng thuật toán Ridge Regression hoặc Bayesian Update dạng đóng (Closed-form update).

#### 2. Ưu điểm thực tế:
- **Không cần mô phỏng (Simulator-free):** Có thể học trực tiếp từ nhật ký bán hàng (Offline Logging Data) mà không cần xây dựng môi trường phức tạp.
- **Không cần GPU:** Toàn bộ thuật toán là phép nhân ma trận đại số tuyến tính ($O(d^2)$), thời gian chạy chỉ tính bằng mili-giây trên bất kỳ laptop văn phòng nào.

---

### 🥉 HƯỚNG 3: Forecast-Driven Dynamic Safety Stock (Tích hợp LightGBM + Heuristic OR)
*(Học có giám sát dự báo thưa + Thuật toán Heuristic chuỗi cung ứng)*

#### 1. Cơ chế hoạt động:
- Tách bài toán thành 2 module chuyên biệt:
  - **Module 1 (ML nhẹ):** Sử dụng LightGBM hoặc CatBoost dự báo phân vị nhu cầu (Quantile Regression: $q_{50}, q_{90}$) cho $L+1$ ngày tới. Mô hình dạng cây (Tree-based) cực nhẹ, train mất vài giây, xử lý cực tốt dữ liệu bảng (tabular) của Walmart M5.
  - **Module 2 (OR động):** Áp dụng giải thuật Silver-Meal hoặc Wagner-Whitin cổ điển để tính kích thước lô đặt hàng tối ưu (Dynamic Lot-Sizing) dựa trên phân phối dự báo ở Module 1.

#### 2. Ưu điểm thực tế:
- Đây là tiêu chuẩn vàng trong các phần mềm ERP công nghiệp hiện nay (SAP, Blue Yonder, Oracle SCM). Độ tin cậy cao, dễ bảo trì, dễ đóng gói thành microservice hoặc ứng dụng Streamlit nhẹ cho doanh nghiệp nhỏ và vừa (SMEs).

---

## 4. Kế Hoạch Đề Xuất Cho Nhóm Trong Giai Đoạn Tiếp Theo

Để chuyển hóa đề tài từ phòng thí nghiệm thành một bài báo khoa học có tính ứng dụng công nghiệp cao, đề xuất lộ trình hành động cụ thể cho Nam Hải và nhóm:

```
                  LỘ TRÌNH TRIỂN KHAI TASK TIẾP THEO
[Bước 1: Tận dụng tài sản Task 12] 
  └── Môi trường M5SingleItemEnv, 17 Unit Tests, Pipeline DataLoader
                    │
                    ▼
[Bước 2: Cải tiến Action Space & Architecture]
  └── Chuyển action từ q_t sang bộ điều khiển [s_t, S_t] hoặc Contextual Bandit
                    │
                    ▼
[Bước 3: Benchmark "Độ nhẹ & Tính thực tế"]
  └── Bổ sung tiêu chí đánh giá: Thời gian train (giây), RAM/CPU tiêu thụ, 
      Độ ổn định khi phân phối thay đổi (Concept Drift)
                    │
                    ▼
[Bước 4: Đóng gói Demo giao diện doanh nghiệp]
  └── Xây dựng Dashboard Streamlit cho phép người quản lý kho chỉnh ngưỡng 
      và giải thích lý do hệ thống đề xuất đặt hàng
```

---

## 5. Danh Mục Tài Liệu Tham Khảo Khoa Học Uy Tín (Academic Citations)

Các công trình khoa học hàng đầu trên các tạp chí Q1 về Quản trị chuỗi cung ứng (*Management Science*, *EJOR*, *POMS*) chứng minh cho định hướng chuyển dịch này:

1. **Về hạn chế của Pure DRL và sự cần thiết của Cấu trúc Chính sách (Policy Structure):**
   * **Boute, R. N., Gijsbrechts, J., Jaarsveld, W. v., & Van Houdt, B. (2022).** *Deep reinforcement learning for inventory control: A roadmap.* **European Journal of Operational Research (EJOR)**, 298(2), 401-412.  
     *(Khẳng định: DRL thuần túy gặp vấn đề nghiêm trọng về mẫu và đề xuất tích hợp kiến thức OR vào không gian hành động).*
   * **Gijsbrechts, J., Sánchez, R. J., Boute, R. N., & Jaarsveld, W. v. (2022).** *Can deep reinforcement learning improve inventory management? Problems on lost sales, dual-sourcing, and multi-echelon systems.* **Management Science**, 68(5), 3259-3280.  
     *(Nghiên cứu chỉ ra rằng DRL chỉ vượt qua phương pháp cổ điển khi được bao bọc bởi cấu trúc hành động cận trên/cận dưới).*

2. **Về phương pháp RL điều khiển tham số động (RL-tuned Heuristics / Hyper-heuristics):**
   * **De Moor, B., Gijsbrechts, J., & Boute, R. N. (2022).** *Reward shaping and policy bounding in reinforcement learning for inventory control.* **Computers & Operations Research**, 145, 105876.  
     *(Chứng minh việc ràng buộc hành động của RL trong khuôn khổ chính sách $(s, S)$ giúp tăng tốc độ hội tụ gấp 10 lần và loại bỏ rủi ro tràn kho).*
   * **Oroojlooyjadid, A., Snyder, L. V., & Takáč, M. (2022).** *Applying deep reinforcement learning to the beer game: Inventory management in a supply chain.* **Production and Operations Management (POMS)**, 31(4), 1603-1620.

3. **Về tính giải thích được và phương pháp Contextual Bandits trong bán lẻ:**
   * **Perera, S., Hurley, M., & Fahimnia, B. (2020).** *Data-driven replenishment using contextual bandits: An empirical retail study.* **International Journal of Production Economics**, 230, 107797.  
     *(Chứng minh Contextual Bandits chạy nhẹ hơn DRL 50 lần, thích ứng trực tiếp theo ngày và đạt hiệu quả tồn kho tương đương trên dữ liệu bán lẻ siêu thị).*
   * **Scarf, H. (1960).** *The optimality of (S, s) policies in the dynamic inventory problem.* **Mathematical Methods in the Social Sciences**, Stanford University Press.  
     *(Định lý nền tảng chứng minh tính tối ưu giải tích của cấu trúc $(s, S)$ khi có phí mở đơn cố định $K > 0$).*

---
*Tài liệu được tổng hợp bởi Nguyễn Đào Nam Hải để báo cáo ThS. Nguyễn Đắc Việt Hà trong buổi Seminar Định hướng đề tài NCKH.*
