# KẾ HOẠCH CHI TIẾT THỰC HIỆN TASK 12
## Chuẩn hóa Môi trường Mô phỏng cho Dữ liệu M5 và Hợp nhất Code giữa Baseline và RL lên GitHub

---

### THÔNG TIN CHUNG
* **Mã công việc:** Task 12 (Bảng theo dõi NCKH Lab 2026–2027)
* **Lĩnh vực:** Thực nghiệm và mô hình hóa
* **Người phụ trách (Lead):** Linh Chi
* **Thành viên thực hiện:** Nam Hải, Yến, Linh Chi, Hằng
* **Hạn hoàn thành (Deadline):** 2026-11-16
* **Đầu ra bắt buộc (Deliverables):**
  1. Mã nguồn trên GitHub được cập nhật và hợp nhất: Môi trường mô phỏng chuẩn dùng chung cho cả Baseline và RL.
  2. Bảng kết quả so sánh đối chuẩn (Benchmark Table) giữa RL và các Baseline trong cùng môi trường và cùng tập dữ liệu M5.
  3. Báo cáo cơ sở học thuật và nguồn số liệu cho bộ tham số chi phí và vận hành ($v_i, c_i, h_i, b_i, K, L, R$), phân biệt số liệu nguồn, giá trị suy ra và giả định mô phỏng.

---

## CẤU TRÚC PHÂN CHIA CÔNG VIỆC (WORK BREAKDOWN STRUCTURE - WBS)

```
Task 12: Chuẩn hóa Mô phỏng M5 & Hợp nhất Baseline - RL
│
├── GÓI 1: Thống nhất Bộ Thông số, Dữ liệu & Cơ sở Học thuật
│   ├── 1.1. Chốt phạm vi dữ liệu M5 & Phân vùng Train / Val / Test
│   ├── 1.2. Thống nhất bộ tham số chi phí & vận hành (v_i, c_i, h_i, b_i, K, L, R)
│   └── 1.3. Văn bản hóa cơ sở trích dẫn học thuật (Literature Citations)
│
├── GÓI 2: Chuẩn hóa Môi trường Mô phỏng Dùng chung (Gymnasium M5)
│   ├── 2.1. Xây dựng Gymnasium Environment nhận chuỗi dữ liệu M5 thực tế
│   ├── 2.2. Chuẩn hóa hàm tính toán chi phí và động học tồn kho (FIFO, Lost Sales)
│   └── 2.3. Viết Unit Test đối chiếu độ chính xác với hàm simulate() hiện tại
│
├── GÓI 3: Đóng gói Thuật toán Baseline và Huấn luyện RL (PPO)
│   ├── 3.1. Đóng gói các chính sách Baseline thành Policy Wrapper chuẩn
│   ├── 3.2. Cấu hình State Space & Huấn luyện PPO trên M5 Train Set
│   └── 3.3. Tinh chỉnh siêu tham số (Hyperparameter Tuning) trên M5 Validation Set
│
├── GÓI 4: Tự động hóa Đánh giá & Xuất Form So sánh Chuẩn
│   ├── 4.1. Xây dựng Pipeline đánh giá tự động (Automated Evaluator)
│   ├── 4.2. Đồng nhất công thức các chỉ số đánh giá (Total Cost, Fill Rate, CSL, GMROI)
│   └── 4.3. Xuất bảng dữ liệu đối chuẩn đa định dạng (DataFrame, CSV, Markdown, Excel)
│
└── GÓI 5: Hợp nhất Mã nguồn, Xây dựng Demo Notebook & Nghiệm thu
    ├── 5.1. Xây dựng Notebook hợp nhất m5_unified_benchmark.ipynb
    ├── 5.2. Soạn thảo Báo cáo Kỹ thuật Tổng kết Thực nghiệm
    └── 5.3. Review chéo, kiểm tra Git và bàn giao cho Lead nghiệm thu
```

---

## CHI TIẾT CÁC GÓI CÔNG VIỆC (DETAILED WORK PACKAGES)

### GÓI 1: THỐNG NHẤT BỘ THÔNG SỐ, DỮ LIỆU & CƠ SỞ HỌC THUẬT
*Mục tiêu: Đảm bảo cả hai nhánh Baseline và RL làm việc trên cùng một hệ quy chiếu dữ liệu và có cơ sở lý thuyết vững chắc.*

* **Công việc 1.1: Chốt phạm vi dữ liệu M5 và Phân vùng thời gian**
  * **Cửa hàng (Store):** Cố định `STORE_ID = "CA_1"` (Cửa hàng chuẩn của Walmart California).
  * **Tập SKU thử nghiệm:** Chọn 1 SKU tiêu biểu (chạy nhanh, kiểm thử thông luồng: ví dụ `FOODS_3_090_CA_1` hoặc SKU có nhu cầu liên tục) và mở rộng cho danh mục 5–20 SKU đại diện từ hàm `load_m5()`.
  * **Phân vùng thời gian (Time Splits):** Dùng `sales_train_evaluation.csv`, có doanh số đến `d_1941`. Giữ `VAL_DAYS = 56` và `TEST_DAYS = 112` theo notebook; đây là split của dự án, không phải split chính thức của cuộc thi.
    * *Tập Huấn luyện (Train):* `d_1` đến `d_1773` (29/01/2011–06/12/2015); mỗi SKU chỉ sử dụng lịch sử từ khi có giá bán hợp lệ.
    * *Tập Hiệu chỉnh (Validation):* `d_1774` đến `d_1829` (07/12/2015–31/01/2016), đủ 56 ngày — dùng để tìm ngưỡng tối ưu $(s, S)$ và tinh chỉnh mạng nơ-ron RL.
    * *Tập Đánh giá Độc lập (Test):* `d_1830` đến `d_1941` (01/02/2016–22/05/2016), đủ 112 ngày — tuyệt đối không dùng trong quá trình huấn luyện/chọn mô hình.
    * Không ghi `d_1914` đến `d_1941` là 112 ngày: khoảng này chỉ có 28 ngày. Ngày lịch được đối chiếu với `calendar.csv` trong dự án.
    * Nếu báo cáo kết quả theo các cửa sổ 28 ngày, vẫn giữ tồn kho và pipeline liên tục trong toàn bộ Test. Độ dài Test không đồng nghĩa với horizon dự báo; mọi dự báo chỉ được dùng thông tin có sẵn tại thời điểm ra quyết định.

* **Công việc 1.2: Thống nhất bộ tham số chi phí & vận hành**
  * **Phạm vi bộ tham số chốt:** Benchmark ban đầu tại `CA_1`, lost-sales, lead time cố định, chưa mô hình hóa hết hạn và điều chuyển đa kho. Bộ này là cấu hình mô phỏng có tham chiếu tài liệu, không phải bộ chi phí vận hành thực tế đã đo của Walmart. M5 cung cấp doanh số và giá bán; các tham số kinh tế còn thiếu phải được khai báo là proxy hoặc giả định.
  * **Ký hiệu:** $i$ là một cặp SKU–cửa hàng; $v_i$ là giá bán tham chiếu, $c_i$ là giá vốn proxy, $h_i$ là chi phí giữ hàng mỗi ngày, $b_i$ là tổn thất mỗi đơn vị mất bán, $K$ là phí mở một đơn mua. Nguồn [1]–[6] được ghi đầy đủ tại Công việc 1.3.

  | Tham số | Giá trị/công thức chốt | Đơn vị | Cơ sở và giới hạn |
  | :--- | :--- | :--- | :--- |
  | Bước thời gian | $\Delta t=1$ ngày | Ngày/bước | Phù hợp dữ liệu doanh số ngày của M5 [1]. |
  | Chu kỳ xét đặt hàng | $R=1$ ngày | Ngày | Lựa chọn thiết kế của dự án theo độ phân giải dữ liệu; không khẳng định Walmart đặt hàng hằng ngày. |
  | Giá bán tham chiếu | $v_i=\operatorname{median}(\text{sell\_price trên train của }i)$ | USD/đơn vị | Tính từ các quan sát giá tuần hợp lệ trong `sell_prices.csv` [1]; không lấy trung vị trên validation/test. Chỉ sử dụng giá được phép biết tại mốc train. |
  | Giá vốn proxy | $c_i=0.754v_i$ | USD/đơn vị | Suy ra từ gross profit margin hợp nhất Walmart FY2016 là 24.6% [2]. Áp dụng một tỷ lệ chung cho từng SKU là giả định, không phải giá nhập thực tế. |
  | Chi phí giữ hàng | $h_i=r_hc_i/365$, với $r_h=20\%$/năm | USD/đơn vị/ngày | WSU nêu carrying cost 20–35%/năm cho doanh nghiệp cung ứng nông nghiệp [3]. Chọn 20% là giả định mô phỏng dựa trên khoảng tham khảo, chưa phải ước lượng cho bán lẻ thực phẩm. |
  | Tổn thất lost-sales | $b_i=v_i-c_i=0.246v_i$ | USD/đơn vị mất bán | Dựa trên lợi nhuận biên bị bỏ lỡ trong mô hình newsvendor [4]. Không tính theo USD/đơn vị/ngày. Chưa cộng tổn thất thiện chí khách hàng. |
  | Phí mở đơn | Cấu hình A: $K=0$; cấu hình B: $K=55$ | USD/đơn mua | A là đối chứng loại bỏ phí mở đơn; B tham chiếu trung vị 55 USD/đơn của APQC, measure 103797 [5]. B là kịch bản tham khảo, không phải chi phí biên đã đo của cửa hàng. |
  | Lead time | $L=2$ ngày cố định | Ngày | Đặt đầu ngày $t$, nhận đầu ngày $t+2$. Mức 2 bằng kỳ vọng của phân phối LT1 trong Bashyam & Fu (1998), mục 5 [6]; quy đổi một kỳ thành một ngày là giả định của dự án. |

  * **Quy tắc sử dụng chi phí:**
    * Giữ cố định $v_i,c_i,h_i,b_i$ trong mỗi lần benchmark; dùng cùng giá tham chiếu để tính lợi nhuận gộp và GMROI. Không dùng tương lai để hiệu chỉnh giá hoặc chọn tham số.
    * Hệ số 24.6% từ báo cáo FY2016 là proxy cấu trúc cho nghiên cứu hồi cứu; không tuyên bố đây là thông tin đã có tại mọi mốc dự báo lịch sử. Nếu kiểm định theo thời điểm công bố thông tin, phải thay bằng báo cáo đã công bố trước mốc đó.
    * Khoảng carrying cost của WSU bao gồm cả rủi ro xuống cấp và lỗi thời. Trong mô hình này, quy ước $h_i$ là phí giữ hàng không chứa khoản ghi giảm giá vốn do hết hạn. Mức 20% sau khi tách khoản này vẫn là giả định cần kiểm tra độ nhạy; tài liệu không chứng minh riêng mức chi phí thuần đó.
    * Hàm mục tiêu dùng chi phí cơ hội $h_iI+b_iU+K\mathbb{I}(q>0)$. Không cộng thêm doanh thu và toàn bộ giá vốn vào cùng hàm rồi tiếp tục phạt phần lợi nhuận mất bán một lần nữa. Khi đổi sang mục tiêu lợi nhuận, phải đối chiếu lại cách hạch toán.
    * APQC đo chi phí bình quân của quy trình đặt mua, không xác nhận toàn bộ khoản đó là phí cố định tăng thêm mỗi đơn. Khi dùng $K=55$, công bố giả định quy đổi này; xác định năm giá và quy đổi trước khi gọi toàn bộ kết quả là chi phí lịch sử theo USD năm 2016.

  * **Định nghĩa đơn hàng và thiết kế đối chứng:**
    * **Cấu hình A — $K=0,L=2,R=1$:** Chạy trước để so sánh khả năng cân bằng tồn kho–lost-sales.
    * **Cấu hình B — $K=55,L=2,R=1$:** Chạy riêng để so sánh thêm quyết định gom lô và thời điểm mở đơn. Hiệu chỉnh chính sách trên validation riêng cho từng cấu hình; so sánh baseline và RL trong cùng cấu hình.
    * Với `M5SingleItemEnv`, mỗi hành động đặt lượng dương được định nghĩa là **một đơn riêng cho một SKU**. Khi tổng hợp các môi trường độc lập, phải ghi rõ giả định này; không diễn giải thành hệ thống mua hàng chung nhiều SKU.
    * Nếu mô hình hóa đơn chung nhiều SKU, phí chỉ tính một lần: $K\mathbb{I}(\sum_i q_{it}>0)$ cho mỗi nhóm SKU thực sự chung đơn. Cả baseline và RL phải dùng chung cách gom đơn; không vừa thu phí chung vừa thu thêm $K$ cho từng SKU.
    * Với bộ hệ số trên, $b_i/h_i\approx595.4$ ngày: mất bán một đơn vị tương đương chi phí giữ một đơn vị khoảng 595 ngày. Vì vậy phải báo cáo cả fill rate và tồn kho bình quân, không chỉ tổng chi phí. Không diễn giải tỷ lệ này thành CSL tối ưu.
    * Dùng chung chuỗi nhu cầu, thông tin quan sát, tồn kho/pipeline ban đầu, warm-up, giới hạn hành động, thời điểm quyết định và split dữ liệu. Doanh số M5 được dùng làm nhu cầu mô phỏng là một giả định; policy chỉ quan sát lượng bán đã phục vụ như notebook hiện tại, không được nhận nhu cầu bị mất nếu không có cơ chế quan sát đó.
    * **Phần mở rộng chưa chốt trong bộ này:** Shelf life, chi phí hàng hết hạn, lead time bất định, dung lượng và điều chuyển đa kho. Khi triển khai, cần bổ sung tham số có nguồn riêng và tránh tính trùng giá vốn hàng hủy trong $h_i$.

* **Công việc 1.3: Trích dẫn cơ sở học thuật (Literature Citations)**
  * **[1] M5 — dữ liệu và độ phân giải thời gian.** Makridakis, S., Spiliotis, E., & Assimakopoulos, V. (2022). *M5 accuracy competition: Results, findings, and conclusions*. International Journal of Forecasting, 38(4), 1346–1364. [DOI: 10.1016/j.ijforecast.2021.11.013](https://doi.org/10.1016/j.ijforecast.2021.11.013). Kèm [M5 Competitors' Guide, kho của ban tổ chức](https://github.com/Mcompetitions/M5-methods/blob/master/M5-Competitors-Guide.pdf), phần mô tả dữ liệu `sell_prices.csv` và doanh số ngày. Nguồn hỗ trợ đặc điểm dữ liệu; $R=1$ và split 56/112 ngày là thiết kế của dự án.
  * **[2] Walmart — biên lợi nhuận gộp.** Wal-Mart Stores, Inc. (2016). *2016 Annual Report*, Exhibit 13, **Five-Year Financial Summary**, dòng **Gross profit margin**, cột năm kết thúc ngày 31/01/2016: **24.6%**. [Bản công bố trên SEC](https://www.sec.gov/Archives/edgar/data/104169/000010416916000079/wmtars-1312016.htm). Công thức suy ra của dự án: $c_i=(1-0.246)v_i=0.754v_i$. Báo cáo không cung cấp tỷ lệ giá vốn riêng cho từng SKU M5.
  * **[3] Washington State University / USDA Cooperative Extension — carrying cost.** *Inventory Management and Control*, **trang 3**, phần chi phí giữ tồn kho: ước lượng **20–35% giá trị tồn kho bình quân mỗi năm** trong doanh nghiệp cung ứng nông nghiệp. [PDF của WSU](https://agribusiness-mgmt.wsu.edu/ExtensionNewsletters/performance/IventoryMgmtControl.pdf). Nguồn trình bày cả chi phí lưu trữ, xử lý và rủi ro tồn kho; mức 20% dùng trong dự án là giả định tham khảo, không phải số đo của Walmart hay một tỷ lệ thuần đã loại spoilage.
  * **[4] Cornell University — lợi nhuận biên bị mất.** *Newsvendor problem*, Computational Optimization Open Textbook, phần mô hình lợi nhuận biên. [Tài liệu Cornell](https://optimization.cbe.cornell.edu/index.php?title=Newsvendor_problem). Nguồn hỗ trợ cách tính lợi nhuận biên bằng giá bán trừ giá mua. Dự án áp dụng thành $b_i=v_i-c_i$ trong hàm chi phí cơ hội đa kỳ; không suy ra trực tiếp chính sách tối ưu đa kỳ từ công thức newsvendor một kỳ.
  * **[5] APQC — chi phí đặt mua.** *Total cost to perform the process "order materials and services" per purchase order*, **Measure ID 103797**, mục **Benchmark Data**: trung vị **55.00 USD/đơn mua**. [Trang chỉ tiêu APQC](https://www.apqc.org/resources/benchmarking/open-standards-benchmarking/measures/total-cost-perform-process-order-1). Trang tham khảo ngày **15/11/2026**; benchmark liên ngành, năm giá không được xác nhận trong bảng công khai. Nguồn hỗ trợ mức tham khảo 55 USD, không xác nhận chi phí biên của đơn bổ sung tại Walmart; $K=0$ là cấu hình đối chứng do dự án thiết kế.
  * **[6] Bashyam & Fu — cơ sở chọn mốc lead time.** Bashyam, S., & Fu, M. C. (1998). *Optimization of (s, S) Inventory Systems with Random Lead Times and a Service Level Constraint*. Management Science, 44(12-part-2), S243–S256. [DOI: 10.1287/mnsc.44.12.S243](https://doi.org/10.1287/mnsc.44.12.S243). **Mục 5, Numerical Experiments**, phân phối **LT1**: $P(L=1)=0.25$, $P(L=2)=0.50$, $P(L=3)=0.25$; suy ra $\mathbb{E}[L]=2$ kỳ. Bộ hiện tại chỉ dùng **$L=2$ ngày cố định**, chưa lấy mẫu LT1. Bài gốc xét backorder; việc mượn phân phối làm cơ sở thiết kế không chuyển kết luận tối ưu sang lost-sales.
  * **Quy tắc diễn giải:** Không dùng $b_i/(b_i+h_i)$ để khẳng định CSL tối ưu. Newsvendor dùng chi phí thiếu $C_u$ và chi phí thừa $C_o$ trên cùng cơ sở thời gian; $h_i$ mỗi ngày không tự động là $C_o$. Tham khảo bổ sung: [MIT OpenCourseWare — Newsvendor Inventory Problem](https://ocw.mit.edu/courses/15-772j-d-lab-supply-chains-fall-2014/6952be57b43aa185119c6f114908bcc5_MIT15_772JF14_Newsboy.pdf).
  * **Truy xuất nguồn:** Cấu hình và báo cáo lưu `source`, `source_year`, `unit`, `scope`, `assumption_type` cho từng tham số. Dùng `assumption_type` để phân biệt `data` (từ M5), `derived` (suy ra bằng công thức), `proxy` (mượn số liệu khác phạm vi) và `design` (thiết kế thí nghiệm); ghi rõ năm nguồn và năm giá khi chúng khác nhau.

---

### GÓI 2: CHUẨN HÓA MÔI TRƯỜNG MÔ PHỎNG DÙNG CHUNG (GYMNASIUM M5)
*Mục tiêu: Đưa toàn bộ mô phỏng về chuẩn Gymnasium v0.26+, hỗ trợ cắm trực tiếp dữ liệu M5 thực tế và đảm bảo độ chính xác tuyệt đối về toán học.*

* **Công việc 2.1: Xây dựng module môi trường `M5SingleItemEnv`**
  * File triển khai: `src/rl_agent/envs/m5_inventory_env.py` (hoặc mở rộng `single_item_env.py`).
  * Kế thừa `gymnasium.Env`, nhận vào mảng `demand_series` và các tham số $v_i,c_i,h_i,b_i,K,L,R$ theo Gói 1. Nếu giữ tên đối số cũ trong code, ánh xạ `h = h_i`, `p = b_i`, `k = K`, `unit_cost = c_i`; không giữ các giá trị mặc định chi phí cũ khi chạy benchmark này.
  * Thiết lập Observation Space: Vector trạng thái gồm tồn kho khả dụng hiện tại và hàng đợi hàng đang về:
    $$S_t = [I_t, q_{t-1}, q_{t-2}, \dots, q_{t-L}]$$
  * Thiết lập Action Space: Rời rạc $a_t \in \{0, 10, 20, \dots, A_{\max}\}$ hoặc liên tục có ràng buộc không âm.

* **Công việc 2.2: Đồng nhất công thức tính toán chi phí và động học tồn kho**
  * Tại mỗi bước $t$, xét một SKU $i$ và thống nhất thứ tự với `simulate()` trong notebook:
    1. Nhận hàng đầu ngày từ đơn đặt tại $t-L$: $I_{it}^{\text{on-hand}} = I_{it} + q_{i,t-L}$.
    2. Policy quan sát tồn kho, pipeline còn lại và lịch sử bán đã phục vụ; quyết định đặt $q_{it}=a_{it}$ **trước khi biết nhu cầu ngày $t$**. Ghi lịch nhận đầu ngày $t+L$; đơn mới không phục vụ nhu cầu ngay ngày đặt.
    3. Nhu cầu $D_{it}$ phát sinh. Lượng bán phục vụ: $Y_{it}=\min(I_{it}^{\text{on-hand}},D_{it})$.
    4. Lost-sales: $U_{it}=\max(0,D_{it}-I_{it}^{\text{on-hand}})$. Không chuyển phần thiếu sang ngày sau và không phạt lại cùng một đơn vị mất bán.
    5. Tồn cuối ngày: $I_{i,t+1}=I_{it}^{\text{on-hand}}-Y_{it}\ge0$. Cập nhật lịch sử quan sát của policy bằng $Y_{it}$.
    6. Tính chi phí cho trường hợp mỗi SKU đặt một đơn riêng:
       $$\text{Cost}_t = \underbrace{h_i I_{i,t+1}}_{\text{Giữ hàng}} + \underbrace{b_i U_{it}}_{\text{Lợi nhuận mất bán}} + \underbrace{K\mathbb{I}(q_{it}>0)}_{\text{Phí mở đơn}}$$
       $$\text{Reward}_t = -\text{Cost}_t$$
       Với đơn chung nhiều SKU, thay thành phần phí mở đơn theo quy tắc ở Công việc 1.2. Giá trị tồn kho, pipeline cuối kỳ và quy tắc xử lý cuối kỳ phải được báo cáo và áp dụng giống nhau cho mọi policy.

* **Công việc 2.3: Viết Unit Test kiểm thử tính tương thích (Consistency Test)**
  * File test: `tests/test_m5_env_consistency.py`.
  * Thực hiện test so sánh song song: Cho cùng một chuỗi nhu cầu M5 và cùng một chuỗi quyết định đặt hàng chạy qua đồng thời hàm `simulate()` trong `m5_baseline.ipynb` và `M5SingleItemEnv.step()`.
  * Khẳng định `assert np.allclose(...)` giữa hai bên với sai số bằng 0 về các đại lượng: `end_inventory`, `holding_cost`, `lost_sales_cost`, `ordering_cost`, `total_cost`.

---

### GÓI 3: ĐÓNG GÓI THUẬT TOÁN BASELINE VÀ HUẤN LUYỆN RL (PPO)
*Mục tiêu: Đưa tất cả các thuật toán về cùng một giao diện điều khiển để chạy trên môi trường chung.*

* **Công việc 3.1: Đóng gói các chính sách Baseline thành Policy Object chuẩn**
  * Xây dựng các lớp Policy tương thích:
    * `ClassicalRSPolicy`: Quyết định lượng đặt $a_t = \max(0, S - \text{Inventory Position})$ theo chu kỳ $R$.
    * `ClassicalRSSPolicy`: Chỉ đặt hàng khi $\text{Inventory Position} \le s$, lượng đặt $a_t = S - \text{Inventory Position}$.
    * `ASLLightGBMPolicy`: Dự báo trực tiếp lượng đặt hàng bằng mô hình LightGBM tối ưu hàm mất mát bất đối xứng (Asymmetric Loss).
  * Tất cả các policy đều triển khai hàm chung `.act(observation) -> action`.

* **Công việc 3.2: Huấn luyện thuật toán Deep RL (PPO)**
  * Sử dụng thư viện `Stable-Baselines3`.
  * Khởi tạo `M5SingleItemEnv` với dữ liệu tập `Train` của SKU M5.
  * Huấn luyện mạng PPO (Policy Gradient với Actor-Critic) học cách cân bằng giữa tồn kho an toàn và gom lô giảm phí mở đơn.
  * Tích hợp `EvalCallback` theo dõi hàm thưởng và lưu trữ checkpoint mô hình tốt nhất (`best_model.zip`).

* **Công việc 3.3: Tinh chỉnh và chọn mô hình trên tập Validation**
  * Chạy đánh giá trên 56 ngày Validation.
  * Lựa chọn cặp $(s, S)$ tối ưu của baseline và checkpoint PPO có tổng chi phí nhỏ nhất (hoặc thỏa mãn điều kiện Fill Rate mục tiêu).

---

### GÓI 4: TỰ ĐỘNG HÓA ĐÁNH GIÁ & XUẤT FORM SO SÁNH CHUẨN
*Mục tiêu: Xây dựng đường ống tự động chạy 100% không thủ công, xuất báo cáo đối chuẩn đa định dạng.*

* **Công việc 4.1: Xây dựng Pipeline đánh giá tự động (Automated Evaluator)**
  * File triển khai: `src/evaluation/m5_benchmark_evaluator.py`.
  * Hàm thực thi: `run_benchmark(env, policies_dict, eval_days=112)`.
  * Vòng lặp kiểm thử tự động nạp môi trường ở tập Test, chạy từng policy qua 112 ngày độc lập, thu thập nhật ký từng bước (`step_log`) và tổng hợp kết quả.

* **Công việc 4.2: Đồng nhất công thức các chỉ số đánh giá hiệu quả**
  * **Tổng chi phí (Total Cost):** $\sum (\text{Holding} + \text{Lost Sales} + \text{Ordering})$.
  * **Tỷ lệ lấp đầy nhu cầu (Fill Rate):** $\frac{\sum \text{Sales}}{\sum \text{Demand}} \times 100\%$.
  * **Mức độ phục vụ chu kỳ (Cycle Service Level - CSL):** Với $R=1$ đã chốt, một chu kỳ xét đặt là một ngày: $\frac{\sum_t \mathbb{I}(U_{it}=0)}{T}\times100\%$. Nếu mở rộng $R>1$, tính tỷ lệ chu kỳ đầy đủ không có lost-sales, không dùng tỷ lệ ngày để thay thế.
  * **Mức tiết kiệm chi phí so với Base-Stock (Savings % vs (R,S)):**
    $$\text{Savings \%} = \frac{\text{Total Cost}_{(R,S)} - \text{Total Cost}_{\text{Policy}}}{\text{Total Cost}_{(R,S)}} \times 100\%$$
  * **Tỷ suất sinh lời gộp trên tồn kho (GMROI):** $\frac{\text{Tổng lợi nhuận gộp}}{\text{Giá trị tồn kho bình quân}}$, với lợi nhuận gộp $\sum_{i,t}(v_i-c_i)Y_{it}$ và giá trị tồn kho bình quân $\frac{1}{T}\sum_{i,t}c_i I_{i,t+1}$. Không trừ lại holding/lost-sales/ordering trong tử số; ghi rõ chưa quy đổi năm, mẫu số bằng 0 thì để `NaN`.
  * **Chỉ số bổ sung bắt buộc:** Tồn kho bình quân, số đơn mở, lượng mất bán, tồn kho và pipeline cuối kỳ. Xuất riêng kết quả cấu hình A/B và ghi phạm vi tính phí đơn hàng; không so sánh Savings giữa hai cấu hình chi phí khác nhau. Nếu chi phí baseline bằng 0, Savings để `NaN`.

* **Công việc 4.3: Xuất bảng kết quả chuẩn hóa**
  * Tự động xuất ra:
    * `reports/tables/m5_benchmark_results.csv`
    * Bảng Markdown trực quan trong Notebook.
    * Bảng Excel `m5_benchmark_summary.xlsx` gửi cho Thầy và Hội đồng.

---

### GÓI 5: HỢP NHẤT MÃ NGUỒN, XÂY DỰNG NOTEBOOK & NGHIỆM THU
*Mục tiêu: Đóng gói toàn bộ sản phẩm hoàn chỉnh lên nhánh chung của GitHub.*

* **Công việc 5.1: Xây dựng Notebook trực quan hóa tổng hợp**
  * File: `notebooks/benchmarks/m5_unified_benchmark.ipynb`.
  * Cấu trúc 5 phần logic:
    1. *Cấu hình & Tải dữ liệu M5 chuẩn hóa.*
    2. *Môi trường Gymnasium mô phỏng M5.*
    3. *Nạp các mô hình: (R,S), (R,s,S), ASL-LS, PPO.*
    4. *Chạy đối chuẩn tự động trên 112 ngày Test.*
    5. *Bảng so sánh tổng hợp & Đồ thị phân rã chi phí / đường tồn kho.*

* **Công việc 5.2: Soạn thảo Báo cáo Kỹ thuật Tổng kết**
  * File: `docs/meeting_notes/report_task12_m5_unified_benchmark.md`.
  * Trình bày tóm tắt kết quả phân tích: Điểm mạnh/yếu của từng phương pháp khi đối mặt với dữ liệu thực tế M5 (tính gián đoạn của nhu cầu, hiệu quả tiết kiệm chi phí của PPO khi có phí $K$). Đính kèm bộ tham số và nguồn ở Gói 1; phân biệt benchmark proxy với chi phí vận hành đã đo, và không suy rộng kết quả ban đầu sang perishable/đa kho khi chưa mô phỏng các cơ chế đó.

* **Công việc 5.3: Rà soát kỹ thuật và Nghiệm thu nội bộ**
  * Chạy kiểm thử toàn bộ (`pytest tests/`).
  * Kiểm tra tính sạch sẽ của mã nguồn (clean git status, không commit checkpoint rác hoặc file dữ liệu thô quá nặng).
  * Bàn giao kết quả cho Linh Chi (Lead) nghiệm thu và đại diện nhóm báo cáo Thầy Việt Hà.

---

## MA TRẬN PHÂN CÔNG TRÁCH NHIỆM (RACI MATRIX)

| Thành viên | Gói 1: Thông số & Lý thuyết | Gói 2: Chuẩn hóa Môi trường | Gói 3: Đóng gói Baseline & RL | Gói 4: Tự động hóa Đánh giá | Gói 5: Hợp nhất & Báo cáo |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Linh Chi (Lead)** | **A / R** (Chốt thông số) | **C** (Rà soát logic) | **I** (Theo dõi tiến độ) | **A / R** (Duyệt form chuẩn) | **A / R** (Nghiệm thu & Báo cáo) |
| **Nam Hải (Tech/RL)**| **C** (Đề xuất $h_i,b_i,K,L$) | **A / R** (Code Gymnasium Env) | **A / R** (Huấn luyện PPO) | **R** (Code Pipeline Eval) | **R** (Build Notebook chung) |
| **Hải Yến (Baseline)**| **R** (Trích xuất M5 data) | **C** (Khớp hàm `simulate`) | **R** (Đóng gói (R,S), ASL) | **C** (Khớp metrics) | **R** (Rà soát code Baseline) |
| **Hằng (Modeling)** | **C** (Tổng hợp trích dẫn) | **C** (Hỗ trợ Unit test) | **R** (Hỗ trợ train/tune PPO) | **R** (Xuất bảng Excel/MD) | **R** (Soạn thảo văn bản báo cáo)|

*(Ghi chú: **A** - Accountable / Chịu trách nhiệm chính; **R** - Responsible / Người trực tiếp làm; **C** - Consulted / Người tham vấn góp ý; **I** - Informed / Người nhận thông tin).*
