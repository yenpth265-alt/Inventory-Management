# Nghiên cứu 1: End-to-End Optimization (Tối ưu toàn cục)

## Câu hỏi nghiên cứu
Làm thế nào để tách biệt hoàn toàn Module Dự báo (Forecasting) và Module Tối ưu hóa
Tồn kho (Inventory Optimization) mà vẫn đảm bảo cả hệ thống hoạt động hiệu quả
(không bị error propagation như bài báo cũ)?

---

## Vấn đề với Pipeline cũ (v1)

Pipeline cũ bị ghép nối cứng nhắc (hard-coupled):
```
Forecast(y_hat) → SS = z·σ_err·√R → S = y_hat + SS → Knapsack(S)
```

Khi Knapsack cắt giảm x_i < S_i, toàn bộ Safety Stock tính ở bước trước bị phá vỡ.
Mô hình tối ưu hóa không biết gì về sai số dự báo, và mô hình dự báo
không biết gì về ràng buộc ngân sách/kho.

---

## Các hướng tiếp cận E2E cần nghiên cứu

### Hướng 1: Predict-then-Optimize (P&O) truyền thống (nhưng làm đúng)
- **Ý tưởng:** Tách hẳn 2 module. Module Forecast xuất ra PHÂN PHỐI XÁC SUẤT
  (không chỉ 1 số), Module Optimization nhận phân phối đó làm tham số đầu vào.
- **Điểm mấu chốt:** Optimization phải nhận toàn bộ dải phân vị (q10, q50, q90, q99),
  không chỉ nhận giá trị kỳ vọng. Nhờ đó SS được tính TRONG bài toán tối ưu,
  không phải ở bước dự báo.
- **Bài báo tham khảo:**
  - Bertsimas & Kallus (2020) - "From Predictive to Prescriptive Analytics"
  - Huyen Thi Pham et al. (2022) - "Data-driven inventory management"

### Hướng 2: Smart Predict-then-Optimize / Decision-Focused Learning (DFL)
- **Ý tưởng:** Thay vì tối thiểu hóa sai số dự báo (RMSE), hàm mất mát
  của mô hình dự báo được thiết kế để tối thiểu hóa TRỰC TIẾP tổng chi phí tồn kho
  (Task Loss / Decision Loss).
- **Điểm đột phá:** Mô hình học để dự báo "đúng cái cần thiết cho quyết định",
  không phải dự báo "chính xác tuyệt đối về mặt thống kê".
- **Bài báo gốc:**
  - Elmachtoub & Grigas (2022) - "Smart 'Predict, then Optimize'" (Management Science)
  - Mandi et al. (2023) - "Decision-Focused Learning: Foundations, State of the Art, Benchmark"

### Hướng 3: Data-Driven Newsvendor / Sample Average Approximation (SAA)
- **Ý tưởng:** Bỏ hoàn toàn bước dự báo trung gian. Bài toán tối ưu tồn kho
  được giải TRỰC TIẾP từ dữ liệu lịch sử quan sát, không cần giả định phân phối.
- **Đặc biệt phù hợp** với dữ liệu bán lẻ có phân phối lệch, nhiều số 0.
- **Bài báo tham khảo:**
  - Levi et al. (2015) - "Data-Driven Newsvendor Problems: Review"
  - Ban & Rudin (2019) - "The Big Data Newsvendor: Practical Insights from Machine Learning"

---

## Kết cấu thực nghiệm đề xuất

```
Module A (Forecasting - Độc lập hoàn toàn)
├── Input: Lịch sử bán hàng, features (lags, rolling, calendar)
├── Model: LightGBM Quantile / TFT / DeepAR
└── Output: Phân phối dự báo P̂(D_{t+1:t+R+L} | history) dạng (q10, q50, q90, q99)

Module B (Inventory Optimization - Nhận phân phối từ A, KHÔNG phụ thuộc cách A tính)
├── Input: Phân phối từ Module A + Ràng buộc (B, V_max, L, K, k_i, C_i, h_i, π_i)
├── Model: Stochastic MILP hoặc SAA-based solver
└── Output: Chính sách đặt hàng tối ưu x_i* tối thiểu chi phí kỳ vọng

Vòng lặp kiểm định (Sau tối ưu):
└── Rolling Simulation (28 ngày, Lost Sales, L > 0) → đánh giá lại Fill Rate sau cắt giảm
```

---

## Tài liệu cần đọc (Ưu tiên)

1. [ ] Elmachtoub & Grigas (2022) - SPO+ paper (Management Science)
2. [ ] Bertsimas & Kallus (2020) - Predictive to Prescriptive Analytics
3. [ ] Mandi et al. (2023) - Decision-Focused Learning Survey
4. [ ] Ban & Rudin (2019) - Big Data Newsvendor
5. [ ] Puterman (1994) - Markov Decision Processes (Nền tảng lý thuyết)
