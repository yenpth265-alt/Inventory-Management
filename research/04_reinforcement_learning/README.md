# Nghiên cứu 4: Reinforcement Learning cho Inventory Optimization

## Câu hỏi nghiên cứu
RL hiện đang được áp dụng như thế nào trong bài toán quản trị tồn kho?
Nhóm có nên áp dụng RL không? Tính khả thi ra sao?

---

## Tại sao RL trở nên phổ biến trong Inventory Optimization?

### Hạn chế của các phương pháp tối ưu cổ điển
- MILP / LP: Cần phải biết chính xác phân phối nhu cầu trước → dễ sai khi nhu cầu
  thực tế phức tạp và bất định.
- (R, S) / EOQ: Giả định nhiều điều kiện cứng nhắc (phân phối chuẩn, lead time cố định,
  đơn sản phẩm độc lập).
- Cả hai đều là STATIC: Tính toán chính sách MỘT LẦN, không thích nghi theo thời gian.

### Điểm mạnh của RL
- RL học chính sách QUY TRÌNH ĐỘNG (Sequential Decision Making): Agent quan sát trạng
  thái tồn kho hiện tại, quyết định nhập bao nhiêu, nhận phần thưởng (reward = -cost),
  học dần qua hàng nghìn lần thử.
- Không cần giả định phân phối nhu cầu trước.
- Thích nghi linh hoạt khi môi trường thay đổi (mùa vụ, sự kiện đột xuất).

---

## Các thuật toán RL phù hợp nhất cho bài toán tồn kho

### 1. DQN (Deep Q-Network) — Hành động rời rạc
- **Phù hợp khi:** Lượng đặt hàng được rời rạc hóa (ví dụ: 0, 10, 20, 50, 100 đơn vị).
- **Hạn chế:** Không hiệu quả khi không gian hành động quá lớn (nhiều SKU).

### 2. PPO (Proximal Policy Optimization) — Hành động liên tục
- **Phù hợp nhất** cho bài toán tồn kho đa sản phẩm vì hành động là vector lượng đặt
  hàng liên tục cho từng SKU.
- Ổn định hơn DQN, phổ biến nhất trong các nghiên cứu gần đây.

### 3. SAC (Soft Actor-Critic) — Hành động liên tục, sample-efficient
- Hiệu quả hơn PPO khi dữ liệu (interaction) khan hiếm.
- Tự động cân bằng giữa exploration và exploitation.

---

## Đánh giá tính khả thi cho nhóm

| Tiêu chí | Đánh giá | Ghi chú |
|:---|:---:|:---|
| Dữ liệu huấn luyện | ⚠️ Trung bình | RL cần MÔI TRƯỜNG GIẢ LẬP (Simulation Env), không train trực tiếp trên M5 |
| Độ phức tạp code | ⚠️ Cao | Cần xây môi trường Gymnasium custom |
| Thời gian huấn luyện | ⚠️ Dài | PPO/SAC cần nhiều iteration (>10,000 episodes) |
| Tính giải thích được | ❌ Thấp | "Black box" — khó thuyết phục reviewer về tính hợp lý kinh tế |
| Tính đóng góp học thuật | ✅ Cao | Còn ít bài báo kết hợp RL + Multi-Item + Budget Constraint |
| Khuyến nghị | ⚠️ | Dùng RL như phương pháp SO SÁNH, không phải phương pháp chính |

---

## Đề xuất hướng tích hợp RL vào dự án

```
Phương pháp chính (Main Method):
├── Module A: Probabilistic Forecasting (LightGBM/CatBoost Quantile)
└── Module B: Stochastic MILP / SAA Optimization

Phương pháp so sánh (Comparison):
└── RL Agent (PPO/SAC) trained on Inventory Simulation Environment
    → So sánh Fill Rate, Total Cost, GMROI với Main Method
    → Nếu RL tốt hơn đáng kể → có thể nâng lên làm Main Method
```

---

## Môi trường Gymnasium cần xây dựng

```python
# Skeleton môi trường tồn kho cho RL
import gymnasium as gym

class MultiItemInventoryEnv(gym.Env):
    """
    State:  [I_i (tồn kho hiện tại), D_i (nhu cầu quan sát được), t (thời gian)]
    Action: x_i (số lượng đặt hàng cho từng SKU i) — continuous vector
    Reward: -(holding_cost + stockout_cost + ordering_cost)
    Constraint: sum(C_i * x_i) <= B  (ngân sách)
                sum(v_i * x_i) <= V_max (thể tích kho)
    """
    pass
```

---

## Bài báo RL cho Inventory cần đọc

1. [ ] Gijsbrechts et al. (2022) - "Can Deep Reinforcement Learning Improve Inventory Management?"
       (Production and Operations Management — Scopus Q1)
2. [ ] Oroojlooyjadid et al. (2022) - "A Deep Q-Network for the Beer Game: Deep Reinforcement
       Learning for Inventory Optimization" (Manufacturing & Service Operations Management)
3. [ ] Madeka et al. (2022) - "Deep Inventory Management" (Amazon Science)
4. [ ] Hubbs et al. (2020) - "OR-Gym: A Reinforcement Learning Library for Operations Research"
