---
name: mdp-rl-inventory-modeling
description: Cẩm nang tác nghiệp quy chuẩn mô hình hóa MDP, thiết kế môi trường Gymnasium (Lost Sales, L >= 2) và huấn luyện, đánh giá Agent Deep RL (PPO, DQN) đối chuẩn cho bài toán quản trị tồn kho đa kỳ theo chuẩn Scopus Q1/Q2.
---

# SKILL: Quy chuẩn Xây dựng & Đánh giá Học tăng cường (RL) cho Quản trị Tồn kho

## 1. Mục tiêu & Phạm vi
Cẩm nang này định nghĩa Quy trình thao tác chuẩn (SOP) cho việc mô hình hóa, cài đặt môi trường giả lập Gymnasium, huấn luyện và đánh giá các thuật toán Học tăng cường (Reinforcement Learning - PPO, DQN, SAC) trong bài toán quản trị tồn kho đa kỳ.
RL đóng vai trò là **phương pháp đối chuẩn (Benchmark)** đối đầu trực diện với khung tích hợp Học máy & Quy hoạch toán học (ML + MILP) theo chuẩn các tạp chí Scopus Q1/Q2 (EJOR, COR, IJPE, POM).

---

## 2. Mô hình hóa Toán học: Quá trình Quyết định Markov (MDP)

Mọi môi trường tồn kho trong dự án BẮT BUỘC phải thỏa mãn tính chất Markov: $\mathbb{P}(s_{t+1} \mid s_t, a_t, s_{t-1}, a_{t-1}, \dots) = \mathbb{P}(s_{t+1} \mid s_t, a_t)$.

Bài toán tồn kho đa kỳ được mô hình hóa thành bộ 5 thành phần $(S, A, P, R, \gamma)$:

### 2.1. Không gian trạng thái (State Space $\mathcal{S}$)
Tại mỗi kỳ $t$ (chu kỳ ra quyết định):
- $I_t \in \mathbb{R}_{\ge 0}$: Mức tồn kho khả dụng hiện tại (On-hand inventory), cơ chế Lost Sales đảm bảo $I_t \ge 0$.
- $\mathbf{PL}_t = [q_{t-L+1}, \dots, q_{t-1}] \in \mathbb{R}_{\ge 0}^{L-1}$: Vector đường ống hàng đang vận chuyển (In-transit pipeline) khi Lead time $L \ge 2$. Nếu $L=1$, vector này rỗng. Bắt buộc phải đưa $\mathbf{PL}_t$ vào trạng thái để bảo toàn tính Markov.
- $D_{t-1} \in \mathbb{R}_{\ge 0}$: Nhu cầu quan sát được của kỳ trước đó (Observed recent demand).
- $\tau_t = t \pmod T$: Đặc trưng chu kỳ thời gian (mùa vụ, vị trí trong tháng/năm).

$$\mathbf{s}_t = [I_t, \mathbf{PL}_t, D_{t-1}, \tau_t]$$

### 2.2. Không gian hành động (Action Space $\mathcal{A}$)
- **Đơn sản phẩm (DQN / PPO discrete):** $a_t = x_t \in \{0, 5, 10, 15, \dots, Q_{\max}\}$ đại diện cho lượng đặt hàng rời rạc.
- **Đa sản phẩm (PPO / SAC continuous):** $\mathbf{a}_t = [x_{1,t}, x_{2,t}, \dots, x_{N,t}]^\top \in \mathbb{R}_{\ge 0}^N$ với cơ chế chuẩn hóa hành động (Action Box $[0, 1]^N$ sau đó denormalize).

### 2.3. Động lực học & Hàm chuyển trạng thái (Transition Function $\mathcal{P}$)
Quy trình diễn ra tuần tự trong mỗi bước thời gian $t$:
1. **Nhận hàng:** Lượng hàng đặt từ kỳ $t-L$ ($Q_{t-L}$) cập bến kho:
   $$I_t^{\text{avail}} = I_t + Q_{t-L}$$
2. **Quyết định đặt hàng:** Agent quan sát trạng thái $\mathbf{s}_t$ và phát sinh hành động đặt hàng $x_t$. Đơn hàng $x_t$ được đẩy vào cuối hàng đợi vận chuyển (Pipeline).
3. **Phát sinh nhu cầu thị trường:** Nhu cầu $D_t \sim \text{Poisson}(\lambda)$ hoặc lấy từ chuỗi dữ liệu thực tế M5 Walmart.
4. **Đáp ứng nhu cầu & Cơ chế Lost Sales (Bán lẻ Walmart):**
   - Lượng bán thực tế: $\text{Sales}_t = \min(I_t^{\text{avail}}, D_t)$
   - Lượng thiếu hàng (mất hẳn, không bảo lưu): $\text{LostSales}_t = \max(0, D_t - I_t^{\text{avail}})$
   - Tồn kho chuyển sang kỳ kế tiếp: $I_{t+1} = I_t^{\text{avail}} - \text{Sales}_t \ge 0$

### 2.4. Hàm thưởng (Reward Function $\mathcal{R}$)
Mục tiêu của quản trị tồn kho là tối thiểu hóa tổng chi phí vận hành. Do đó, hàm thưởng là số âm của tổng chi phí:
$$R_t = - \Big( h \cdot I_{t+1} + p \cdot \text{LostSales}_t + k \cdot \mathbb{I}(x_t > 0) + C \cdot x_t \Big)$$
Trong đó:
- $h$: Chi phí lưu kho cho mỗi đơn vị tồn dư cuối kỳ ($/unit/period).
- $p$: Chi phí phạt đứt hàng / lợi nhuận biên bị mất cho mỗi đơn vị Lost Sales ($p \gg h$).
- $k$: Chi phí cố định mỗi lần phát sinh đặt hàng (Fixed setup cost). $\mathbb{I}(x_t > 0) = 1$ nếu $x_t > 0$, ngược lại bằng $0$.
- $C$: Giá vốn đơn vị sản phẩm (tính theo Gross Margin chuẩn ngành).

### 2.5. Hệ số chiết khấu (Discount Factor $\gamma$)
- $\gamma = 0.99$: Khuyến khích Agent tối ưu tổng chi phí kỳ vọng dài hạn qua toàn bộ chu kỳ, triệt tiêu xu hướng tham lam (myopic) chỉ đặt hàng cho riêng kỳ hiện tại.

---

## 3. Kiến trúc Cài đặt Môi trường Gymnasium (`MultiItemInventoryEnv`)

Mọi lớp môi trường trong `src/rl_agent/envs/` BẮT BUỘC kế thừa `gymnasium.Env` và tuân thủ:
1. `__init__(self, config)`: Khởi tạo observation_space và action_space chuẩn `gymnasium.spaces.Box` hoặc `gymnasium.spaces.Discrete`.
2. `reset(self, seed=None, options=None)`: Cố định seed cho `np.random` và reset trạng thái kho, pipeline về trạng thái ban đầu.
3. `step(self, action)`: Trả về bộ 5 giá trị chuẩn Gymnasium: `(observation, reward, terminated, truncated, info)`.
   - `info` BẮT BUỘC chứa: `{'sales': ..., 'lost_sales': ..., 'holding_cost': ..., 'stockout_cost': ..., 'ordering_cost': ..., 'demand': ...}`.
4. Xử lý ràng buộc vốn ($B$) và thể tích ($V_{\max}$):
   - Đơn giản / Benchmark: Dùng hàm phạt mềm (Penalty Reward) khi $\sum C_i x_i > B$ hoặc $\sum v_i (x_i + I_i) > V_{\max}$.
   - Nâng cao: Dùng Action Masking (cho Discrete) hoặc Action Projection (chiếu về tập khả thi lồi) trước khi thực thi `step()`.

---

## 4. Bộ Tiêu chí Đánh giá & Đối chuẩn Bắt buộc (Evaluation Metrics)

Sau khi huấn luyện Agent, BẮT BUỘC đánh giá trên tập kiểm thử (100 episodes độc lập với các seed khác nhau) và tính toán 5 chỉ số:

| Chỉ số | Ký hiệu | Công thức toán học | Ý nghĩa kinh tế |
| :--- | :---: | :--- | :--- |
| **Fill Rate** | $\text{FR}$ | $\frac{\sum_{t=1}^T \text{Sales}_t}{\sum_{t=1}^T D_t} \times 100\%$ | Tỷ lệ phần trăm tổng nhu cầu thực tế được đáp ứng ngay lập tức. |
| **Cycle Service Level** | $\text{CSL}$ | $\frac{1}{T} \sum_{t=1}^T \mathbb{I}(\text{LostSales}_t = 0) \times 100\%$ | Tỷ lệ các kỳ kinh doanh hoàn toàn không bị hết hàng. |
| **Tổng Chi phí Vận hành** | $\text{TC}$ | $\sum_{t=1}^T (\text{Holding} + \text{Stockout} + \text{Ordering})$ | Thước đo tối thượng để xếp hạng các mô hình. |
| **GMROI** | $\text{GMROI}$ | $\frac{\text{Lợi nhuận gộp}}{\text{Giá trị tồn kho bình quân}}$ | Hiệu quả sinh lời trên một đồng vốn lưu kho. |
| **Số lần Đứt hàng** | $\text{Stockouts}$ | $\sum_{t=1}^T \mathbb{I}(I_t^{\text{avail}} < D_t)$ | Tần suất gián đoạn nguồn cung. |

---

## 5. Danh mục Mô hình Đối chuẩn (Baselines)

Agent RL (PPO/DQN) BẮT BUỘC phải được so sánh đối đầu với:
1. **Random Policy:** Chọn hành động ngẫu nhiên đều (Lower bound - chứng minh mô hình có học được).
2. **Base-Stock Policy:** Chính sách chuẩn tắc: Luôn đặt hàng để tổng $I_t + \sum \text{Pipeline} = S^*$.
3. **Classical $(s, S)$ Policy:** Nếu tồn kho hiệu dụng $< s^*$, đặt hàng lên mức $S^*$; ngược lại không đặt hàng.
4. **Predict-then-Optimize (v1 Baseline):** Dự báo điểm Poisson $\to$ tính $SS = z \cdot \sigma \sqrt{R}$ $\to$ giải Knapsack.

---

## 6. Kỷ luật Kỹ thuật & Cấu trúc Mã nguồn

- Mã nguồn đặt trong:
  - `src/rl_agent/envs/`: Môi trường Gymnasium (single-item & multi-item).
  - `src/rl_agent/policies/`: Các policy baseline cổ điển.
  - `src/rl_agent/train.py`: Script huấn luyện Stable-Baselines3.
  - `src/rl_agent/evaluate.py`: Script chạy benchmark và xuất bảng số liệu.
- Tham số thực nghiệm lưu tại: `experiments/configs/rl_demo_config.yaml`.
- Kiểm thử đơn vị: `tests/test_single_item_env.py` (chạy qua pytest).
- CẤM hardcode tham số chi phí $h, p, k, C$ trong code.
- CẤM commit file checkpoint trọng số model (`*.zip`, `*.pt`, `*.pkl`) lên GitHub.
