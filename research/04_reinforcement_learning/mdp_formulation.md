# Mô hình hóa Toán học: Quá trình Quyết định Markov (MDP) cho Quản trị Tồn kho Đa kỳ

> **Đề tài:** A Data-Driven Integrated Predict-and-Optimize Framework for Multi-Item Inventory Replenishment Under Budget and Space Constraints  
> **Tác giả:** Nguyễn Đào Nam Hải (Technical Lead & Kỹ sư Mô hình hóa)  
> **Mục tiêu:** Tài liệu toán học cơ sở cho phân hệ Học tăng cường (Module C) phục vụ thuyết trình nhóm và bản thảo Scopus Q1/Q2.

---

## 1. Giới thiệu & Bối cảnh Vận trù học

Trong lý thuyết quản trị chuỗi cung ứng cổ điển, bài toán bổ sung tồn kho định kỳ có Lead time $L \ge 1$ và chi phí cố định phát sinh khi đặt hàng ($K > 0$) thường dẫn đến chính sách tối ưu dạng $(s, S)$ (theo định lý Scarf 1960). Tuy nhiên, khi chuyển sang bài toán thực tế với:
1. Nhu cầu phi chuẩn (Non-stationary / Zero-inflated),
2. Cơ chế **Lost Sales** (hàng thiếu mất hẳn, tồn kho luôn không âm $I_t \ge 0$),
3. Sự hiện diện của ràng buộc kép (Vốn lưu động $B$ và Thể tích lưu kho $V_{\max}$),

hàm chi phí kỳ vọng không còn giữ được tính chất $K$-convexity, khiến việc tìm nghiệm giải tích hoặc quy hoạch động chính xác (Exact Dynamic Programming) rơi vào **"lời nguyền số chiều" (Curse of Dimensionality)**. 

Phương pháp Học tăng cường (Reinforcement Learning - RL) được ứng dụng như một bộ giải xấp xỉ không mô hình (Model-Free Approximate Dynamic Programming) nhằm tìm ra chính sách đặt hàng thích nghi $\pi^*(a \mid s)$ tối ưu hóa tổng chi phí vận hành qua thời gian.

---

## 2. Hệ thống Ký hiệu Toán học (OR Standard Notation)

### 2.1. Tập hợp và Chỉ số
- $\mathcal{T} = \{1, 2, \dots, T\}$: Tập các chu kỳ ra quyết định (thời gian rời rạc, $T = 52$ kỳ).
- $t \in \mathcal{T}$: Chỉ số chu kỳ hiện tại.
- $\mathcal{I} = \{1, 2, \dots, N\}$: Tập các mặt hàng (SKU). Với kịch bản demo: $N = 1$.

### 2.2. Tham số Kinh tế & Vận hành
- $L \in \mathbb{Z}_{\ge 1}$: Thời gian chờ giao hàng (Lead time cố định, $L = 2$ kỳ).
- $h$: Chi phí lưu kho trên mỗi đơn vị hàng hóa tồn cuối kỳ ($/đơn vị/kỳ).
- $p$: Chi phí phạt đứt hàng / thiệt hại lợi nhuận biên cho mỗi đơn vị Lost Sales ($p > h$).
- $k$: Chi phí cố định mỗi lần phát hành đơn hàng (Setup cost).
- $C$: Chi phí nhập đơn vị hàng hóa (Unit purchasing cost).
- $P$: Giá bán lẻ niêm yết (Selling price).
- $D_t$: Nhu cầu ngẫu nhiên phát sinh trong kỳ $t$, $D_t \ge 0$.

### 2.3. Biến Quyết định (Decision Variable)
- $x_t \ge 0$: Lượng đặt hàng tại đầu kỳ $t$.
- $y_t \in \{0, 1\}$: Biến chỉ thị đặt hàng, $y_t = \mathbb{I}(x_t > 0)$.

---

## 3. Cấu trúc Quá trình Quyết định Markov (MDP Tuple: $\langle \mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma \rangle$)

### 3.1. Không gian Trạng thái (State Space $\mathcal{S}$)
Để quá trình đạt tính chất Markov, trạng thái tại kỳ $t$ phải chứa đủ thông tin để phân phối xác suất của kỳ $t+1$ chỉ phụ thuộc vào $(s_t, a_t)$. Khi Lead time $L \ge 2$, lượng hàng đang di chuyển trên đường (Pipeline) bắt buộc phải nằm trong không gian trạng thái:

$$\mathbf{s}_t = \Big( I_t, \mathbf{PL}_t, D_{t-1}, \tau_t \Big) \in \mathcal{S}$$

Trong đó:
- $I_t \in \mathbb{R}_{\ge 0}$: Tồn kho khả dụng thực tế tại kho đầu kỳ $t$.
- $\mathbf{PL}_t = [q_{t-L+1}, q_{t-L+2}, \dots, q_{t-1}] \in \mathbb{R}_{\ge 0}^{L-1}$: Vector các đơn hàng đã đặt ở các kỳ trước nhưng chưa cập bến. Với $L = 2$, $\mathbf{PL}_t = [q_{t-1}]$.
- $D_{t-1}$: Mức nhu cầu quan sát được của kỳ trước (cung cấp tín hiệu xu hướng).
- $\tau_t = t \pmod T$: Vị trí thời gian trong chu kỳ năm.

### 3.2. Không gian Hành động (Action Space $\mathcal{A}$)
Tại mỗi kỳ $t$, Agent lựa chọn hành động đặt hàng:
$$a_t = x_t \in \mathcal{A} \subseteq [0, Q_{\max}]$$
- Đối với thuật toán DQN: Hành động được rời rạc hóa thành lưới $K$ mức: $\mathcal{A} = \{0, 5, 10, 15, \dots, Q_{\max}\}$.
- Đối với thuật toán PPO: Hành động là biến liên tục trên đoạn $[0, Q_{\max}]$, sau đó làm tròn nguyên $x_t = \lfloor a_t \rceil$ trước khi thực thi vào môi trường.

### 3.3. Phương trình Cân bằng Động & Hàm chuyển trạng thái ($\mathcal{P}$)
Quy trình chuyển đổi trạng thái $s_t \to s_{t+1}$ tuân thủ đúng thứ tự vật lý của chuỗi cung ứng bán lẻ:

1. **Cập bến đơn hàng:** Đơn hàng phát hành tại kỳ $t-L$ chính thức nhập kho:
   $$I_t^{\text{avail}} = I_t + Q_{t-L}$$
2. **Cập nhật đường ống:** Đơn hàng mới $x_t$ được đưa vào cuối pipeline:
   $$\mathbf{PL}_{t+1} = [q_{t-L+2}, \dots, q_{t-1}, x_t]$$
3. **Phát sinh nhu cầu ngẫu nhiên:** $D_t \sim F_D$.
4. **Bán hàng & Xử lý Lost Sales:**
   $$\text{Sales}_t = \min\big(I_t^{\text{avail}}, D_t\big)$$
   $$\text{LostSales}_t = \max\big(0, D_t - I_t^{\text{avail}}\big)$$
5. **Tồn kho chuyển kỳ:**
   $$I_{t+1} = I_t^{\text{avail}} - \text{Sales}_t = \max\big(0, I_t^{\text{avail}} - D_t\big) \ge 0$$

### 3.4. Hàm Thưởng (Reward Function $\mathcal{R}$)
Mục tiêu là tối thiểu hóa tổng chi phí vận hành kỳ vọng. Trong khuôn khổ RL, hàm phần thưởng tại bước $t$ được định nghĩa bằng giá trị âm của tổng chi phí phát sinh:

$$R(s_t, a_t, s_{t+1}) = - \Big( \underbrace{h \cdot I_{t+1}}_{\text{Holding Cost}} + \underbrace{p \cdot \text{LostSales}_t}_{\text{Stockout Penalty}} + \underbrace{k \cdot \mathbb{I}(x_t > 0)}_{\text{Fixed Ordering Cost}} + \underbrace{C \cdot x_t}_{\text{Purchasing Cost}} \Big)$$

### 3.5. Hàm Mục tiêu Tối ưu hóa Toàn cục
Tìm kiếm chính sách tối ưu $\pi^*: \mathcal{S} \to \mathcal{A}$ cực đại hóa tổng phần thưởng chiết khấu kỳ vọng tích lũy qua toàn bộ chân trời quyết định:

$$\pi^* = \arg\max_\pi \mathbb{E}\left[ \sum_{t=0}^{T} \gamma^t R(s_t, a_t, s_{t+1}) \;\Bigg|\; s_0, \pi \right]$$

với hệ số chiết khấu $\gamma = 0.99$.

---

## 4. Phương trình Bellman & Cơ sở Huấn luyện PPO/DQN

### 4.1. Phương trình Bellman cho Hàm Giá trị Tối ưu
Theo nguyên lý tối ưu Bellman:
$$V^*(s) = \max_{a \in \mathcal{A}} \left\{ R(s, a) + \gamma \sum_{s' \in \mathcal{S}} \mathcal{P}(s' \mid s, a) V^*(s') \right\}$$

Hàm giá trị hành động tối ưu $Q^*(s, a)$:
$$Q^*(s, a) = R(s, a) + \gamma \mathbb{E}_{s'}\left[ \max_{a'} Q^*(s', a') \right]$$

- **DQN (Deep Q-Network):** Xấp xỉ $Q^*(s, a) \approx Q(s, a; \theta)$ bằng mạng nơ-ron sâu, cập nhật thông qua hàm mất mát sai số Bellman (Mean Squared Bellman Error):
  $$\mathcal{L}(\theta) = \mathbb{E}_{(s, a, r, s')}\left[ \left( r + \gamma \max_{a'} Q(s', a'; \theta^-) - Q(s, a; \theta) \right)^2 \right]$$

- **PPO (Proximal Policy Optimization):** Sử dụng kiến trúc Actor-Critic. Actor tối ưu hàm mục tiêu cắt tỉa (Clipped Surrogate Objective) nhằm kiểm soát bước cập nhật chính sách ổn định:
  $$\mathcal{L}^{\text{CLIP}}(\theta) = \hat{\mathbb{E}}_t \left[ \min\left( r_t(\theta) \hat{A}_t, \; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]$$
  trong đó $r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}$ và $\hat{A}_t$ là giá trị hàm lợi thế ước lượng qua GAE (Generalized Advantage Estimation).

---

## 5. Luận giải Khoa học: Tại sao RL Vượt trội hơn Phương pháp Cổ điển trong Bối cảnh v2?

| Khía cạnh | Chính sách $(s, S)$ / (R, S) cổ điển | Khung Predict-then-Optimize v1 | Phương pháp Học tăng cường (RL) |
| :--- | :--- | :--- | :--- |
| **Giả định phân phối nhu cầu** | Thường ép phân phối chuẩn độc lập i.i.d. | Dự báo điểm trung bình (Poisson/MSE). | **Không tham số (Model-Free)**: Học trực tiếp từ dữ liệu mô phỏng bất kỳ. |
| **Bảo toàn Safety Stock** | Dễ vỡ khi có ràng buộc ngân sách bên ngoài. | Bị Knapsack cắt tỉa phá hủy hoàn toàn. | **Tích hợp nội sinh**: Tự cân đối mức tồn kho đệm ngay trong không gian phần thưởng. |
| **Xử lý Lead time $L \ge 2$** | Phức tạp, công thức xấp xỉ sai lệch khi Lost Sales. | Bị bỏ qua (ép $L=0$). | **Tự nhiên**: Pipeline được nhúng trực tiếp vào không gian trạng thái Markov. |
| **Tính thích nghi thời gian** | Tham số $s, S$ tĩnh, không đổi theo mùa vụ. | Phụ thuộc hoàn toàn vào sai số dự báo cuốn chiếu. | **Động lực học thích nghi**: Nhận diện đặc trưng thời gian $\tau_t$ để gom đơn đón đầu mùa cao điểm. |
