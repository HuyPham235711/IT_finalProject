

## 🧩 1️⃣ ROI / PnL — Lợi nhuận cơ bản

**ROI (Return on Investment)**:
[
ROI_t = \frac{P_{t+1} - P_t}{P_t}
]

* Là phần trăm thay đổi giá giữa 2 thời điểm.
* Dùng làm reward cơ bản nhất cho agent (mua thì lời khi giá tăng, bán thì lời khi giá giảm).
* Nhưng nếu chỉ dùng ROI → agent sẽ **đánh đổi rủi ro** để ăn reward cao → cần thêm các chỉ số sau để cân bằng.

---

## 🧩 2️⃣ Sharpe Ratio — Lợi nhuận so với rủi ro

**Sharpe Ratio** đo xem lợi nhuận có “xứng đáng” với mức biến động rủi ro hay không:

[
Sharpe = \frac{E[R - R_f]}{\sigma_R}
]

Trong đó:

* ( R ): chuỗi lợi nhuận của agent
* ( R_f ): lãi suất phi rủi ro (≈ 0 trong crypto)
* ( \sigma_R ): độ lệch chuẩn (biến động) của lợi nhuận

**Giá trị càng cao càng tốt.**
Nếu reward = ROI, agent có thể “liều mạng”;
nếu reward = ROI / volatility → agent sẽ **ưu tiên lợi nhuận ổn định**.

→ Ta có thể áp dụng như reward phụ:
[
Reward_{Sharpe} = \frac{ROI_t}{std(ROI_{1:t}) + \epsilon}
]

---

## 🧩 3️⃣ MDD — Maximum Drawdown (Sụt giảm cực đại)

**MDD** đo rủi ro thua lỗ tồi tệ nhất trong chu kỳ vốn:

[
MDD = \frac{Peak - Trough}{Peak}
]

* Peak: giá trị vốn cao nhất đạt được
* Trough: giá trị vốn thấp nhất sau đó

**Ví dụ:**
vốn tăng 1000 → 1200 → rơi còn 900
→ MDD = (1200-900)/1200 = 25%

**Reward liên quan:**
phạt agent khi drawdown lớn:
[
Reward_{MDD} = - \lambda \cdot \frac{MDD_t}{MDD_{max}}
]

---

## 🧩 4️⃣ Transaction Cost — Phí giao dịch

Mỗi lần thực hiện *Buy/Sell*, agent sẽ bị trừ phí:
[
Reward_{cost} = - \text{fee_rate} \times |\text{position change}|
]

Ví dụ sàn Binance phí 0.1% → `fee_rate = 0.001`.

→ Agent sẽ học cách **giảm giao dịch thừa**, chỉ trade khi xác suất cao.

---

## 🧩 5️⃣ Slippage — Trượt giá

Thực tế lệnh mua bán **không khớp đúng giá thị trường** do thanh khoản.
→ Ta mô phỏng bằng:
[
price_{exec} = price_t \times (1 + \text{slippage_rate})
]
với buy thì cộng, sell thì trừ.

Reward thực tế = lợi nhuận dựa trên `price_exec` thay vì `price_t`.

---

## 🧩 6️⃣ Composite Reward Function

Kết hợp 4 yếu tố:
[
R_t = w_1 \cdot ROI_t + w_2 \cdot Sharpe_t - w_3 \cdot MDD_t - w_4 \cdot Cost_t
]

* ( w_1 \dots w_4 ) là trọng số, ví dụ:
  `w1=1.0, w2=0.5, w3=0.8, w4=0.3`

---

## ✅ Tóm tắt hướng bạn sẽ làm trong Task 3:

| Thành phần       | Vai trò                 | Cách tính                         | Ghi chú        |
| ---------------- | ----------------------- | --------------------------------- | -------------- |
| ROI              | lợi nhuận cơ bản        | (Pₜ₊₁ - Pₜ)/Pₜ                    | base reward    |
| Sharpe           | lợi nhuận so với rủi ro | mean(ROI)/std(ROI)                | ổn định        |
| MDD              | rủi ro drawdown         | max peak−trough                   | phạt           |
| Transaction cost | phí trade               | fee × volume                      | phạt           |
| Slippage         | trượt giá               | ± slip_rate                       | điều chỉnh giá |
| Composite reward | kết hợp có trọng số     | w₁ROI + w₂Sharpe − w₃MDD − w₄Cost | tối ưu RL      |

---


