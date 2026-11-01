# Skills & AI Agent Guidelines

Thư mục này chứa các "cẩm nang tác nghiệp" (Skills) hướng dẫn AI Agent
làm việc đúng chuẩn với dự án này.

## Cấu trúc

```
skills/
├── forecasting/          # Cẩm nang dự báo chuỗi thời gian (Quantile, WRMSSE, v.v.)
├── inventory/            # Cẩm nang tồn kho (R,S policy, Lead time, Lost Sales)
├── optimization/         # Cẩm nang tối ưu hóa (MILP, OR-Tools, Knapsack)
├── rl/                   # Cẩm nang Reinforcement Learning (Gymnasium, PPO, SAC)
└── git_workflow/         # Cẩm nang quy tắc Git của nhóm
```

## Cách sử dụng

Khi nhờ AI Agent (Antigravity, Claude Code, Codex,...) thực hiện
một tác vụ trong dự án, trỏ AI vào file skill tương ứng để AI
làm đúng chuẩn phương pháp của nhóm, ví dụ:

> "Đọc file skills/forecasting/SKILL.md rồi giúp tôi
>  implement LightGBM Quantile Regression cho dữ liệu M5."
