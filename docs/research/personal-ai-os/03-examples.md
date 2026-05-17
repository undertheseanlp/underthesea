# Personal AI OS — Examples

> Last updated: 2026-05-17
> Direction note (2026-05-17): Ví dụ vẫn dùng ngôn ngữ Việt (vì đây là internal research notes), nhưng channel mapping ở Example 8 đã cập nhật theo M2 plan (Telegram/Slack/Email thay vì Zalo).

Các ví dụ minh hoạ cách Personal AI OS hoạt động khác với CLI agent / chatbot truyền thống.

## Example 1: Email triage 24/7 (Ambient)

### Chatbot truyền thống
```
User: "Đọc inbox và summarize giúp tôi"
Bot:  [đọc 50 mail, trả về summary]
User đóng app → bot ngừng.
```

### Personal AI OS
```
[Daemon chạy nền]
07:00 - Heartbeat fires: "check inbox"
      - Email watcher tìm mail mới: 3 mail
      - Email triage agent phân loại: 1 urgent, 2 newsletter
      - Push notification: "Anh ơi, có 1 mail urgent từ sếp về Q2 planning"
07:05 - User reply trong notification: "Draft phản hồi"
      - Agent draft → user approve → send qua Gmail API
```

Khác biệt: agent **chủ động trigger**, user chỉ đóng vai quyết định.

## Example 2: Multi-channel context handoff

```
[10:00 Slack] Sếp: "Em chuẩn bị deck cho meeting 14:00 nhé"
              → Agent ghi nhận task vào Memory
              
[12:30 iMessage] Bạn: "Đi cafe không?"
              → Agent nhắc user: "Bạn có deck cần xong trước 14:00"

[13:00 WhatsApp] Vợ: "Mua sữa về"
              → Agent gộp vào shopping list, nhắc 18:00
              
[13:55] Heartbeat: "deck đã xong chưa?"
              → Check folder ./decks/Q2.pptx → exists, modified 13:50
              → Push: "Deck OK, sẵn sàng meeting"
```

Channel khác nhau, agent vẫn maintain **một shared memory**.

## Example 3: Multi-device delegation

```
User (laptop): "Chụp ảnh whiteboard rồi gửi vào group"
              ↓
   Gateway nhận, parse intent
              ↓
   Orchestrator: "cần camera + Slack send"
              ↓
   ├── Node[iPhone]: capability=camera → chụp ảnh
   └── Node[laptop]: capability=slack_api → upload + gửi
```

Mỗi device là 1 node, gateway route theo capability.

## Example 4: Lane Queue protecting side-effects

```python
# Nếu KHÔNG có lane queue (parallel tool call):
agent: "Tôi sẽ gửi xác nhận"
  → tool: send_message("OK") | send_message("Đã nhận") | send_message("Cảm ơn")
  → User nhận 3 tin nhắn cùng lúc, có thể lệch thứ tự

# Với Lane Queue (serial-by-default cho channel 'messaging'):
agent: "Tôi sẽ gửi xác nhận"
  → lane[messaging]: send_message("OK") → wait → send_message("Đã nhận") → wait → ...
  → User nhận đúng 1 tin cuối cùng (agent thấy duplicate, dedupe).
```

## Example 5: Semantic Snapshot cho web

```python
# Approach cũ - raw HTML:
html = fetch_url("https://booking.com/search?...")
# → 200 KB HTML, ~50k tokens
# → LLM phải parse div/span lồng nhau, dễ lạc

# Semantic Snapshot:
snapshot = accessibility_tree(page)
# → 8 KB
# → cấu trúc:
#   - role=heading "Hotels in Hanoi"
#   - role=listbox
#     - role=option "Hilton Hanoi - $120/night - 4.5 stars"
#     - role=option "Sofitel Legend - $250/night - 4.8 stars"
# → LLM hiểu ngay, ít token, accuracy cao
```

## Example 6: Memory-as-Markdown (Karpathy style)

```
my-personal-os/
├── memory/
│   ├── people/
│   │   ├── boss-mr-nam.md      # "Thích deck dạng minimal, không bullet point"
│   │   └── wife-linh.md         # "Birthday 15/8, thích trà đào"
│   ├── projects/
│   │   ├── uts1-roadmap.md
│   │   └── q2-planning.md
│   └── habits/
│       └── morning-routine.md
├── inbox/                       # raw mail dump
└── digest/                      # daily summary do agent tạo
```

Agent **đọc/ghi markdown trực tiếp**. User cũng đọc/ghi được bằng editor. Git diff thấy agent đã thay đổi gì.

## Example 7: Heartbeat triggers

```yaml
# heartbeat.yml — schedule + trigger config
triggers:
  - name: morning_brief
    cron: "0 7 * * *"          # 7AM mỗi ngày
    action: agent.run("Tóm tắt inbox + calendar hôm nay")
    
  - name: urgent_mail
    watch: gmail.inbox
    filter: "from:boss@company.com"
    action: agent.run("Phân loại + draft reply")
    
  - name: standup_reminder
    cron: "55 9 * * 1-5"        # 9:55 sáng thứ 2-6
    action: agent.run("Chuẩn bị standup update từ git log hôm qua")
```

## Example 8: Mapping về Underthesea hiện tại

```python
# Hiện tại underthesea.agent — sync, single-shot:
from underthesea.agent import Agent, default_tools
a = Agent("assistant", tools=default_tools)
a("Tóm tắt inbox")   # ← user phải gọi tay

# Hình dung sau khi thêm Personal AI OS layer:
from underthesea.agent.os import PersonalAIOS

os = PersonalAIOS(agent=a, memory_dir="./memory")
os.add_channel("telegram", token=...)        # M2 default
os.add_channel("slack", bot_token=...)       # M2 optional
os.add_channel("email", imap_host=...)       # M2 optional
os.add_heartbeat("0 7 * * *", "Tóm tắt inbox")
os.serve()   # ← daemon chạy 24/7

# User gửi tin Telegram → Gateway nhận → Agent xử lý → reply
# 7AM mỗi sáng → Heartbeat trigger → push morning brief
```

Đây là **gap** giữa SDK hiện tại và Personal AI OS — không cần xây hết, có thể chỉ chọn 1-2 pattern.
