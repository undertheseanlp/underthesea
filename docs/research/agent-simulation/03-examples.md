# Agent Simulation — Example Scenarios

> Last updated: 2026-04-11

## 1. CS Chatbot: Hỗ trợ cài đặt underthesea

### Scenario 1: Python version không tương thích

```yaml
name: install-error-python-version
persona: |
  Sinh viên IT năm 2, dùng Windows 11.
  Biết cơ bản về Python, đã cài Python từ Microsoft Store.
  Chưa biết cách quản lý nhiều version Python.
task: |
  pip install underthesea bị lỗi:
  ERROR: Package 'underthesea' requires a different Python: 3.8.10 not in '>=3.10'
expected_outcome: |
  Agent hướng dẫn:
  1. Kiểm tra Python version hiện tại
  2. Cài Python >= 3.10 từ python.org
  3. Tạo virtual environment với Python mới
  4. Cài underthesea trong venv mới
  User confirm cài thành công.
max_turns: 10
```

**Expected conversation flow:**

```
User: Em cài underthesea bị lỗi ạ. Nó báo "requires a different Python: 3.8.10"
Agent: Bạn đang dùng Python 3.8, nhưng underthesea cần Python >= 3.10. Hãy kiểm tra version...
User: Em gõ python --version thì ra 3.8.10 ạ
Agent: Bạn cần cài Python 3.10 trở lên. Vào python.org/downloads tải bản mới nhất...
User: Em tải xong rồi, giờ gõ python --version vẫn ra 3.8 ạ
Agent: Trên Windows, bạn cần dùng py -3.12 để chọn đúng version. Hoặc tạo venv...
User: Em tạo venv rồi, pip install underthesea chạy được rồi! Cảm ơn ạ
```

---

### Scenario 2: Thiếu Rust compiler

```yaml
name: install-error-rust-compiler
persona: |
  Data scientist, dùng macOS M2.
  Quen dùng pip, conda. Chưa bao giờ cài Rust.
task: |
  pip install underthesea bị lỗi dài:
  error: can't find Rust compiler
  ...
  note: This error originates from a subprocess, and is likely not a problem with pip.
expected_outcome: |
  Agent hướng dẫn cài Rust toolchain qua rustup và cài lại underthesea.
max_turns: 8
```

---

### Scenario 3: Import error — thiếu deep learning deps

```yaml
name: import-error-torch
persona: |
  Researcher NLP, dùng Ubuntu 22.04, muốn dùng dependency parsing.
  Quen với PyTorch nhưng chưa cài trong environment hiện tại.
task: |
  from underthesea import dependency_parse
  → ImportError: No module named 'torch'
expected_outcome: |
  Agent hướng dẫn: pip install "underthesea[deep]"
  hoặc pip install torch transformers riêng.
max_turns: 6
```

---

### Scenario 4: word_tokenize output sai

```yaml
name: word-tokenize-wrong-output
persona: |
  Backend developer Python, đang integrate underthesea vào search engine.
  Có kinh nghiệm với NLP cơ bản.
task: |
  word_tokenize("Hà Nội là thủ đô") trả về ['Hà', 'Nội', 'là', 'thủ', 'đô']
  thay vì ['Hà Nội', 'là', 'thủ đô'].
  User muốn biết tại sao compound words không được nhận diện.
expected_outcome: |
  Agent giải thích word_tokenize cần download model,
  hướng dẫn chạy đúng cách để có kết quả word segmentation chính xác.
max_turns: 6
```

---

### Scenario 5: Agent không phản hồi (API key issue)

```yaml
name: agent-no-response
persona: |
  Developer mới dùng underthesea agent lần đầu.
  Copy code từ README nhưng chưa set API key.
task: |
  from underthesea.agent import Agent, LLM
  agent = Agent(name="bot", provider=LLM())
  → ValueError: No API key found. Set OPENAI_API_KEY...
expected_outcome: |
  Agent hướng dẫn cách set API key cho provider (OpenAI, Azure, Anthropic, Gemini)
  và test lại.
max_turns: 8
```

## 2. Customer Service: E-commerce Chatbot

### Scenario 6: Hỏi về tình trạng đơn hàng

```yaml
name: order-status-inquiry
persona: |
  Khách hàng 45 tuổi, không rành công nghệ.
  Đặt hàng 5 ngày trước, chưa nhận được.
  Bắt đầu sốt ruột.
task: |
  Hỏi tình trạng đơn hàng mã #DH20260411.
  Nếu agent hỏi thêm thông tin, cung cấp: tên Nguyễn Văn A, SĐT 0901234567.
expected_outcome: |
  Agent tra cứu đơn hàng, cung cấp thông tin trạng thái,
  và ước tính thời gian giao hàng.
max_turns: 6
```

### Scenario 7: Khách hàng giận dữ — sản phẩm lỗi

```yaml
name: angry-customer-defective-product
persona: |
  Khách hàng mua laptop, nhận hàng bị trầy xước màn hình.
  Rất giận, đã gọi hotline 2 lần không được.
  Nói thẳng, không kiên nhẫn.
task: |
  Yêu cầu đổi trả sản phẩm hoặc hoàn tiền.
  Nếu agent đề nghị sửa chữa, từ chối và yêu cầu đổi mới.
expected_outcome: |
  Agent xử lý khiếu nại chuyên nghiệp, xin lỗi, và hướng dẫn quy trình đổi trả.
  Không escalate lên manager trừ khi user yêu cầu.
max_turns: 10
```

## 3. Education: Tutoring Agent

### Scenario 8: Học sinh hỏi về NLP

```yaml
name: student-asks-about-nlp
persona: |
  Sinh viên năm 3 ngành CNTT, đang làm đồ án tốt nghiệp về NLP.
  Biết cơ bản về machine learning, chưa biết gì về NLP.
task: |
  Hỏi agent giải thích word segmentation trong tiếng Việt là gì,
  tại sao nó khó hơn tiếng Anh, và underthesea xử lý như thế nào.
expected_outcome: |
  Agent giải thích rõ ràng, có ví dụ cụ thể,
  phù hợp level sinh viên (không quá academic, không quá đơn giản).
max_turns: 6
```

## 4. Adversarial Scenarios

### Scenario 9: User cố tình off-topic

```yaml
name: off-topic-redirect
persona: |
  User bắt đầu hỏi về cài đặt nhưng chuyển sang hỏi về thời tiết,
  bóng đá, rồi quay lại vấn đề kỹ thuật.
task: |
  Bắt đầu: "Cài underthesea bị lỗi"
  Sau 2 turn: "À mà hôm nay thời tiết đẹp nhỉ, bạn có xem bóng đá không?"
  Sau đó quay lại: "Ok quay lại vấn đề cài đặt..."
expected_outcome: |
  Agent lịch sự redirect về topic chính, không bị distracted.
max_turns: 8
```

### Scenario 10: User hỏi ngoài khả năng

```yaml
name: out-of-scope
persona: |
  Developer hỏi agent hỗ trợ underthesea về vấn đề không liên quan.
task: |
  "Bạn có thể giúp tôi deploy ứng dụng lên AWS không?"
expected_outcome: |
  Agent thừa nhận ngoài phạm vi hỗ trợ,
  gợi ý nơi tìm giúp đỡ phù hợp.
max_turns: 4
```
