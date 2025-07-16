# Translation chatbot – Dịch văn bản dài tiếng Anh sang tiếng Việt

## Vấn đề đặt ra
Đầu vào là một văn bản rất dài, vượt quá giới hạn mà một mô hình ngôn ngữ lớn (LLM) có thể xử lý trong một lần. Do đó, **việc chia nhỏ văn bản thành các đoạn (chunk)** là bắt buộc. Tuy nhiên, điều này đặt ra **ba thách thức chính**:

**1. Duy trì ngữ cảnh xuyên suốt**: Mỗi đoạn dịch phải đảm bảo liền mạch và nhất quán với các phần trước đó, tránh việc ngữ nghĩa bị rời rạc hoặc lặp lại không cần thiết.

**2. Giữ nguyên cách dịch từ ngữ/thuật ngữ chuyên ngành**: Một số từ có thể có nhiều nghĩa tùy theo ngữ cảnh. Đối với các thuật ngữ quan trọng, việc giữ nguyên cách dịch nhất quán trong toàn bộ văn bản là điều rất quan trọng để đảm bảo tính chính xác và dễ hiểu.

**3. Tối ưu chi phí xử lý**: Việc dịch một văn bản dài có thể tiêu tốn rất nhiều token, cả ở đầu vào lẫn đầu ra, dẫn đến chi phí cao. Do đó, các giải pháp được đưa ra cần cân bằng giữa chất lượng bản dịch và chi phí, sao cho tối ưu hóa chi phí mà vẫn đảm bảo hiệu quả.

## Ý tưởng

Mượn ý tưởng từ bài báo (https://arxiv.org/abs/2410.08143) về một agent dịch thuật có khả năng dịch một luồng dữ liệu liên tục (dịch khi chúng mới xuất hiện) nhưng vẫn duy trì được bối cảnh ở cấp độ toàn văn bản (tức là không chỉ dịch tốt câu rời rạc, mà phải giữ mạch văn của cả bài). 

Bài báo này đã đề xuất thiết kế một **Multi-Level Memory** để hỗ trợ dịch các đoạn nhỏ nhưng vẫn duy trì mạch văn và thuật ngữ toàn cục. Bao gồm:
- **Proper Noun Records**: Danh sách các danh từ riêng và bản dịch đầu tiên tương ứng (giúp duy trì sự nhất quán)
- **Source Summary**: Các văn bản nguồn được tóm tắt.
- **Target Summary**: Các tóm tắt bản dịch sau mỗi đoạn.
- **Short-Term Memory**: Lưu trữ các câu hoặc đoạn gần nhất, giúp duy trì ngữ cảnh ngắn hạn.
- **Long-Term Memory**: Lưu trữ các tóm tắt, thông tin quan trọng xuyên suốt văn bản.

![Multi-Level Memory Translation Framework](./image/232f8741-a7c7-48bd-827d-7afbc7d90696.png)


Tuy nhiên, cách tiếp cận này chỉ có thể giải quyết tốt 2 thách thức đầu tiên. Vì việc duy trì bộ nhớ này **phụ thuộc nhiều vào LLM** ở mỗi bước, chi phí xử lý sẽ rất cao – điều không phù hợp với bài toán yêu cầu **tối ưu hóa chi phí dịch thuật**.

## Hướng tiếp cận

Nhận thấy khả năng dịch của các mô hình đặc biệt là GPT-4 hiện tại đã rất mạnh, việc duy trì quá nhiều ngữ cảnh khi dịch từng đoạn nhỏ không còn quá cần thiết. Do đó, cách tiếp cận trong hệ thống này là:
- Duy trì một lượng ngữ cảnh vừa đủ nhằm đảm bảo tính mạch lạc và nhất quán của bản dịch.
- Tập trung vào các từ khóa chuyên ngành hoặc quan trọng, nhằm giữ nguyên cách dịch nhất quán trong toàn bộ văn bản.

**Ví dụ**: 

Cung cấp cho LLM một đầu vào bao gồm:
- Trích **một vài câu cuối** của **chunk đã dịch trước đó** (context_sentences) để làm ngữ cảnh cho chunk hiện tại. Việc này giúp mô hình giữ được mạch văn khi chuyển sang đoạn mới.
- Duy trì **một danh sách các từ khóa** (keywords) chuyên ngành, tên riêng hoặc cụm từ quan trọng đã xuất hiện trước đó cùng với bản dịch tương ứng. Danh sách này sẽ được cung cấp cho mô hình để:
  - Bắt buộc sử dụng đúng từ dịch đã cho nếu xuất hiện lại.
  - Phát hiện thêm từ khóa mới quan trọng trong đoạn mới và cập nhật danh sách.
```json
{
  "chunk": "Reached by phone, Kaggle co-founder CEO Anthony Goldbloom declined to deny that the acquisition is happening..."
  "context_sentences": "Thông tin chi tiết về giao dịch vẫn còn khá mơ hồ, nhưng vì Google sẽ tổ chức hội nghị Cloud Next tại San Francisco vào tuần này nên thông báo chính thức có thể sẽ được đưa ra sớm nhất là   vào ngày mai."
  "keywords": {
    "CEO": "giám đốc điều hành",
    "Google": "Google",
    "data science": "khoa học dữ liệu",
  }
}
```
Yêu cầu LLM:
- Dịch phần `"chunk"` sang tiếng Việt.
- Sử dụng `"context_sentences"` để đảm bảo sự liên kết mạch lạc với phần đã dịch trước đó.
- Bắt buộc dùng từ vựng đã được cung cấp trong `"keywords"`, không được thay đổi cách dịch.
-	Nếu phát hiện từ khóa mới quan trọng, thêm vào phần `"keywords"` để các chunk sau tiếp tục sử dụng nhất quán.

## Triển khai

Thực hiện triển khai theo 2 cách:
- **Phiên bản vòng for thủ công**: phiên bản này sẽ chia chunk sau đó duyệt, dich và trích xuất thông tin từng chunk. Đây là phiên bản đơn giản, dễ cài đặt và dễ kiểm soát (`Translation_chatbot_v1.py`).
- **Phiên bản LangGraph**: phiên bản sẽ thực hiện graph hóa workflow. Phiên bản này sẽ phức tạp hơn và cần hiểu về LangGraph, mở rộng tốt cho agent và streaming (`Translation_chatbot_v2.py`).

## Cách chạy

**Bước 1**: 

```bash
pip install -r requirements.txt
```

**Bước 2**: Điều chỉnh `OPENAI_API_KEY` trong `.env`


**Bước 3:** Chạy giao diện

```bash
streamlit run app.py
```

**Lưu ý**:
- Trong file `app.py` bạn có thể điều chỉnh lại max và min `chunk_size` cho giao diện để phù hợp hơn với một số tác vụ dịch văn bản lớn:
  - Mô hình GPT-3.5 Turbo phù hợp với chunk < 4000 tokens.
  - GPT-4o có thể dùng chunk lớn hơn (tối đa ~128k tokens).
