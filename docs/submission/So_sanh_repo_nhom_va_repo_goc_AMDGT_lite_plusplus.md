# BẢN SO SÁNH REPO NHÓM VÀ REPO GỐC

Repo nhóm: DucTri2207/do-an-drug-disease-hgt

Repo gốc tham chiếu: JK-Liu7/AMDGT

Mô hình mới của nhóm: AMDGT-lite++

## 1. Kết luận ngắn gọn

Repo của nhóm không sao chép nguyên xi repo gốc AMDGT. Nhóm giữ lại bài toán, dữ liệu và tinh thần học trên mạng Drug - Protein - Disease, nhưng chủ động thiết kế lại pipeline và mô hình để sạch hơn, dễ triển khai hơn và có điểm mới riêng.

Nếu nói một câu ngắn nhất:

- Repo gốc là nguồn tham chiếu.
- Repo nhóm là bản phát triển lại.
- Mô hình mới của nhóm là AMDGT-lite++.

## 2. Repo nhóm đã cải tiến gì so với repo gốc

### 2.1 Về framework

- Repo gốc dùng DGL.
- Repo nhóm dùng PyG.

Ý nghĩa:

- phù hợp hơn với môi trường Python hiện tại
- dễ tích hợp tiếp với pipeline inference và dashboard của nhóm

### 2.2 Về protocol đánh giá

- Repo gốc có cách train dễ bị nhìn vào test khi chọn model tốt nhất.
- Repo nhóm dùng validation rõ ràng để chọn checkpoint.
- Repo nhóm còn bổ sung outer 10-fold và inner validation trong từng fold.

Ý nghĩa:

- cách đánh giá sạch hơn
- dễ bảo vệ hơn trước giảng viên

### 2.3 Về đồ thị dị thể

- Repo gốc dùng các cạnh xuôi chính.
- Repo nhóm thêm reverse edges cho message passing hai chiều.

Ý nghĩa:

- node nhận được ngữ cảnh đầy đủ hơn
- phù hợp hơn với cách học trên heterograph

### 2.4 Về mức độ mô hình

- Repo gốc là AMDGT, mô hình lai nhiều nhánh.
- Repo nhóm ban đầu đi từ HGT-only baseline.
- Repo nhóm hiện nay nâng lên AMDGT-lite++.

AMDGT-lite++ của nhóm gồm:

- 1 nhánh HGT trên heterograph Drug - Protein - Disease
- 1 nhánh similarity graph cho drug
- 1 nhánh similarity graph cho disease
- 1 cơ chế gated fusion để hợp nhất các embedding

Ý nghĩa:

- giàu thông tin hơn HGT-only
- vẫn gọn hơn AMDGT gốc
- tạo được điểm mới rõ ràng cho đồ án

### 2.5 Về similarity graph

- Repo gốc dùng similarity information ở dạng dense hơn
- Repo nhóm chuyển similarity matrices thành sparse top-k graph

Ý nghĩa:

- giảm nhiễu
- giảm độ nặng khi train
- sát ý tưởng neighbor-based của repo gốc

### 2.6 Về decoder

- Repo nhóm dùng hybrid decoder cho cặp drug - disease
- đầu vào gồm embedding drug, embedding disease, tích từng phần tử và độ lệch tuyệt đối

Ý nghĩa:

- decoder biểu diễn cặp tốt hơn cách ghép đơn giản
- dễ mở rộng và thử nghiệm hơn

### 2.7 Về baseline và khả năng so sánh

Repo nhóm giữ ba mốc so sánh:

- baseline MLP
- HGT-only
- fusion_hgt tức AMDGT-lite++

Ý nghĩa:

- dễ chứng minh mô hình mới có giá trị hơn baseline
- dễ trình bày tiến hóa mô hình theo từng tầng

### 2.8 Về khả năng demo

Repo nhóm có thêm:

- inference top-k
- export artifact cho web
- dashboard demo

Ý nghĩa:

- repo nhóm không chỉ dừng ở train model
- có thể trình bày kết quả trực quan hơn

## 3. Những gì repo nhóm kế thừa từ repo gốc

- cùng bài toán drug-disease link prediction
- cùng cấu trúc dữ liệu Drug - Protein - Disease
- cùng dùng feature của drug, disease, protein
- cùng khai thác similarity information
- cùng tinh thần transformer-based graph learning

Nói cách khác:

- repo gốc là nguồn tham chiếu về ý tưởng
- repo nhóm là phần hiện thực hóa lại theo hướng riêng

## 4. Những gì repo nhóm không giống repo gốc

- không dùng DGL
- không bê nguyên class AMNTDDA
- không giữ nguyên pipeline chọn model của repo gốc
- không giữ nguyên thiết kế dense similarity
- không tự nhận là reproduction 100 phần trăm

Đây không phải điểm yếu. Ngược lại, đây là phần thể hiện nhóm có tư duy thiết kế lại, chứ không chỉ sao chép.

## 5. Cách nói khi giảng viên hỏi

Nếu thầy hỏi repo nhóm cải tiến gì so với repo gốc, có thể trả lời:

Thưa thầy, nhóm em giữ bài toán và tinh thần của AMDGT, nhưng không sao chép nguyên repo gốc. Nhóm em chuyển sang PyG, làm lại protocol đánh giá sạch hơn với validation và 10-fold, thêm reverse edges cho heterograph, rồi thiết kế mô hình AMDGT-lite++ với hai similarity branches và gated fusion. Nghĩa là repo nhóm kế thừa ý tưởng từ repo gốc, nhưng có phần thiết kế mới rõ ràng và phù hợp hơn với hướng đồ án hiện tại.

## 6. Chốt lại bằng một câu

Điểm mới cốt lõi của repo nhóm so với repo gốc là: không chỉ đổi framework, mà còn nâng mô hình từ HGT-only lên AMDGT-lite++ và đồng thời làm pipeline đánh giá, inference và trình bày kết quả chặt chẽ hơn.
