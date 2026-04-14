# BÁO CÁO MÔ HÌNH MỚI CỦA NHÓM

AMDGT-lite++ cho bài toán dự đoán liên kết Drug - Disease trên mạng Drug - Protein - Disease

Repo của nhóm: https://github.com/DucTri2207/do-an-drug-disease-hgt

Repo gốc tham chiếu: https://github.com/JK-Liu7/AMDGT

## 1. Mục tiêu của mô hình mới

Mục tiêu của nhóm là nâng từ mô hình HGT-only trước đây lên một mô hình mới có tính mới rõ hơn, nhưng vẫn train được và vẫn bảo vệ được logic học thuật.

Mô hình mới của nhóm có tên trình bày là AMDGT-lite++.

Ý tưởng chính là:

- giữ HGT heterograph branch làm lõi
- thêm drug similarity branch
- thêm disease similarity branch
- dùng gated fusion để hợp nhất các embedding

## 2. Vì sao cần mô hình mới

Phiên bản HGT-only trước đây có ưu điểm là rõ ràng, dễ chạy và dễ giải thích. Tuy nhiên, nếu chỉ dùng heterograph thì mô hình vẫn chưa tận dụng hết phần similarity information vốn đã có trong dữ liệu AMDGT.

Trong khi đó, repo gốc AMDGT đã dùng nhiều nguồn thông tin hơn. Vì vậy, để đồ án có tính mới hơn và gần hơn với hướng của paper tham chiếu, nhóm cần một mô hình mới ở giữa hai mức:

- mạnh hơn HGT-only
- gọn hơn AMDGT gốc

AMDGT-lite++ chính là lời giải nhóm chọn.

## 3. Kiến trúc AMDGT-lite++

### 3.1 Nhánh heterograph

Nhánh này giữ nguyên tinh thần của mô hình HGT hiện có:

- dùng mạng Drug - Protein - Disease
- có reverse edges
- dùng HGTConv để học embedding dị thể

Vai trò của nhánh này là học ngữ cảnh sinh học từ quan hệ:

- drug - protein
- protein - disease
- drug - disease

### 3.2 Nhánh similarity cho drug

Nhóm kết hợp:

- DrugFingerprint
- DrugGIP

Sau đó tạo một drug similarity graph dạng sparse top-k. Nhánh này học embedding cho drug từ góc nhìn tương đồng giữa các thuốc.

### 3.3 Nhánh similarity cho disease

Nhóm kết hợp:

- DiseasePS
- DiseaseGIP

Sau đó tạo disease similarity graph dạng sparse top-k. Nhánh này học embedding cho disease từ góc nhìn tương đồng giữa các bệnh.

### 3.4 Fusion

Sau khi có:

- embedding từ heterograph
- embedding từ similarity graph

nhóm dùng gated fusion để học cách trộn hai nguồn thông tin. Đây là điểm mới quan trọng nhất của AMDGT-lite++.

Nói đơn giản:

- nếu node nào cần tin nhiều vào heterograph thì gate sẽ nghiêng về heterograph
- nếu node nào cần tin nhiều vào similarity view thì gate sẽ nghiêng về similarity

### 3.5 Decoder

Ở đầu ra, mô hình lấy embedding của drug và disease, sau đó dùng hybrid decoder để tạo điểm số liên kết.

Đầu vào của decoder gồm:

- embedding drug
- embedding disease
- tích từng phần tử
- độ lệch tuyệt đối

## 4. Điểm mới của AMDGT-lite++ so với bản HGT-only cũ

- thêm 2 similarity branches
- thêm sparse top-k similarity graph
- thêm gated fusion
- dùng hybrid decoder làm mặc định
- tăng hard negative ratio cho mô hình fusion

Điều này làm mô hình mới giàu thông tin hơn phiên bản HGT-only trước đây.

## 5. Điểm mới của repo nhóm so với repo gốc AMDGT

- dùng PyG thay vì DGL
- có protocol outer 10-fold và inner validation
- không chọn model theo test
- thêm reverse edges rõ ràng
- tách riêng baseline, HGT-only và fusion_hgt để so sánh
- có inference và dashboard

Nghĩa là repo nhóm không chỉ thay đổi mô hình, mà còn làm sạch lại toàn bộ pipeline thực nghiệm.

## 6. Cách đánh giá mô hình mới

Nhóm chốt protocol đánh giá như sau:

- outer 10-fold stratified cross-validation
- trong mỗi fold tách thêm validation
- chọn checkpoint tốt nhất theo validation AUPR
- test fold chỉ dùng để đánh giá cuối cùng

Các metric báo cáo gồm:

- AUC
- AUPR
- Accuracy
- Precision
- Recall
- F1
- MCC

## 7. Ý nghĩa học thuật của mô hình mới

AMDGT-lite++ giúp nhóm bảo vệ đề tài theo một logic rõ:

- mô hình gốc nền tảng là HGT
- mô hình tham chiếu trực tiếp là AMDGT
- mô hình mới của nhóm là AMDGT-lite++

Như vậy, nhóm không chỉ học lại mô hình cũ mà còn có phần tự thiết kế:

- similarity fusion
- sparse top-k graph
- gated fusion
- protocol đánh giá sạch hơn

## 8. Cách nói ngắn với giảng viên

Thưa thầy, repo gốc AMDGT là nguồn tham chiếu chính của nhóm về dữ liệu và ý tưởng mô hình. Tuy nhiên nhóm không sao chép nguyên repo đó. Nhóm phát triển một biến thể mới tên AMDGT-lite++, trong đó giữ lõi HGT trên heterograph, thêm hai nhánh similarity cho drug và disease, rồi dùng gated fusion để hợp nhất embedding. Đồng thời nhóm dùng protocol đánh giá sạch hơn với 10-fold và validation trong từng fold.

## 9. Kết luận

AMDGT-lite++ là mô hình mới của nhóm, được thiết kế để tạo điểm mới rõ ràng so với bản HGT-only cũ nhưng vẫn khả thi hơn việc tái hiện toàn bộ AMDGT gốc. Giá trị chính của mô hình này là kết hợp được cả thông tin dị thể và thông tin tương đồng trong một pipeline đánh giá chặt chẽ hơn.
