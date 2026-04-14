# BÁO CÁO NGHIÊN CỨU ĐỒ ÁN

Heterogeneous Graph Transformer (HGT) áp dụng cho mạng lưới Drug - Protein - Disease

Xây dựng website dự đoán thuốc điều trị bệnh dựa trên mạng liên kết thuốc - protein - bệnh

Nhóm thực hiện: ........................................................

Lớp: ........................................................

Giảng viên hướng dẫn: ........................................................

Ngày nộp: 14/04/2026

## 0. Phân biệt repo của nhóm và repo gốc

- Repo của nhóm: https://github.com/DucTri2207/do-an-drug-disease-hgt
- Repo gốc để tham chiếu: https://github.com/JK-Liu7/AMDGT

Trong báo cáo này, khi nói “repo của nhóm” là nói đến repo `do-an-drug-disease-hgt`. Khi nói “repo gốc” hoặc “repo AMDGT” là nói đến repo `JK-Liu7/AMDGT`.

## 1. Tóm tắt đề tài

Đề tài của nhóm thuộc bài toán dự đoán liên kết thuốc - bệnh trên đồ thị dị thể. Mục tiêu là dùng trí tuệ nhân tạo để học từ các quan hệ đã biết giữa thuốc, protein và bệnh, sau đó gợi ý các liên kết thuốc - bệnh mới. Hướng làm này phù hợp với bài toán drug repositioning, tức là tìm công dụng mới cho thuốc đã có.

Sau khi đối chiếu paper, repo gốc AMDGT và code hiện tại trong repo của nhóm, cách kết luận an toàn nhất là chia làm ba tầng. Tầng nền tảng kiến trúc là Heterogeneous Graph Transformer, gọi tắt là HGT. Tầng tham chiếu trực tiếp là AMDGT, là công trình dùng dữ liệu Drug - Protein - Disease và kết hợp nhiều nhánh đồ thị. Tầng triển khai hiện tại của nhóm là phiên bản HGT-only bằng PyG, có thêm baseline MLP, có chia train/val/test rõ ràng, có inference top-k và có dashboard demo.

Nói ngắn gọn, repo của nhóm không phải là bản sao nguyên xi của paper AMDGT. Nhóm đang giữ phần lõi hợp lý nhất cho mức đồ án cơ sở, đồng thời làm lại pipeline theo hướng sạch hơn và dễ giải thích hơn.

## 2. Bối cảnh và lý do chọn đề tài

Trong thực tế, phát triển một thuốc mới rất tốn thời gian, chi phí và có rủi ro cao. Vì vậy, một hướng nghiên cứu quan trọng là tìm cách tái sử dụng thuốc đã có cho bệnh khác. Đây là lý do bài toán drug repositioning được quan tâm trong biomedical AI.

Dữ liệu sinh học của bài toán này có dạng mạng lưới phức tạp. Một thuốc có thể tác động lên nhiều protein, và protein lại liên quan đến nhiều bệnh. Vì vậy, nếu chỉ nhìn từng cặp thuốc - bệnh riêng lẻ thì sẽ bỏ qua phần lớn thông tin quan hệ. Hướng tiếp cận bằng đồ thị giúp mô hình khai thác được cả đặc trưng của từng thực thể lẫn mối liên hệ giữa chúng.

## 3. Phát biểu bài toán

Cho một đồ thị dị thể gồm ba loại nút là drug, protein và disease. Đồ thị này chứa các liên kết đã biết như drug - protein, protein - disease và drug - disease. Mỗi loại nút cũng có vector đặc trưng riêng.

Mục tiêu của mô hình là học một hàm chấm điểm cho cặp drug - disease. Nếu điểm số cao, mô hình cho rằng cặp đó có khả năng có liên kết. Đầu ra cuối cùng của hệ là danh sách các bệnh được xếp hạng theo độ phù hợp với một thuốc đầu vào, hoặc ngược lại là điểm số của một cặp thuốc - bệnh cụ thể.

## 4. Cơ sở lý thuyết ngắn gọn

### 4.1 Drug repositioning

Drug repositioning là tìm chỉ định mới cho thuốc đã tồn tại. Điểm mạnh là có thể giảm thời gian nghiên cứu và giảm chi phí so với phát triển thuốc từ đầu.

### 4.2 Link prediction

Link prediction là dự đoán cạnh còn thiếu trong đồ thị. Trong đề tài này, cạnh cần dự đoán là liên kết giữa drug và disease.

### 4.3 Graph Neural Network

Graph Neural Network là họ mô hình học sâu trên đồ thị. Ý tưởng cơ bản là mỗi nút sẽ cập nhật biểu diễn của mình bằng cách thu thập thông tin từ các nút lân cận.

### 4.4 Heterogeneous graph

Đồ thị dị thể là đồ thị có nhiều loại nút và nhiều loại cạnh. Bài toán của nhóm là dạng dị thể vì thuốc, protein và bệnh là ba loại thực thể khác nhau và quan hệ giữa chúng cũng khác nhau.

### 4.5 Attention và Transformer

Attention cho phép mô hình học xem nên chú ý nhiều hơn vào hàng xóm nào. Transformer là kiến trúc dựa trên attention. Khi áp dụng lên đồ thị dị thể, attention giúp mô hình học được quan hệ nào quan trọng hơn cho từng loại nút.

### 4.6 HGT

HGT là Transformer dành cho heterogeneous graph. Điểm mạnh của HGT là nó không coi mọi nút và mọi cạnh là giống nhau. Thay vào đó, mô hình có thể dùng tham số khác nhau cho từng loại nút và từng loại quan hệ. Điều này phù hợp với bài toán Drug - Protein - Disease vì quan hệ drug - protein không giống quan hệ protein - disease.

## 5. Xác định mô hình gốc, mô hình tham chiếu và mô hình hiện tại

### 5.1 Cách xác định

Trong đề tài này, không nên nói đơn giản rằng chỉ có một mô hình gốc duy nhất. Cách nói an toàn và đúng logic hơn là phân ra ba tầng:

- Mô hình gốc nền tảng: Heterogeneous Graph Transformer, tức HGT.
- Mô hình tham chiếu trực tiếp từ bài báo và repo gốc: AMDGT.
- Mô hình nhóm đang triển khai hiện tại: biến thể HGT-only bằng PyG.

### 5.2 Vì sao kết luận như vậy

HGT là nền tảng kiến trúc vì paper HGT gốc được xây dựng cho heterogeneous graph và rất phù hợp với bài toán có nhiều loại nút, nhiều loại cạnh. Trong khi đó, AMDGT là công trình gần nhất với đề tài của nhóm vì dùng đúng bài toán dự đoán liên kết thuốc - bệnh trên mạng Drug - Protein - Disease, đồng thời sử dụng các bộ dữ liệu và các loại feature tương tự.

Tuy nhiên, repo của nhóm không triển khai đầy đủ toàn bộ kiến trúc AMDGT. Thay vào đó, nhóm giữ phần lõi là HGT để xây pipeline sạch, dễ chạy và dễ bảo vệ hơn. Vì vậy, cách nói đúng là repo của nhóm là một phiên bản triển khai lại theo tinh thần của bài toán AMDGT nhưng rút gọn về lõi HGT.

## 6. Mô hình tham chiếu từ paper AMDGT

Paper AMDGT đề xuất một framework đa mô thức cho bài toán drug - disease association prediction. Điểm chính của công trình này là kết hợp nhiều nguồn thông tin:

- similarity giữa các drug
- similarity giữa các disease
- heterogeneous association network giữa drug, protein và disease

Sau đó, AMDGT dùng nhiều nhánh transformer để học embedding từ các góc nhìn khác nhau, rồi fusion các embedding đó trước khi dự đoán cặp drug - disease.

Điểm quan trọng cần nói rõ là AMDGT không phải HGT thuần. Nó là mô hình hybrid, phức tạp hơn, vì ngoài heterogeneous branch còn có thêm các similarity branch và phần tương tác đa mô thức.

## 7. Mô hình hiện tại trong repo GitHub của nhóm

### 7.1 Mục tiêu của repo

Repo của nhóm tại `DucTri2207/do-an-drug-disease-hgt` tập trung xây một pipeline hoàn chỉnh cho đồ án cơ sở. Mục tiêu không phải sao chép toàn bộ AMDGT, mà là làm được phần cốt lõi, chạy được, đánh giá được và có thể demo được.

### 7.2 Dữ liệu đang dùng

Repo hiện tại đọc dữ liệu theo schema của AMDGT, gồm:

- Drug_mol2vec cho drug feature
- DiseaseFeature cho disease feature
- Protein_ESM cho protein feature
- DrugDiseaseAssociationNumber
- DrugProteinAssociationNumber
- ProteinDiseaseAssociationNumber

Ngoài ra repo cũng đã đọc được các ma trận similarity, nhưng ở giai đoạn hiện tại chúng chưa phải phần chính của mô hình.

### 7.3 Cấu trúc đồ thị

Nhóm đang xây đồ thị dị thể gồm ba loại nút:

- drug
- protein
- disease

Các loại cạnh chính hiện tại gồm:

- drug -> disease
- drug -> protein
- protein -> disease

Đồng thời repo hiện tại còn thêm reverse edges để message passing diễn ra hai chiều:

- disease -> drug
- protein -> drug
- disease -> protein

Điểm này là một thay đổi triển khai quan trọng so với repo AMDGT gốc.

### 7.4 Chia train/val/test

Một điểm mạnh của repo hiện tại là nhóm chia dữ liệu theo train/val/test rõ ràng, thay vì chỉ train/test. Tỉ lệ đang dùng là:

- train: 70%
- val: 15%
- test: 15%

Việc có validation set giúp chọn mô hình tốt nhất đúng cách. Đây là điểm cần nhấn mạnh khi bảo vệ, vì nếu chọn mô hình theo test thì kết quả sẽ thiếu chặt chẽ về phương pháp.

### 7.5 Negative sampling

Nhóm dùng nguyên tắc unknown as negative, tức là các cặp drug - disease chưa biết sẽ được xem là negative tạm thời khi tạo dữ liệu huấn luyện. Đây là cách làm phổ biến trong benchmark kiểu này, nhưng cũng là một hạn chế vì có thể tồn tại những cặp thật sự đúng nhưng chưa được phát hiện trong dữ liệu gốc.

### 7.6 Baseline hiện có

Repo hiện tại có baseline MLP. Baseline này chỉ lấy feature của drug và disease, ghép lại rồi đưa qua mạng MLP để dự đoán liên kết. Baseline có vai trò làm mốc so sánh. Nếu HGT không tốt hơn baseline, thì chưa thể kết luận việc học trên đồ thị mang lại lợi ích rõ ràng.

### 7.7 Mô hình chính hiện tại

Mô hình chính hiện tại là HGT-only bằng PyG. Quy trình tổng quát như sau:

- chiếu feature của từng loại nút về cùng một hidden dimension
- đưa graph qua nhiều lớp HGTConv
- lấy embedding của drug và disease
- kết hợp hai embedding bằng decoder
- dự đoán điểm số liên kết

Repo cũng hỗ trợ nhiều kiểu decoder khác nhau như product, concat và hybrid. Điều này giúp nhóm thử nghiệm các cách kết hợp embedding khác nhau mà không phải thay toàn bộ mô hình.

### 7.8 Loss và metric

Loss hiện tại là BCEWithLogitsLoss hoặc focal loss. Các metric được theo dõi gồm AUC, AUPR, accuracy, precision, recall, F1 và MCC. Trong đó, AUPR đặc biệt quan trọng vì bài toán thường mất cân bằng giữa positive và negative.

### 7.9 Inference và dashboard

Một điểm cộng của repo hiện tại là đã có inference top-k và dashboard demo. Sau khi huấn luyện xong, hệ có thể nhận một drug đầu vào, chấm điểm với tất cả disease, rồi trả ra danh sách top-k bệnh có điểm cao nhất. Repo cũng đã có Streamlit dashboard để trình bày kết quả trực quan hơn.

## 8. So sánh ngắn giữa HGT nền tảng, AMDGT paper và repo hiện tại

Nếu so theo ý tưởng tổng thể, repo hiện tại giống AMDGT ở chỗ cùng giải cùng một bài toán và cùng dùng dữ liệu Drug - Protein - Disease. Tuy nhiên, repo hiện tại chưa triển khai similarity branch và phần fusion đa mô thức của AMDGT.

Nếu so theo kiến trúc lõi, repo hiện tại gần HGT hơn AMDGT, vì phần mô hình chính đang dùng HGTConv và học embedding trực tiếp trên heterogeneous graph.

Vì vậy, có thể nói repo hiện tại là một phiên bản rút gọn: giữ HGT core, bỏ bớt phần fusion phức tạp của AMDGT, đồng thời làm lại quy trình train và evaluation theo hướng chặt chẽ hơn.

## 9. Kết quả bước đầu trong repo hiện tại

Theo các file kết quả hiện có trong repo:

- Baseline MLP trên C-dataset đạt test AUC khoảng 0.5621 và test AUPR khoảng 0.5625.
- Một cấu hình HGT mặc định đạt test AUC khoảng 0.5514 và test AUPR khoảng 0.5709.
- Trong quá trình tìm siêu tham số, trial tốt nhất theo validation AUPR đạt val AUPR khoảng 0.7273 và test AUPR khoảng 0.6901.

Điểm cần trình bày khéo ở đây là nhóm không chọn mô hình theo test, mà chọn theo validation. Đây là cách làm đúng và nên được xem là điểm mạnh của quy trình hiện tại.

## 10. Đánh giá mức độ hoàn thiện

Ở thời điểm hiện tại, repo đã hoàn thành được các khối chính:

- load dữ liệu
- preprocessing
- build graph
- split train/val/test
- baseline MLP
- HGT model
- training và evaluation
- inference top-k
- dashboard demo

Như vậy, repo đã đi được gần hết một pipeline nghiên cứu và demo hoàn chỉnh cho mức đồ án cơ sở.

## 11. Hạn chế hiện tại

Nhóm vẫn còn một số hạn chế cần nói rõ:

- chưa triển khai đầy đủ toàn bộ kiến trúc AMDGT
- similarity branch mới dừng ở mức chuẩn bị dữ liệu
- negative sampling vẫn dùng giả định unknown as negative
- chưa có phân tích sinh học sâu cho từng dự đoán top-k
- dashboard hiện tại là bản demo, chưa phải hệ web production hoàn chỉnh

Những điểm này không làm sai hướng nghiên cứu, nhưng cần nói trung thực để tránh khẳng định quá mức.

## 12. Hướng phát triển tiếp theo

Trong giai đoạn sau, nhóm có thể phát triển theo các hướng sau:

- thêm similarity branch theo tinh thần AMDGT
- làm ablation study, ví dụ có và không có protein, có và không có reverse edges
- thử thêm decoder và chiến lược negative sampling khác
- bổ sung case study sinh học cho các kết quả top-k
- đóng gói backend và giao diện web hoàn chỉnh hơn

## 13. Kết luận

Kết luận quan trọng nhất của báo cáo này là: repo của nhóm không phải là bản sao nguyên xi của paper AMDGT. Cách hiểu đúng hơn là nhóm đang triển khai một biến thể HGT-only trên cùng bài toán và cùng hệ dữ liệu Drug - Protein - Disease.

Nếu cần trả lời ngắn trước giảng viên, nhóm có thể nói theo chuỗi sau: HGT là mô hình nền tảng, AMDGT là công trình tham chiếu trực tiếp, còn repo hiện tại là phiên bản triển khai lõi HGT của nhóm. Cách làm này hợp lý vì vừa bám đúng bài toán, vừa phù hợp mức đồ án cơ sở, đồng thời giúp pipeline rõ ràng, dễ kiểm soát và dễ demo hơn.

## 14. Tài liệu tham khảo chính

- Paper HGT: Heterogeneous Graph Transformer
- Paper AMDGT: Attention-aware multimodal dual graph transformers for disease-drug association prediction
- Repo AMDGT gốc: https://github.com/JK-Liu7/AMDGT
- Repo của nhóm: https://github.com/DucTri2207/do-an-drug-disease-hgt

