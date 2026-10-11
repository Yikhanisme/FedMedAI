from docx import Document
from docx.shared import Pt, RGBColor, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

doc = Document()

# ── Margins ───────────────────────────────────────────────────────────────────
s = doc.sections[0]
s.top_margin = s.bottom_margin = Cm(2.5)
s.left_margin = Cm(3.0)
s.right_margin = Cm(2.0)

# ── Helpers ───────────────────────────────────────────────────────────────────
TNR = "Times New Roman"

def h1(text):
    p = doc.add_heading(text, level=1)
    run = p.runs[0]
    run.font.size = Pt(13)
    run.font.name = TNR
    run.font.color.rgb = RGBColor(0x1F, 0x45, 0x7A)

def para(text, bold=False, italic=False, size=12):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.font.name = TNR
    run.font.size = Pt(size)
    run.bold   = bold
    run.italic = italic
    p.paragraph_format.space_after = Pt(4)
    return p

def bullet(plain_text, bold_prefix=None):
    p = doc.add_paragraph(style="List Bullet")
    p.paragraph_format.space_after = Pt(2)
    if bold_prefix:
        r1 = p.add_run(bold_prefix)
        r1.bold = True; r1.font.name = TNR; r1.font.size = Pt(12)
        r2 = p.add_run(plain_text)
        r2.font.name = TNR; r2.font.size = Pt(12)
    else:
        r = p.add_run(plain_text)
        r.font.name = TNR; r.font.size = Pt(12)

def add_table(headers, rows, col_widths=None):
    t = doc.add_table(rows=1 + len(rows), cols=len(headers))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, h in enumerate(headers):
        c = t.rows[0].cells[i]
        c.text = h
        c.paragraphs[0].runs[0].bold = True
        c.paragraphs[0].runs[0].font.size = Pt(11)
        c.paragraphs[0].runs[0].font.name = TNR
        c.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
        shd = OxmlElement("w:shd")
        shd.set(qn("w:val"), "clear")
        shd.set(qn("w:color"), "auto")
        shd.set(qn("w:fill"), "BDD7EE")
        c._tc.get_or_add_tcPr().append(shd)
    for ri, row_data in enumerate(rows):
        for ci, val in enumerate(row_data):
            c = t.rows[ri+1].cells[ci]
            c.text = val
            c.paragraphs[0].runs[0].font.size = Pt(11)
            c.paragraphs[0].runs[0].font.name = TNR
            c.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
    if col_widths:
        for row in t.rows:
            for i, w in enumerate(col_widths):
                row.cells[i].width = Cm(w)

# ══════════════════════════════════════════════════════════════════════════════
#  TITLE
# ══════════════════════════════════════════════════════════════════════════════
title = doc.add_heading("ĐỀ XUẤT HƯỚNG NGHIÊN CỨU", level=0)
title.alignment = WD_ALIGN_PARAGRAPH.CENTER
title.runs[0].font.size = Pt(15)
title.runs[0].font.name = TNR
title.runs[0].font.color.rgb = RGBColor(0x1F, 0x45, 0x7A)

doc.add_paragraph()

# ══════════════════════════════════════════════════════════════════════════════
#  1. PAPER LÀM BASELINE
# ══════════════════════════════════════════════════════════════════════════════
h1("1. Paper tham khảo làm Baseline")

para(
    'Chúng tôi đề xuất sử dụng paper "FedSLD: Federated Learning with Shared Label Distribution" '
    "(Luo & Wu) làm tài liệu tham khảo cơ sở cho thiết kế thí nghiệm. "
    "Paper này được đăng tải tại hội nghị quốc tế uy tín (đã kiểm tra xếp hạng và chỉ số impact), "
    "cung cấp kết quả FedAvg và FedProx trên các bộ dữ liệu y tế làm cơ sở so sánh đáng tin cậy."
)

para(
    "Paper sử dụng hai bộ dữ liệu y tế từ bộ MedMNIST:"
)
bullet("PathMNIST — 9 lớp, hình ảnh mô học ung thư đại trực tràng, 28×28 RGB")
bullet("OrganMNIST (Axial) — 11 lớp, hình ảnh CT bụng, 28×28 grayscale")
doc.add_paragraph()
para(
    "Kết quả FedAvg và FedProx được paper công bố (Practical Non-IID, 12 clients):",
)
add_table(
    headers=["Dataset", "FedAvg (BTA)", "FedProx (BTA)", "FedSLD (BTA)"],
    rows=[
        ["PathMNIST",          "57.38%", "57.72%", "57.90%"],
        ["OrganMNIST (Axial)", "85.69%", "85.54%", "87.37%"],
    ],
    col_widths=[6, 3.5, 3.5, 3.5]
)
doc.add_paragraph()

# ══════════════════════════════════════════════════════════════════════════════
#  2. LÝ DO KHÔNG IMPLEMENT FEDSLD
# ══════════════════════════════════════════════════════════════════════════════
h1("2. Lý Do Không Triển Khai FedSLD")

para("FedSLD yêu cầu mỗi client chia sẻ thống kê phân phối nhãn với Server trước khi huấn luyện. "
     "Chúng tôi không triển khai FedSLD vì hai lý do:")

bullet(
    "Chia sẻ phân phối nhãn có thể tiết lộ đặc điểm dữ liệu bệnh nhân — không phù hợp với "
    "yêu cầu bảo mật trong môi trường y tế thực tế.",
    bold_prefix="Privacy: "
)
bullet(
    "FedSLD can thiệp vào hàm loss phía client. Nghiên cứu của chúng tôi tập trung vào "
    "chiến lược aggregation phía server — đây là hướng nghiên cứu bổ sung, không trùng lặp.",
    bold_prefix="Phạm vi: "
)
doc.add_paragraph()

# ══════════════════════════════════════════════════════════════════════════════
#  3. LÝ DO DÙNG MODEL 20K THAY VÌ MODEL CỦA HỌ
# ══════════════════════════════════════════════════════════════════════════════
h1("3. Lý Do Sử Dụng Mô Hình ~20K Tham Số")

para(
    "FedSLD paper sử dụng CNN 4 lớp (~570K tham số). Chúng tôi sử dụng SimpleCNN "
    "nhẹ hơn (~20K tham số) vì:"
)
bullet(
    "Phù hợp với giới hạn bộ nhớ và tính toán của NVIDIA Jetson Nano/Orin — "
    "đây là mục tiêu triển khai thực tế của đề tài.",
    bold_prefix="Edge deployment: "
)
bullet(
    "Giảm chi phí truyền thông mỗi FL round, phù hợp với môi trường mạng không ổn định.",
    bold_prefix="Communication cost: "
)
bullet(
    "Kết quả so sánh giữa các FL strategies vẫn có ý nghĩa khi cùng sử dụng một model — "
    "xu hướng cải thiện (FedBN > FedProx > FedAvg) không phụ thuộc vào kích thước model.",
    bold_prefix="Tính nhất quán: "
)
doc.add_paragraph()

# ══════════════════════════════════════════════════════════════════════════════
#  4. KẾ HOẠCH THỰC HIỆN
# ══════════════════════════════════════════════════════════════════════════════
h1("4. Kế Hoạch Thực Hiện")

para(
    "Chúng tôi sẽ áp dụng đúng cách chia dữ liệu Practical Non-IID của FedSLD "
    "(12 clients, phân phối 1%/10%/80%) để chạy thử FedAvg và FedProx trên "
    "PathMNIST và OrganMNIST. Mục tiêu là kiểm tra xem xu hướng kết quả "
    "(FedProx ≥ FedAvg, OrganMNIST dễ hơn PathMNIST) có phù hợp với paper không. "
    "Nếu xu hướng đúng, tức là cách hiện thực của chúng tôi là đáng tin cậy."
)
doc.add_paragraph()
para("Các giai đoạn thực hiện theo thứ tự:")

add_table(
    headers=["Bước", "Nội dung", "Mục tiêu"],
    rows=[
        ["1",
         "Chạy FedAvg + FedProx trên PathMNIST & OrganMNIST\n(Practical Non-IID, 12 clients, cách chia như FedSLD)",
         "Verify xu hướng kết quả so với paper"],
        ["2",
         "Implement & chạy giả lập FedBN, FedNova, SCAFFOLD\ntrên cùng 2 dataset trên",
         "So sánh phương pháp đề xuất với baseline"],
        ["3",
         "Triển khai trên NVIDIA Jetson (real federated)\nvới chiến lược cho kết quả tốt nhất ở bước 2",
         "Đánh giá thực tế trên edge device"],
    ],
    col_widths=[1.2, 9, 6]
)
doc.add_paragraph()

# ── Closing note ──────────────────────────────────────────────────────────────
note = doc.add_paragraph(
    "Kính mong Thầy/Cô cho ý kiến về hướng đề xuất trên, "
    "đặc biệt về tính phù hợp của việc sử dụng kết quả FedAvg/FedProx từ paper "
    "làm cơ sở đối chiếu và các phương pháp đề xuất (FedBN, FedNova, SCAFFOLD)."
)
note.alignment = WD_ALIGN_PARAGRAPH.CENTER
note.runs[0].italic = True
note.runs[0].font.size = Pt(11)
note.runs[0].font.name = TNR
note.runs[0].font.color.rgb = RGBColor(0x50, 0x50, 0x50)

doc.save("De_xuat_phuong_phap_FL.docx")
print("Done: De_xuat_phuong_phap_FL.docx")
