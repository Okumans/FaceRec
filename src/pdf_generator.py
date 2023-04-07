from fpdf import FPDF
from typing import Union
import numpy as np
from student_sorter import StudentSorter, Student
from DataBase import DataBase


class PdfTable:
    def __init__(self,
                 header: Union[list[str, ...], tuple[str, ...], None] = None,
                 data: Union[list[tuple[str, ...]], tuple[tuple[str, ...]], None] = None,
                 column_ratio: Union[tuple[Union[int, float], ...], None] = None,
                 column_align: Union[tuple[str, ...], None] = None,
                 output_filename: Union[str, None] = None,
                 **kwargs):

        self.table_header: Union[tuple[str], list] = [] if header is None else list(header)
        self.tabel_data: Union[list[tuple[str]], list] = [] if data is None else list(data)
        self.dimension: int = len(self.table_header)
        self.column_ratio: Union[tuple[int], tuple] = () if column_ratio is None else column_ratio
        self.column_align: Union[tuple[str], tuple] = () if column_align is None else column_align
        self.pdf: FPDF = FPDF()
        self.header_color: tuple[int, int, int] = (191, 191, 191)
        self.paper_size: tuple[int, int] = self.pdf.epw, self.pdf.eph
        self.output_filename = "" if output_filename is None else output_filename
        self.scius_image_path = "scius.png"

        self.__column_size = [((100 / (100 if column_ratio is None else sum(column_ratio)) * i) *
                               self.paper_size[0])/100 for i in column_ratio]
        self.__line_height = self.pdf.font_size * 2

        self.pdf.add_font("THSarabunNew", fname="THSarabunNew.ttf")
        self.pdf.add_font("THSarabunNew", style="B", fname="THSarabunNew Bold.ttf")
        self.pdf.set_font("THSarabunNew", size=16)
        self.student_program = kwargs.get("student_program", "")
        self.student_class = kwargs.get("student_class", "")

    def change_header(self, columns: tuple[str]):
        if not len(columns) == self.dimension:
            raise IndexError("data dimension is not the same.")

        self.table_header = columns

    def add_data(self, columns: tuple[str]):
        if not len(columns) == self.dimension:
            raise IndexError("data dimension is not the same.")

        self.tabel_data.append(columns)

    def add_datas(self, columns: Union[tuple[tuple[str, ...]], list[tuple[str, ...]]]):
        if not (len(np.shape(columns)) == 1 and np.shape(columns)[1] == self.dimension):
            raise IndexError("data dimension is not the same.")

        self.tabel_data.extend(columns)

    def __render_header(self):
        self.pdf.set_fill_color(*self.header_color)
        self.pdf.set_font(style="B")

        for col_ind, col_data in enumerate(self.table_header):
            self.pdf.cell(self.__column_size[col_ind], self.__line_height, col_data, border=1, fill=True, align="C")

        self.pdf.ln(self.__line_height)
        self.pdf.set_font(style="")

    def __render_header_with_information(self, class_="", program=""):
        print(program, class_)
        self.pdf.add_page()
        self.pdf.image(self.scius_image_path, x=8, y=10, w=30, h=30)
        self.pdf.set_font(style='B', size=24)
        self.pdf.set_stretching(90)
        self.pdf.set_x(40)
        self.pdf.write(13, "โรงเรียนสาธิต พิบูลบําเพ็ญ มหาวิทยาลัยบูรพา")
        self.pdf.set_x(-50)
        self.pdf.write(13, f"ใบรายชื่อ {class_}")
        self.pdf.set_font(style="", size=16)

        self.pdf.ln(10)
        self.pdf.set_x(40)
        self.pdf.write(6, "73 ถ.บางแสนล่าง แสนสุข เมือง")
        self.pdf.set_x(-50)
        self.pdf.set_font(style="B", size=18)
        self.pdf.write(13, "มัธยมศึกษาตอนปลาย")
        self.pdf.set_font(style="", size=16)

        self.pdf.ln(5)
        self.pdf.set_x(40)
        self.pdf.write(6, "ชลบุรี 20131 โทรศัพท์: 0-3810-2251 โทรสาร: 0-3839-3238")
        self.pdf.ln(5)
        self.pdf.set_font(style="B", size=16)
        self.pdf.set_x(-50)
        self.pdf.write(6, f"โปรแกรมทั่วไป {program}")

        self.pdf.set_stretching(100)
        self.pdf.ln(35-20)

    def get_pdf(self):
        self.__render_header_with_information(self.student_class, self.student_program)
        self.__render_header()
        for row in self.tabel_data:
            if self.pdf.will_page_break(self.__line_height):
                self.__render_header()
            for column_ind, column_data in enumerate(row):
                self.pdf.cell(self.__column_size[column_ind],
                              self.__line_height, column_data,
                              border=1,
                              align=self.column_align[column_ind]
                              )
            self.pdf.ln(self.__line_height)
        self.pdf.output(self.output_filename)


def get_data_by_class_from_db(db_name, output_path, data_path=None):
    if data_path is not None:
        db = DataBase(db_name, sync_with_offline_db=True)
        db.offline_db_folder_path = data_path
    else:
        db = DataBase(db_name)

    student_sorter = StudentSorter(data_path, db.get_database())
    student_classes = student_sorter.sort_as_classes().id_to_student()
    print(sorted(student_classes))
    for index, student_class in enumerate(sorted(student_classes)):
        students: list = []
        for student in student_classes[student_class]:
            students.append((student.student_class_number,
                            student.student_id,
                            student.realname,
                            student.checked_state))

        PdfTable(header=("เลขที่", "เลขประจำตัวนักเรียน", "ชื่อ - นามสกุล", "สถานะ"),
                 data=students,
                 column_ratio=(0.7, 2.7, 6, 2),
                 column_align=('C', 'C', 'L', 'C'),
                 output_filename=output_path,
                 student_class=student_class,
                 student_program="วิทย์-คณิต"
                 ).get_pdf()


if __name__ == "__main__":
    get_data_by_class_from_db("Students",
                              r"C:\general\Science_project\Science_project_cp39\result")
    data_path = r"C:\general\Science_project\Science_project_cp39\resources_test_3"

    db.offline_db_folder_path = data_path