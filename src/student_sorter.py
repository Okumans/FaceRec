from __future__ import annotations
from typing import Union, Any, Callable
from src.attendant_graph import Arrange, AttendantGraph
from datetime import datetime
from src.DataBase import DataBase


class Student:
    CHECKED = "CHECKED_LATE"
    CHECKED_LATE = "CHECKED_LATE"
    NOT_CHECKED = "NOT_CHECKED"
    UNKNOWN = "UNKNOWN"
    KNOWN = "KNOWN"

    FIRSTNAME = "realname"
    LASTNAME = "surname"
    NICKNAME = "nickname"
    STUDENT_ID = "student_id"
    STUDENT_CLASS = "student_class"
    STUDENT_CLASS_NUMBER = "class_number"
    LAST_CHECKED = "last_checked"
    STUDENT_ATTENDANT_GRAPH_DATA = "graph_info"

    def __init__(self,
                 firstname: str = None,
                 lastname: str = None,
                 nickname: str = None,
                 student_id: int = None,
                 student_class: str = None,
                 student_class_number: int = None,
                 last_checked: datetime = None,
                 student_attendant_graph_data: list[datetime, ...] = None,
                 late_checked_hour_minute: Union[tuple[int, int], None] = None):
        self.firstname: str = "" if firstname is None else firstname
        self.lastname: str = "" if lastname is None else lastname
        self.nickname: str = "" if nickname is None else nickname
        self.student_id: int = 0 if student_id is None else student_id
        self.student_class: str = "" if student_class is None else student_class
        self.student_class_number = 0 if student_class_number is None else student_class_number
        self.active_days = 0
        self.last_checked: datetime = datetime.fromtimestamp(0) if last_checked is None else last_checked
        self._student_attendant_graph_data: list[datetime, ...] = [] if student_attendant_graph_data is None else \
            student_attendant_graph_data
        self.late_checked: Callable[[datetime, int, int], datetime] = \
            lambda consider_datetime: datetime(year=consider_datetime.year,
                                               month=consider_datetime.month,
                                               day=consider_datetime.day,
                                               hour=late_checked_hour_minute[0]
                                               if late_checked_hour_minute is not None else 0,
                                               minute=late_checked_hour_minute[1]
                                               if late_checked_hour_minute is not None else 0
                                               )
        self.__calculate_active_days()

    @staticmethod
    def load_students_from_dict_db(students: dict) -> dict[str, Student]:
        returned_student: dict[str, Student] = {}
        student: str

        for student in students:
            if type(students[student]) is dict:
                returned_student[student] = Student().load_from_dict(students[student])

        return returned_student

    def __calculate_active_days(self):
        self.active_days = len(Arrange(self._student_attendant_graph_data, datetime.now()).arrange_in_all_as_day())

    def __repr__(self):
        return f"Student(" \
               f"firstname: '{self.firstname}', " \
               f"lastname: '{self.lastname}', " \
               f"nickname: '{self.nickname}', " \
               f"student_id: {self.student_id}, " \
               f"student_class: '{self.student_class}', " \
               f"student_class_number: {self.student_class_number}, " \
               f"active_days: {self.active_days}, " \
               f"last_checked: '{self.last_checked}')" \


    @property
    def realname(self):
        return self.firstname + (" " + self.lastname if self.lastname != '' else "")

    @realname.setter
    def realname(self, name: str):
        raw_split_name = name.split()
        self.firstname = raw_split_name[0] if raw_split_name else ""
        self.lastname = " ".join(raw_split_name[1:]) if len(raw_split_name) > 1 else ""

    @property
    def checked_state(self):
        state = self.NOT_CHECKED
        consider_datetime = self.last_checked
        if Arrange.same_day(datetime.now(), consider_datetime):
            state = self.CHECKED
            if consider_datetime > self.late_checked(consider_datetime):
                state = self.CHECKED_LATE
        return state

    @property
    def student_attendant_graph_data(self):
        return self._student_attendant_graph_data

    @student_attendant_graph_data.setter
    def student_attendant_graph_data(self, value):
        self._student_attendant_graph_data = value
        self.__calculate_active_days()

    def load_from_db(self, db: DataBase, IDD: str):
        student: dict = db.get_data(IDD)
        self.firstname = student[self.FIRSTNAME]
        self.lastname = student[self.LASTNAME]
        self.nickname = student[self.NICKNAME]
        self.student_id = student[self.STUDENT_ID]
        self.student_class = student[self.STUDENT_CLASS]
        self.student_class_number = student[self.STUDENT_CLASS_NUMBER]
        self.last_checked = datetime.fromtimestamp(student[self.LAST_CHECKED])
        self._student_attendant_graph_data = AttendantGraph().load_floats(
            student.get(self.STUDENT_ATTENDANT_GRAPH_DATA, [])).dates
        self.__calculate_active_days()
        return self

    def load_from_dict(self, student: dict):
        self.firstname = student[self.FIRSTNAME]
        self.lastname = student[self.LASTNAME]
        self.nickname = student[self.NICKNAME]
        self.student_id = student[self.STUDENT_ID]
        self.student_class = student[self.STUDENT_CLASS]
        self.student_class_number = student[self.STUDENT_CLASS_NUMBER]
        self.last_checked = datetime.fromtimestamp(student[self.LAST_CHECKED])
        self._student_attendant_graph_data = AttendantGraph().load_floats(
            student.get(self.STUDENT_ATTENDANT_GRAPH_DATA, [])).dates
        self.__calculate_active_days()
        return self


class StudentSorter:
    CHECKED = "CHECKED_LATE"
    CHECKED_LATE = "CHECKED_LATE"
    NOT_CHECKED = "NOT_CHECKED"
    UNKNOWN = "UNKNOWN"
    KNOWN = "KNOWN"

    FIRSTNAME = "realname"
    LASTNAME = "surname"
    NICKNAME = "nickname"
    STUDENT_ID = "student_id"
    STUDENT_CLASS = "student_class"
    STUDENT_CLASS_NUMBER = "class_number"
    LAST_CHECKED = "last_checked"
    STUDENT_ATTENDANT_GRAPH_DATA = "graph_info"

    def __init__(self,
                 data_path: str,
                 data: Union[dict[Any, ...], None] = None,
                 late_checked_hour_minute: Union[tuple[int, int], None] = None):

        self.db: DataBase = DataBase("Students", sync_with_offline_db=True)
        self.db.offline_db_folder_path = data_path
        self.data: dict[str, Any] = self.db.get_database() if data is None else data
        self.late_checked: Callable[[datetime, int, int], datetime] = \
            lambda consider_datetime: datetime(year=consider_datetime.year,
                                               month=consider_datetime.month,
                                               day=consider_datetime.day,
                                               hour=late_checked_hour_minute[0]
                                               if late_checked_hour_minute is not None else 0,
                                               minute=late_checked_hour_minute[1]
                                               if late_checked_hour_minute is not None else 0
                                               )

        self.__data: dict[str, list[str, ...]] = {}
        self.__classes: dict[str, list[str, ...]] = {}
        self.__id_state: dict[str, list[str, ...]] = {}
        self.__states: dict[str, list[str, ...]] = {}

        del self.data["last_update"]

    def get(self):
        return self.__data

    def id_to_(self, key: str, data: dict[str, list[str, ...]] = None) -> dict[str, list[str, ...]]:
        data = self.__data if data is None else data
        temp_data: dict[str, list[str, ...]] = {}
        for sub_data in data:
            temp_data[sub_data] = []
            for datum in data[sub_data]:
                temp_data[sub_data].append(self.data[datum].get(key))
        return temp_data

    def id_to_student(self, data: dict[str, list[str, ...]] = None) -> dict[str, list[Student, ...]]:
        data = self.__data if data is None else data
        temp_data: dict[str, list[Student, ...]] = {}
        for sub_data in data:
            temp_data[sub_data] = []
            for datum in data[sub_data]:
                temp_data[sub_data].append(Student().load_from_dict(self.data[datum]))
        return temp_data

    def sort_as_classes(self) -> StudentSorter:
        self.__classes = {}
        for student in self.data:
            if self.data[student][self.STUDENT_CLASS] == "":
                self.data[student][self.STUDENT_CLASS] = self.UNKNOWN

            if self.__classes.get(self.data[student][self.STUDENT_CLASS]) is None:
                self.__classes[self.data[student][self.STUDENT_CLASS]] = []

            self.__classes[self.data[student][self.STUDENT_CLASS]].append(student)
            self.__classes[self.data[student][self.STUDENT_CLASS]].sort(
                key=lambda i: int(self.data[i][self.STUDENT_CLASS_NUMBER]) if
                self.data[i][self.STUDENT_CLASS_NUMBER] != 0 else 99999)

        self.__data = self.__classes
        return self

    def sort_as_unknown(self):
        self.__id_state = {self.KNOWN: [], self.UNKNOWN: []}
        for student in self.data:
            if student.startswith("unknown:"):
                self.__id_state[self.UNKNOWN].append(student)
            else:
                self.__id_state[self.KNOWN].append(student)
        self.__data = self.__id_state
        return self

    def sort_as_state(self) -> StudentSorter:
        self.__states = {self.CHECKED: [], self.CHECKED_LATE: [], self.NOT_CHECKED: []}
        for student in self.data:
            state = self.NOT_CHECKED
            consider_datetime = datetime.fromtimestamp(self.data[student][self.LAST_CHECKED])
            if Arrange.same_day(datetime.now(), consider_datetime):
                state = self.CHECKED
                if consider_datetime > self.late_checked(consider_datetime):
                    state = self.CHECKED_LATE

            self.__states[state].append(student)
            self.__states[state].sort(key=lambda i: int(self.data[i][self.STUDENT_ID])
                                      if self.data[i][self.STUDENT_ID] != 0 else 9999999)
        self.__data = self.__states
        return self


if __name__ == "__main__":
    a = StudentSorter(r"C:\general\Science_project\Science_project_cp39\resources_test_3")
    dbb = DataBase("Students")

    print(a.sort_as_classes().id_to_student())
