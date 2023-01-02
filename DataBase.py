import firebase_admin
from firebase_admin import credentials, db


class DataBase:
    def __init__(self, database_name):
        cred = credentials.Certificate("serviceAccountKey.json")
        firebase_admin.initialize_app(cred, {
            "databaseURL": "https://facerec-24eea-default-rtdb.asia-southeast1.firebasedatabase.app"
        })
        self.ref = db.reference("Students")
        self.db_name = database_name

    def add_data(
        self,
        ID,
        realname,
        surname,
        nickname,
        student_id,
        student_class,
        class_number,
        active_days,
        last_checked,
        graph_info,
        **kwargs
    ):
        # update database
        data = {ID: {
            "realname": realname,
            "surname": surname,
            "nickname": nickname,
            "student_id": student_id,
            "class": student_class,
            "class_number": class_number,
            "active_days": active_days,
            "last_checked": last_checked,
            "graph_info": graph_info,
            **kwargs
        }}

        for key, values in data.items():
            self.ref.child(key).set(values)


if __name__ == "__main__":
    database = DataBase("Students")
    database.add_data(
        "test1234",
        realname="Jeerabhat",
        surname="Supapinit",
        nickname="Kaopan",
        student_id="29157",
        student_class="",
        class_number=4,
        active_days=0,
        last_checked="55555555",
        graph_info=""
    )
