from pydantic import BaseModel

class Student(BaseModel):
    name: str
    age: Optional[int] = None

new_student = {'age': '32'}

student = Student(**new_student)

print(student)