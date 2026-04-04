from pydantic import BaseModel, EmailStr

class CreateUserRequest(BaseModel):
    email: EmailStr
    password: str
    full_name: str | None = None
    role: str = "viewer"

try:
    req = CreateUserRequest(**{"email":"test@example.com", "password":"pw", "full_name":"", "role":"viewer"})
    print("Success:", req)
except Exception as e:
    import traceback; traceback.print_exc()
