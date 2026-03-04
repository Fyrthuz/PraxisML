from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import declared_attr
import re

def camel_to_snake(name: str) -> str:
    name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()

class CustomBase:
    @declared_attr
    def __tablename__(cls):
        # Genera el nombre de la tabla pasando el nombre de la clase de CamelCase a snake_case
        return camel_to_snake(cls.__name__)

Base = declarative_base(cls=CustomBase)
