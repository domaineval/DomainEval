import ast, astor
from typing import List, Dict

class ImportVisitor(ast.NodeVisitor):
    def __init__(self):
        self.imports = []

    def visit_Import(self, node):
        for alias in node.names:
            statement = {"from": None, "import": None, "as": None}
            statement["import"] =  alias.name
            if alias.asname:
                statement["as"] = alias.asname
            self.imports.append(statement)
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        for alias in node.names:
            statement = {"from": None, "import": None, "as": None}
            if node.module: 
                statement["from"] = node.module
            statement["import"] =  alias.name
            if alias.asname:
                statement["as"] = alias.asname
            self.imports.append(statement)
            # Handle from . import *
            source_code = astor.to_source(node).strip()
            # find "from " and " import "
            from_pos = source_code.find('from ')
            import_pos = source_code.find(' import ')
            if from_pos != -1 and import_pos != -1:
                statement["from"] = source_code[from_pos + len('from ') : import_pos]
        self.generic_visit(node)
        
class FunctionVisitor(ast.NodeVisitor):
    def __init__(self):
        self.functions = []

    def visit_FunctionDef(self, node):
        function_info_body = []
        for stmt in node.body:
            function_info_body.append(astor.to_source(stmt).strip())
        function_info = {
            "name": node.name,
            "arguments": [arg.arg for arg in node.args.args],
            "body": function_info_body,
        }
        # Fix docstring indentation
        function_info_body = []
        for line in function_info["body"]:
            if line.startswith("\'\'\'") or line.startswith('\"\"\"'):
                lines = line.split('\n')
                function_info_body.append('\n'.join([l.strip() for l in lines]))
            else:
                function_info_body.append(line)
        function_info["body"] = function_info_body
        self.functions.append(function_info)
        self.generic_visit(node)

class ClassVisitor(ast.NodeVisitor):
    def __init__(self):
        self.classes = []

    def visit_ClassDef(self, node):
        class_info_body = []
        for stmt in node.body:
            class_info_body.append(astor.to_source(stmt).strip())
        class_info = {
            "name": node.name,
            "body": class_info_body,
        }
        self.classes.append(class_info)
        self.generic_visit(node)

def extract_classes_all(code: str) -> list:
    tree = ast.parse(code)
    visitor = ClassVisitor()
    visitor.visit(tree)
    return visitor.classes

def extract_imports_all(code: str) -> List[Dict]:
    try:
        tree = ast.parse(code)
    except Exception as e:
        raise Exception(f"Error parsing python code =\n{code}") from e
    visitor = ImportVisitor()
    visitor.visit(tree)
    # Remove duplicates
    deduplicated_import_list = []
    for _i in visitor.imports:
        if _i not in deduplicated_import_list:
            deduplicated_import_list.append(_i)
    return deduplicated_import_list

def extract_functions_all(code: str) -> list:
    tree = ast.parse(code)
    visitor = FunctionVisitor()
    visitor.visit(tree)
    return visitor.functions

if __name__ == "__main__":
    code = """
from utils import *
utils.my_print()
import numpy
import unittest
from math import (
    sin, 
    cos
)
import numpy as np
from matplotlib import pyplot as plt

class MyClass():
    def __init__(self):
        self.my_var = 10    
    def my_function(self):
        print("MyClass() Hello, World!")
    
    def another_function(self):
        print(numpy.array([1, 2, 3]))
        
    class InnerClass():
        def __init__(self):
            print("This is InnerClass")

class MyOtherClass():
    pass
    
my_function()
print(np.array([1, 2, 3]))
"""
    for item in extract_functions_all(code):
        print(item)
    print("-"*20)
    for item in extract_classes_all(code):
        print(item)
    print("-"*20)    
    for item in extract_imports_all(code):
        print(item)