import os
import re
import ast, astor
from typing import List, Dict, Tuple
from utils import extract_imports_all

class MethodCollect(object):
    def __init__(self):
        self.__method_list = []
        self.__import_list = []
        self.__class_list = []
        
    def __call__(self, **kwargs) -> List[Dict]:
        return self.collect_function(**kwargs)
    
    def reconstruct(self, code : str = None, ast_node : ast.AST = None) -> str:
        """
        Reconstructs the source code from an AST node.
        
        This method can only preserve the docstring, comments will be lost.
        
        Args:
            code (str, optional): The original source code. Defaults to None.
            ast_node (ast.AST, optional): The AST node to reconstruct from. Defaults to None.
        
        Returns:
            str: The reconstructed source code.
        """
        return astor.to_source(ast_node)
    
    def dataload(self, file_path: str = None, code: str = None, tree: ast.AST = None) -> Tuple[str, str, ast.AST]:
        """
        Load Python code or syntax tree from a file or provided code string.

        Parameters:
        - file_path (str, optional): The path to the Python file to load.
        - code (str, optional): The Python code string to parse.
        - tree (ast.AST, optional): The abstract syntax tree to use directly.

        Returns:
        Tuple[str, str, ast.AST]: A tuple containing the file path, the code string, and the abstract syntax tree.

        Raises:
        - FileNotFoundError: If the provided file path is not a valid file.
        - Exception: If there is an error parsing the Python code.
        - ValueError: If no syntax tree is provided or generated.
        """
        # Load Python code/syntax tree
        if file_path is not None:
            if not os.path.isfile(file_path):
                raise FileNotFoundError(f"{file_path} is not a valid file.")
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                code = f.read()
            
        if code is not None:
            try:
                tree = ast.parse(code)
            except SyntaxError:
                code = code.replace("\t", " "*4)
                try:
                    tree = ast.parse(code)
                except Exception as e:
                    if file_path is not None:
                        raise Exception(f"Error parsing python code from file path = {file_path}\n{e}")
                    else:
                        raise Exception(f"Error parsing python code =\n{code}\n{e}")
        if tree is None: raise ValueError("No syntax tree")
        
        return file_path, code, tree
    
    def visit_function(self, tree: ast.AST, code: str, import_context: List[str] = None, class_context: ast.ClassDef = None):
        """
        Visits and extracts information about function definitions in the given AST.

        Args:
            tree (ast.AST): The abstract syntax tree to traverse.
            code (str): The source code from which the AST was generated.
            import_context (List[str], optional): A list of import statements that provide context for the code. Defaults to None.
            class_context (ast.ClassDef, optional): The class definition that provides context for the code. Defaults to None.

        Returns:
            List[Dict]: A list of dictionaries, each containing information about a function definition.
        """
        method_list = [] # Initialize an empty list to store function information
        for node in tree.body: # Traverse only the top-level nodes
            if isinstance(node, ast.FunctionDef):
                method_dict = {
                    'name': node.name,
                    'args': astor.to_source(node.args).strip(),
                    'returns': None if node.returns is None else astor.to_source(node.returns).strip(),
                    'decorator_list': [astor.to_source(d).strip() for d in node.decorator_list],
                    'node': node,
                    'import_context': import_context,
                    'class_context': class_context,
                }
                method_dict['src_code'] = self.reconstruct(code = code, ast_node = node)
                method_list.append(method_dict)
        return method_list
    
    def visit_class(self, tree: ast.AST, code: str):
        """
        Visit and extract information about classes defined in the provided AST.

        This method iterates through the top-level nodes of the AST to find class definitions.
        For each class found, it extracts the class name, base classes, decorators, keywords,
        and the source code of the class definition. The extracted information is stored in a
        dictionary and added to a list.

        Parameters:
        tree (ast.AST): The abstract syntax tree to traverse.
        code (str): The source code from which the AST was generated.

        Returns:
        list: A list of dictionaries, each containing information about a class found in the AST.
        """        
        class_list = []
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                # print(node._fields)
                class_dict = {
                    'name': node.name,
                    'bases': [astor.to_source(b).strip() for b in node.bases],
                    'decorator_list': [astor.to_source(d).strip() for d in node.decorator_list],
                    'keywords': [astor.to_source(k.value).strip() for k in node.keywords],
                    'node' : node,
                }
                class_dict['src_code'] = self.reconstruct(code = code, ast_node = node)        
                class_list.append(class_dict)
        return class_list

    def collect_function(self, debug : bool = False, **kwargs) -> List[Dict]:
        """
        Collects functions and classes from the provided code.

        Args:
            debug (bool, optional): If True, prints debug information. Defaults to False.
            **kwargs: Additional keyword arguments to pass to dataload and collect_imports methods.

        Returns:
            List[Dict]: A list of dictionaries containing information about the collected functions.
        """
        _, code, tree = self.dataload(**kwargs)
        
        # Collect imports
        self.__import_list = self.collect_imports(**kwargs)
        if debug:
            print("[imports]")
            for _i in self.__import_list: print(_i)
            print()
        
        # Traverse the top-level of the syntax tree to collect functions
        self.__method_list = []
        self.__method_list.extend(
            self.visit_function(
                tree = tree, 
                code = code,
                import_context = self.__import_list, 
                class_context = None
            )
        )
        self.__class_list = []
        self.__class_list.extend(self.visit_class(tree = tree, code = code))
        
        # Traverse the top-level functions within top-level classes
        for top_c in self.__class_list:
            self.__method_list.extend(
                self.visit_function(
                    tree = top_c['node'], 
                    code = code,
                    import_context = self.__import_list,
                    class_context = {
                        'name': top_c['name'],
                        'node': top_c['node']
                    }
                )
            )
        
        # Output debug information
        if debug:
            print("[functions]")
            for _i in self.__method_list: print(_i)
            print()
            print("[classes]")
            for _i in self.__class_list: print(_i)
            print()
        return self.__method_list
    
    def collect_imports(self, **kwargs) -> List[Dict]:
        """
        Collects all import statements from the code generated by the dataload method.

        Args:
            **kwargs: Arbitrary keyword arguments passed to the dataload method.

        Returns:
            List[Dict]: A list of dictionaries containing information about the imports.
        """
        _, code, _ = self.dataload(**kwargs)
        import_list = extract_imports_all(code)
        return import_list
        
if __name__ == '__main__':
    test_code = """
from utils import *
from . import *
utils.my_print()
import numpy
import unittest
from .math import (
    sin, 
    cos
)
from ..k.t import utils as u1
from . import utils as u2
import numpy as np
from matplotlib import pyplot as plt

class Meta(type):
    def __new__(cls, name, bases, attrs):
        print(f"Creating class: {name}")
        return super().__new__(cls, name, bases, attrs)

class Base:
    def base_method(self):
        return "Base method"

class MyClass(Base, metaclass=Meta):
    def __init__(self):
        self.my_var = 10    
    def my_function(self):
        print("MyClass() Hello, World!")
    
    def another_function(self):
        print(numpy.array([1, 2, 3]))
        
    class InnerClass():
        def __init__(self):
            print("This is InnerClass")

class MyOtherClass(MyClass):
    pass
    
my_function()
print(np.array([1, 2, 3]))

try:
    import heiheiehei
except ImportError:
    heiheiehei = False
"""
    methodcollect = MethodCollect()
    methodcollect(code = test_code, debug = True)