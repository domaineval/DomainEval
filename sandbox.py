from methodmatch import MethodMatch
from methodcollect import MethodCollect
from typing import List, Dict, Union
import os
import ast, astor
import json
import re
from tqdm import tqdm
from utils import check_correctness
import argparse

class DataSandbox(object):
    """Test-Method Matching & Selection"""
    BANNED_KEYWORDS_PATH = "resource/banned_keywords.txt"
        
    def __init__(self):
        """Import banned keywords"""
        with open(self.BANNED_KEYWORDS_PATH, "r", encoding="utf-8") as file:
            self.banned_keywords_list = file.read().splitlines()
    
    @staticmethod        
    def extract_classdef(class_context_dict : Dict, file_path : str):
        """
        Extracts a class definition from the given context dictionary and file path.

        Args:
            class_context_dict (Dict): A dictionary containing the class context.
            file_path (str): The path to the file containing the class definition.

        Returns:
            str: The extracted class definition as a string.
        """
        if class_context_dict is None: return None
        if "node" not in class_context_dict: return None
        class_node = class_context_dict["node"]
        if not isinstance(class_node, ast.ClassDef): return None
        # Extract base classes as a list of strings
        base_str_list = [astor.to_source(base).strip() for base in class_node.bases] if class_node.bases else []
        methodcollect = MethodCollect()
        _, _, tree = methodcollect.dataload(file_path = file_path)
        # Initialize the base test case string
        basetestcase = ""
        # Check for base class implementations within the same file
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                if node.name in base_str_list:
                    basetestcase += astor.to_source(node).strip() + "\n"
        # Check for base classes assigned to variables
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        for base_s in base_str_list:
                            if base_s.startswith(target.id + "."):
                                basetestcase += astor.to_source(node).strip() + "\n"
        # Join base classes into a single string
        bases_str = ', '.join(base_str_list)
        # Return the combined base test case and class definition
        return basetestcase + f"\nclass {class_node.name}({bases_str}):"
        
    
    def ReshapeDataList(self, test_method_list : List[Dict], debug : bool = False) -> List[Dict]:
        """
        Reshapes the list of test methods by filtering and organizing the data.

        Args:
            test_method_list (List[Dict]): A list of dictionaries containing test method data.
            debug (bool, optional): If True, displays the method list after processing. Defaults to False.

        Returns:
            List[Dict]: A list of reshaped and filtered method dictionaries.
        """
        methodlist = []
        for test_index, test_method in enumerate(test_method_list):
            for test_method_call in test_method["call"]:
                # Skip if the target method is empty
                if len(test_method_call["match"]["target_method"]) <= 0: continue
                method_record_list = test_method_call["match"]["method_record_list"]
                if len(method_record_list) <= 0: continue
                method_record = method_record_list[-1]
                
                # Filter out records containing banned keywords
                flag_ban = False
                for banned_keyword in self.banned_keywords_list:
                    if banned_keyword in method_record["source"]:
                        flag_ban = True
                        break
                if flag_ban: continue
                
                # Only keep function definitions
                if not isinstance(method_record["node"], ast.FunctionDef): continue
                
                # Filter out test function calls within the test method itself
                if "assert" in method_record["source"]: continue
                if "test" in method_record["name"]: continue
                
                method_dict = {
                    "method_name" : method_record["name"],
                    "full_method_name" : ".".join(test_method_call["match"]["target_method"]),
                    "method_path" : test_method_call["match"]["path"],
                    "method_context" : method_record_list[0]["source"], # class_context included
                    "test_method_list" : [{
                        "path" : test_method["source_file"], 
                        "code" : test_method["src_code"],
                        "classdef" : self.extract_classdef(test_method["class_context"],test_method["source_file"]),
                        "callname" : test_method_call["callname"],
                        "testname" : [test_method["name"]] if test_method["class_context"] is None else 
                                        [test_method["class_context"]["name"], test_method["name"]]
                    }]
                }
                
                # Check if the method dictionary already exists in the list and merge
                flag_same = False
                for m in methodlist:
                    if m["method_name"] == method_dict["method_name"] and \
                       m["full_method_name"] == method_dict["full_method_name"] and \
                       m["method_path"] == method_dict["method_path"] and \
                       m["method_context"] == method_dict["method_context"] :
                        flag_same = True
                        m["test_method_list"].extend(method_dict["test_method_list"])
                        break
                if flag_same: continue
                methodlist.append(method_dict)
        
        if debug: self.displayMethodList(methodlist = methodlist)    
        return methodlist
    
    @staticmethod
    def displayMethodList(methodlist : List[Dict]):
        """
        Display a list of methods, each represented as a dictionary.

        Parameters:
        methodlist (List[Dict]): A list of dictionaries, each representing a method.

        Each dictionary can contain various keys, such as 'name', 'description', 'code', etc.
        This method will print out the contents of each dictionary in a formatted manner.
        """
        for index, method in enumerate(methodlist):
            print(f"No.{index}")
            for key in method:
                print(f"[{key}]")
                if key == "test_method_list":
                    print("\t[")
                    for test_method in method[key]:
                        for k in test_method:
                            if k == "code":         
                                print("\t\t[code]")
                                lines = test_method['code'].split("\n")
                                print("\n".join(["\t\t\t" + line for line in lines]))
                            else:
                                print(f"\t\t[{k}] {test_method[k]}")
                        print()
                    print("\t]")
                else:
                    lines = method[key].split("\n")
                    print("\n".join(["\t" + line for line in lines]))
            print()
    
    @staticmethod
    def exec_import_context(import_context: List[Dict], debug : bool = False) -> List[str]:
        """
        Filters and constructs executable import statements from a list of dictionaries.

        Args:
            import_context (List[Dict]): A list of dictionaries, each representing an import statement with keys "from", "import", and "as".
            debug (bool, optional): If True, prints failed import statements. Defaults to False.

        Returns:
            List[str]: A list of valid and executable import statements.
        """
        exec_method_import_context = []
        for im_dict in import_context:
            keys = ["from", "import", "as"]
            import_statement = ""
            for key in keys:
                if im_dict[key] is None: continue
                if len(import_statement) > 0: import_statement += " "
                import_statement += f"{key} {im_dict[key]}"
            if len(import_statement) <= 0: continue
            if check_correctness(import_statement)["passed"] == True:
                exec_method_import_context.append(import_statement)
            elif debug: print(f"import failed: {import_statement}")
        return exec_method_import_context
    
    @staticmethod
    def remove_args(line : str, func_name : str) -> str:
        """
        Modifies the function definition to remove arguments from a given line of code.

        Parameters:
        line (str): The line of code containing the function definition.
        func_name (str): The name of the function to modify.

        Returns:
        str: The modified line of code with arguments removed.
        """
        test_funcdef_begin = "def " + func_name + "("
        test_funcdef_end = line[len(test_funcdef_begin):]
        test_funcdef_end = test_funcdef_end.split(")")
        test_funcdef_args = test_funcdef_end[0].lstrip()
        if test_funcdef_args.startswith("self"): test_funcdef_args = "self"
        else: test_funcdef_args = ""
        
        line = test_funcdef_begin + test_funcdef_args
        for i in range(1, len(test_funcdef_end)): 
            line += ")" + test_funcdef_end[i]
        return line
    
    @staticmethod
    def extract_name_from_error_msg(error_msg: Union[str, None]) -> Union[str, None]:
        """
        Extracts the undefined variable name from a resource missing error message.

        Args:
            error_msg (Union[str, None]): The error message string or None.

        Returns:
            Union[str, None]: The extracted variable name or None if no match is found.
        """
        if error_msg is None: return None
        pattern = r"failed: name '(.+?)' is not defined"
        match = re.search(pattern, error_msg)
        if match: return match.group(1)
        else: return None
    
    @staticmethod
    def find_name_from(path : str, name : str) -> Union[str, None]:
        """
        Find the definition of a function, class, or variable from a given file.

        Parameters:
        - path (str): The path to the file to search in.
        - name (str): The name of the function, class, or variable to find.

        Returns:
        - Union[str, None]: The source code of the definition if found, otherwise None.

        Raises:
        - FileExistsError: If the specified file does not exist.
        """
        if not os.path.exists(path): raise FileExistsError(f"Error: {path} not exists")
        methodcollect = MethodCollect()
        try:
            _, _, tree = methodcollect.dataload(file_path = path)
        except: 
            return None
        for node in tree.body:
            subcode = astor.to_source(node).strip()
            for childnode in ast.walk(node):
                if isinstance(childnode, ast.FunctionDef) and childnode.name == name:
                    return subcode
                if isinstance(childnode, ast.ClassDef) and childnode.name == name:
                    return subcode
                if isinstance(childnode, ast.Assign):
                    for target in childnode.targets:
                        if isinstance(target, ast.Name) and target.id == name:
                            return subcode
                if isinstance(childnode, ast.AnnAssign):
                    if isinstance(childnode.target, ast.Name) and childnode.target.id == name:
                        return subcode
        return None
    
    @staticmethod
    def insert_context(code : str, addition : str) -> str:
        """
        Inserts the given addition string into the provided code string.
        
        The addition is placed after the import statements (if any) and before any other statements.
        
        Parameters:
        - code (str): The original code string where the addition will be inserted.
        - addition (str): The string to be inserted into the code.
        
        Returns:
        - str: The modified code string with the addition inserted at the appropriate location.
        
        Raises:
        - ValueError: If the code or addition string is empty.
        """
        if len(code) <= 0: raise ValueError("Error: Code is empty")
        if len(addition) <= 0: raise ValueError("Error: Addition is empty")

        import_lines = []
        other_lines = []
        import_done = False      

        for line in code.splitlines():
            line = line.rstrip()
            if len(line) <= 0: continue
            if (line.startswith("import ") or line.startswith("from ")) and not import_done:
                import_lines.append(line)
            else:
                other_lines.append(line)
                import_done = True
        return "\n".join(import_lines + [addition.strip()] + other_lines)
    
    def mergeDataList(self, bench_method_list : List[Dict]) -> List[Dict]:
        """
        Merges the list of benchmark method dictionaries by combining entries with the same method name and code.

        Args:
            bench_method_list (List[Dict]): A list of dictionaries, each containing details of a benchmark method.

        Returns:
            List[Dict]: A list of merged dictionaries, where each dictionary represents a unique method with its associated test codes.
        """
        merge_bench_method_list = []
        for method_dict in bench_method_list:
            flag_same = False
            for m in merge_bench_method_list:
                if m["method_name"] == method_dict["method_name"] and \
                    m["method_code"] == method_dict["method_code"] :
                    flag_same = True
                    m["test_code_list"].append({
                        "test_code" : method_dict["test_code"],
                        "code_start" : method_dict["code_start"],
                        "test_path" : method_dict["test_path"]
                    })
                    break
            if flag_same: continue
            method_dict["test_code_list"] = [
                {
                    "test_code" : method_dict["test_code"],
                    "code_start" : method_dict["code_start"],
                    "test_path" : method_dict["test_path"],
                }
            ]
            method_dict.pop("test_code", None)
            method_dict.pop("code_start", None)
            method_dict.pop("test_path", None)
            merge_bench_method_list.append(method_dict)
        return merge_bench_method_list
        
    def sandboxingDataList(self, repo : str, test_method_list : List[Dict], debug : bool = False) -> List[Dict]:
        """
        Sandboxes the provided test methods by resolving dependencies and ensuring their executability.
        
        Args:
            repo (str): The repository name.
            test_method_list (List[Dict]): List of dictionaries containing test methods.
            debug (bool, optional): If True, prints debug information. Defaults to False.
        
        Returns:
            List[Dict]: List of dictionaries containing sandboxed test methods.
        """
        # Reshape the data list to a more usable format
        method_list = self.ReshapeDataList(test_method_list = test_method_list, debug = False)
        methodcollect = MethodCollect()
        sandboxing_datalist = []
        if debug: print(f"len(method_list) = {len(method_list)}")
        for method in tqdm(method_list):
            method_path = method["method_path"]
            
            # Collect and refine the import dependencies of the method under test
            method_import_context = methodcollect.collect_imports(file_path = method_path)
            method_import_context = self.exec_import_context(
                import_context = method_import_context, 
                debug = False
            )
            method_code = "\n".join(method_import_context) + "\n"
            method_code += method["method_context"]
            
            # Iterate over all corresponding test methods
            if debug: print(f"len(method['test_method_list']) = {len(method['test_method_list'])}")
            for test_method_dict in method["test_method_list"]:
                test_method_import_context = methodcollect.collect_imports(file_path = test_method_dict["path"])
                test_method_import_context = self.exec_import_context(
                    import_context = test_method_import_context, 
                    debug = False
                )
                
                test_code = ""
                code_start = ""
                # Handle from __future__ import statements
                for import_statement in test_method_import_context:
                    if import_statement.startswith("from __future__ import "):
                        code_start += import_statement + "\n"
                    else: test_code += import_statement + "\n"

                indentation = ""
                if test_method_dict["classdef"] is not None: 
                    test_code += test_method_dict["classdef"] + "\n"
                    indentation = " " * 4
                lines = test_method_dict['code'].split("\n")
                for line in lines:
                    # Remove unnecessary input arguments from the test function
                    if line.startswith("def " + test_method_dict["testname"][-1] + "("):
                        line = self.remove_args(line, test_method_dict["testname"][-1])
                    test_code += indentation + line + "\n"
                test_code += "().".join(test_method_dict["testname"]) + "()" + "\n"
                test_code = test_code.replace(test_method_dict["callname"], method["full_method_name"])
                
                # Check executability and make final attempts to resolve issues
                undefined_name_list = []
                while True:
                    exec_code = code_start + "\n" + method_code + "\n" + test_code
                    # if debug:
                    #     print("-"*10)
                    #     print(f"[exec_code]\n{exec_code}\n")
                    exec_result = check_correctness(exec_code)
                    if exec_result["passed"] == True:
                        sandboxing_datadict = {
                            "method_name" : method["method_name"],
                            "full_method_name" : method["full_method_name"],
                            "method_path" : method["method_path"],
                            "method_code" : method_code,
                            "test_path" : test_method_dict["path"],
                            "test_code" : test_code,
                            "code_start" : code_start,
                        }
                        sandboxing_datalist.append(sandboxing_datadict)
                        break
                    
                    # Attempt to resolve undefined names
                    undefined_name = self.extract_name_from_error_msg(error_msg = exec_result["result"])
                    if undefined_name in undefined_name_list: break
                    if undefined_name is None: break
                    undefined_name_list.append(undefined_name)
                    if undefined_name in method_code:
                        addition = self.find_name_from(path = method["method_path"], name = undefined_name)
                        if addition is None: break
                        method_code = self.insert_context(code = method_code, addition = addition)
                    elif undefined_name in test_code:
                        addition = self.find_name_from(path = test_method_dict["path"], name = undefined_name)
                        if addition is None: break
                        test_code = self.insert_context(code = test_code, addition = addition)
                    else: break
                if debug:
                    print("\n" + "-"*20)
                    print(f"exec_result = {exec_result['result']}")
                    print(f"method_name = {method['method_name']}\t test_name = {test_method_dict['testname'][-1]}")
                    print(f"method_path = {method['method_path']}")
                    print(f"test_path = {test_method_dict['path']}")
                if debug: print(f"exec_code =\n{exec_code}")
        
        sandboxing_datalist = self.mergeDataList(sandboxing_datalist)                
        return sandboxing_datalist

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process some domains.")
    parser.add_argument(
        '--domain', 
        dest='domain_list', 
        action='append',
        default=[],
        help='List of domains to process'
    )
    parser.add_argument('--output_dir', type=str, help='Output Directory')
    parser.add_argument('--srcdata_dir', type=str, help='Source Code Data Directory')
    args = parser.parse_args()
    
    domain_list = args.domain_list
    bench_path = os.path.join(args.output_dir)
    src_path_all = os.path.join(args.srcdata_dir) # Source code repository path
    methodmatch = MethodMatch()
    datasandboxing = DataSandbox()
    if not os.path.exists(bench_path): os.mkdir(bench_path)
    for domain in os.listdir(src_path_all):
        
        if domain not in domain_list: continue
        
        print(f"Handle domain = {domain}")
        
        domain_path = os.path.join(src_path_all, domain)
        for repo in os.listdir(domain_path):
            
            class_path = os.path.join(bench_path, domain)
            if not os.path.exists(class_path): os.mkdir(class_path)
            repo_jsonl_path = os.path.join(class_path, f"{repo}.jsonl")
            if os.path.exists(repo_jsonl_path):
                print(f"repo = {repo} already exists")
                continue
            
            repo_path = os.path.join(domain_path, repo)
            print(f"\nrepo_path = {repo_path}")
            
            test_method_list = methodmatch.matchPymethod(repo_path = repo_path, debug = False)
            print("\nDataList Sandboxing")
            merged_bench_method_list = \
                datasandboxing.sandboxingDataList( 
                    test_method_list = test_method_list,
                    debug = False,
                    repo = repo,
                )
            
            with open(repo_jsonl_path, "w", encoding="utf-8") as f:
                for bench_method in merged_bench_method_list:
                    f.write(json.dumps(bench_method, ensure_ascii = False) + "\n")

        print(f"domain = {domain} is done")