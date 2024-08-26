import os
import ast, astor
from tqdm import tqdm
from methodcollect import MethodCollect
from utils import extract_imports_all
from typing import List, Dict, Tuple

class MethodMatch(object):
    """
    Test-Method Matching
    The MethodMatch class is designed to analyze Python repositories 
        and identify relationships between test methods and the methods they call.
    It performs the following tasks:
    1. Identify Python files and test files within a repository.
    2. Parse test files to extract test methods and the methods they call.
    3. Match the called methods to their definitions within the repository.
    4. Provide detailed information about the matched methods, 
        including their source code and location within the repository.
    """
    def __init__(self):
        self.__pyfiles = []
        self.__testfile = []
    
    def getPyFiles(self, repo_path : str = "", debug : bool = False) -> List[str]:
        """Retrieves all Python files within the specified repository path.
        """
        if not os.path.exists(repo_path):
            raise FileNotFoundError(f"{repo_path} not exists") 
        pyfiles = []
        for root, _, files in os.walk(repo_path):
            for file in files:
                if file.endswith('.py'):
                    if debug: print(f"file = {os.path.join(root, file)}")
                    pyfiles.append(os.path.join(root, file))
        return pyfiles
    
    def getTestFiles(self, repo_path : str = "", debug : bool = False) -> List[str]:
        """
        Identifies test files within the repository by checking for filenames containing 'test' or
        methods/classes prefixed with 'Test' or 'test'.
        """
        pyfiles = self.getPyFiles(repo_path = repo_path)
        testfiles = []
        for pyfile in pyfiles:
            flag = False
            if 'test' in os.path.basename(pyfile): flag = True
            else:
                methodcollect = MethodCollect()
                try:
                    _, code, tree = methodcollect.dataload(file_path = pyfile)
                except:
                    print(f"Error parsing python code from file path = {pyfile}")
                    continue
                for item in methodcollect.visit_class(tree = tree, code = code):
                    if 'Test' in item['name'] or 'test' in item['name']:
                        flag = True
                        break
                for item in methodcollect.visit_function(tree = tree, code = code):
                    if 'Test' in item['name'] or 'test' in item['name']:
                        flag = True
                        break
            if flag:
                if debug: print(f"testfile = {pyfile}")
                testfiles.append(pyfile)
        return testfiles
    
    def parseCallName(self, node : ast.AST) -> str:
        """Parses the source code of a function call node to extract the call name.
        """
        return self.keepBeforeFirstLeftParenthesis(s = astor.to_source(node).strip())
    
    def parseTestFiles(self, repo_path : str = "", debug : bool = False) -> List[Dict]:
        """
        Parses test files to extract test methods and the methods they call, 
        returning a list of dictionaries containing detailed information about each test method.
        """
        testfiles = self.getTestFiles(repo_path = repo_path)
        methodcollect = MethodCollect()
        test_method_list = []
        for testfile in testfiles:
            try:
                all_method_list = methodcollect(file_path = testfile)
            except Exception as e:
                print(f"Error: parsing python code from file path = {testfile}\n{e}")
                continue
            for method_dict in all_method_list:
                if 'assert' not in method_dict['src_code']: continue
                test_method_dict = method_dict
                test_method_dict['source_file'] = testfile
                test_method_dict['call'] = []
                method_node = method_dict['node']
                assert isinstance(method_node, ast.FunctionDef)
                for node in ast.walk(method_node):
                    if isinstance(node, ast.Call):
                        test_method_dict['call'].append(self.parseCallName(node = node))
                test_method_dict['call'] = list(set(test_method_dict['call']))                              
                if debug:
                    self.displayTestDict(test_method_dict = test_method_dict)
                test_method_list.append(test_method_dict)
        if debug: print(f"len(test_method_list) = {len(test_method_list)}")
        return test_method_list
    
    def matchImport(self, call_name : str, import_context : List[Dict]) -> List[Dict]:
        """
        Matches a called method name to import statements within the context, 
        returning a list of matching import statements.
        """
        if call_name is None: return []
        # Match import statements
        match_import_list = []
        for _import in import_context:
            # Convert to alias
            use_name = _import['import'] if _import['as'] is None else _import['as']
            if use_name is None: continue
            if use_name == "*": 
                match_import_list.append(_import)
                continue
            if not call_name.startswith(use_name): continue
            if len(call_name) > len(use_name) and not call_name.startswith(use_name + "."): continue
            match_import_list.append(_import)
        return match_import_list
    
    def keepBeforeFirstLeftParenthesis(self, s : str) -> str:
        """Trims a string to keep only the content before the first left parenthesis."""
        index = s.find('(')
        return s[:index] if index != -1 else s
    
    def findPathFromImport(
        self, 
        repo_path : str, cur_path : str, 
        import_statement : Dict, 
        import_target : str, 
        debug : bool = False) -> Tuple[str, List[str]]:
        """Resolves the path of a Python file from an import statement and target method name, 
        returning the path and remaining target method names."""
        
        if not os.path.exists(repo_path):
            raise FileExistsError(f"repo_path {repo_path} not exists")
        if not os.path.exists(cur_path):
            raise FileExistsError(f"cur_path {cur_path} not exists")
        if not cur_path.startswith(repo_path):
            raise ValueError(f"{cur_path} is not in {repo_path}")
        if import_statement is None:
            raise ValueError(f"import_statement is None")
        if not isinstance(import_statement, dict):
            raise ValueError(f"import_statement is not dict")
        
        if debug:
            print("-"*20)
            print(f"repo_path = {repo_path}")
            print(f"cur_path = {cur_path}")
            print(f"import_statement = {import_statement}")
            print(f"import_target = {import_target}")
        
        if import_statement['from'] is not None:
            from_list = import_statement['from'].split('.')
            if debug: print(f"from_list = {from_list}")
            assert len(from_list) > 0
            # Absolute import
            if len(from_list[0]) > 0: cur_path = repo_path
            # Relative import calculation
            for i in range(len(from_list) - 1):
                if len(from_list[i]) == 0: cur_path = os.path.dirname(cur_path)
                else: break
        else: cur_path = repo_path # Absolute import of import type statements
        
        if debug: print(f"cur_path = {cur_path}")
        
        """
        Ensure the rationality of the path. 
        If it is not reasonable, return a null value to indicate a search failure, 
        such as finding an external installation dependency
        """
        if not cur_path.startswith(repo_path): 
            if debug: print(f"{cur_path} is not in {repo_path}")
            return None
        
        import_list = []
        if import_statement['from'] is not None: import_list += import_statement['from'].split('.') 
        if import_statement['import'] is not None: import_list += import_statement['import'].split('.')
        use_name = import_statement['import'] if import_statement['as'] is None else import_statement['as']
        assert use_name is not None
        assert import_target.startswith(use_name) or use_name == "*"
        new_import_target = import_target
        if use_name != "*":
            new_import_target = new_import_target[len(use_name):]
            if new_import_target.startswith('.'): new_import_target = new_import_target[1:]
        if len(new_import_target) > 0 : import_list += new_import_target.split('.')
        if debug: print(f"import_list = {import_list}")
        
        remain_list = [item for item in import_list if len(item) > 0 and item != "*"]
        for index, _import_item in enumerate(import_list):
            if len(_import_item) <= 0: continue # Empty values are processed in relative imports
            if _import_item == "*": continue # Wildcards are processed in relative imports
            try_path = os.path.join(cur_path, _import_item)
            try_remain = []
            if os.path.exists(try_path + ".py"):
            # Find the target Python file
                try_path += ".py"
                remain_list = [item for item in import_list[index+1:] if len(item) > 0 and item != "*"]
                return try_path, try_remain + remain_list
            if not os.path.exists(try_path):
            # Direct access to the directory failed. Instead, parse __init__.py and check the alias
                if debug: print(f"try: {try_path} not exists")
                init_path = os.path.join(cur_path, "__init__.py")
                if not os.path.exists(init_path): 
                    if debug: print(f"init_path {init_path} not exists")
                    return None
                with open(init_path, 'r', encoding = 'utf-8', errors='ignore') as file:
                    content = file.read()
                import_context = extract_imports_all(content)
                match_import_list = self.matchImport(
                    call_name = _import_item, 
                    import_context = import_context
                )
                if len(match_import_list) == 0:
                    if debug: 
                        print(f"import_context = {import_context}")
                        print(f"call_name = {_import_item}")
                        print(f"match_import_list is empty")
                    return None
                for match_import in match_import_list:
                    if debug: 
                        print(f"match_import = {match_import}")
                        print(f"import_statement = {import_statement}")
                        print(f"_import_item = {_import_item}")
                        print(f"import_target = {import_target}")
                    if match_import != import_statement or _import_item != import_target:
                        recursion = self.findPathFromImport(
                            repo_path = repo_path,
                            cur_path = init_path,
                            import_statement = match_import,
                            import_target = _import_item,
                            debug = debug
                        )
                        if recursion is not None:
                            recall_path, remain_list = recursion
                            if os.path.exists(recall_path):
                                try_path = recall_path
                                try_remain = remain_list
        
            # Verify the legality of try_path
            if try_path is None: return None
            if not os.path.exists(try_path): 
                if debug: print(f"try_path {try_path} not exists")
                return None
            if not try_path.startswith(repo_path): 
                if debug: print(f"{try_path} is not in {repo_path}")
                return None
            # Iterative update
            cur_path = try_path
            if debug: print(f"cur: {cur_path}")
            if cur_path.endswith(".py"): 
                # Find the target Python file
                remain_list = [item for item in import_list[index+1:] if len(item) > 0 and item != "*"]
                return cur_path, try_remain + remain_list
        
        # After traversing import_list, the target file is still not found. Return the last path
        return cur_path, [] 
    
    def matchPyfile(self, repo_path : str = "", debug : bool = False) -> List[Dict]:
        """Matches test methods to the Python files containing the methods they call, 
        updating the test method dictionaries with detailed information about the matched files."""
        print("\nPyFile Matching")
        test_method_list = self.parseTestFiles(repo_path = repo_path, debug = False)
        for test_method_dict in tqdm(test_method_list):
            test_call_list = []
            for i in range(len(test_method_dict['call'])):
                test_call = {
                    "callname": test_method_dict['call'][i],
                    "match": {
                        "import": [],
                        "path": None,
                        "target_method": []
                    }
                }

                import_statement_list = self.matchImport(test_call["callname"], test_method_dict['import_context'])
                test_call["match"]["import"] = import_statement_list
                if len(import_statement_list) == 0:
                    # Not imported from external sources, it may be a function in this file
                    pyfile_path = test_method_dict['source_file']
                    test_call["match"]["path"] = pyfile_path
                    if isinstance(test_call["callname"], str):
                        test_call["match"]["target_method"] = test_call["callname"].split(".")
                    else: test_call["match"]["target_method"] = []
                else: 
                    # Cross file import
                    for import_statement in import_statement_list:
                        try:
                            pyfile = self.findPathFromImport(
                                repo_path = repo_path,
                                cur_path = test_method_dict['source_file'],
                                import_statement = import_statement.copy(), # Prevent modification of incoming dictionary copies
                                import_target = test_call["callname"],
                                debug = debug
                            )
                        except Exception as e:
                            print(f"Error: {e}")
                            pyfile = None
                        if debug: print(f"pyfile = {pyfile}")
                        if pyfile is not None:
                            pyfile_path, remain_list = pyfile
                            if pyfile_path.endswith(".py"):
                            # Find one and end it
                                test_call["match"]["path"] = pyfile_path
                                test_call["match"]["target_method"] = remain_list
                                break
                test_call_list.append(test_call)
            test_method_dict['call'] = test_call_list
            if debug: self.displayTestDict(test_method_dict = test_method_dict)
        return test_method_list
    
    def findMethodFromPyfile(self, target_name : str, tree : ast.AST) -> ast.AST:
        """Finds a method or class node within a Python file's abstract syntax tree (AST) by its name."""
        for node in tree.body: # Traverse only the top-level nodes
            if isinstance(node, ast.FunctionDef) or isinstance(node, ast.ClassDef):
                if node.name == target_name:
                    return node
        return None
    
    def matchPymethod(self, repo_path : str = "", debug : bool = False) -> List[Dict]:
        """Matches test methods to the specific methods they call within the repository, 
        updating the test method dictionaries with detailed information about the matched methods."""
        test_method_list = self.matchPyfile(repo_path = repo_path, debug = False)
        print("\nPyMethod Matching")
        methodcollect = MethodCollect()
        for test_method_dict in tqdm(test_method_list):
            for test_call_dict in test_method_dict['call']:
                test_match = test_call_dict['match']
                test_match["method_record_list"] = []
                if test_match["path"] is None: continue
                # Find the target file and match the function
                try:
                    _, _, tree = methodcollect.dataload(file_path = test_match["path"])
                except:
                    print(f"Error parsing python code from file path = {test_match['path']}")
                    continue
                cur_node = tree
                for tgm in test_match["target_method"]:
                    cur_node = self.findMethodFromPyfile(target_name = tgm, tree = cur_node)
                    if cur_node is None: break
                    test_match["method_record_list"].append({
                            "name": tgm,
                            "node": cur_node,
                            "source": astor.to_source(cur_node).strip()
                        }
                    )
                
                test_call_dict['match'] = test_match
            if debug: self.displayTestDict(test_method_dict = test_method_dict)
        return test_method_list
    
    @staticmethod
    def displayTestDict(test_method_dict : Dict):
        """
        Display the contents of a test method dictionary in a structured format.

        This method iterates through the provided dictionary and prints its contents
        in a nested, hierarchical manner, making it easier to visualize the structure
        and data within the dictionary.

        Parameters:
        test_method_dict (Dict): The dictionary containing test method data.
        """
        for key in test_method_dict:
            print(f"[{key}]")
            if key == "call":
                print("[")
                for d in test_method_dict[key]:
                    for k in d:
                        if k == "match":
                            print(f"\t[{k}]")
                            for kk in d[k]:
                                if kk == "method_record_list":
                                    print(f"\t\t[{kk}]\n\t\t\t[")
                                    for dd in d[k][kk]:
                                        for kkk in dd:
                                            if kkk == "source":
                                                if isinstance(dd[kkk], str): 
                                                    lines = dd[kkk].split("\n")
                                                    lines = "\n".join(["\t\t\t\t\t" + line for line in lines])
                                                    print(f"\t\t\t\t[{kkk}] =\n{lines}")
                                                else: print(f"\t\t\t\t[{kkk}] = {dd[kkk]}")
                                            else: print(f"\t\t\t\t[{kkk}] = {dd[kkk]}")
                                    print("\t\t\t]")
                                else: print(f"\t\t[{kk}] = {d[k][kk]}")
                        else: print(f"\t[{k}] {d[k]}")
                    print()
                print("]")
            elif isinstance(test_method_dict[key], list):
                print("[")
                for i in test_method_dict[key]:
                    print(f"\t{i}")
                print("]")
            else: print(f"{test_method_dict[key]}")
        print()

if __name__ == '__main__':

    src_path_all = os.path.join("..", "srcdata")
    # Customize processing scope
    topic_list = ["AI"]
    exclude_repo_list = ["pytorch", "pandas"]
    todo_repo_list = ["nltk"]
    
    methodmatch = MethodMatch()
    for topic in os.listdir(src_path_all):
        
        if topic not in topic_list: continue
        
        print(f"topic = {topic}")
        
        topic_path = os.path.join(src_path_all, topic)
        for repo in os.listdir(topic_path):
            
            if repo in exclude_repo_list : continue
            if repo not in todo_repo_list : continue
            
            repo_path = os.path.join(topic_path, repo)
            print(f"repo_path = {repo_path}")
            methodmatch.matchPymethod(repo_path = repo_path, debug = True)
            
        
        print(f"topic = {topic} is done")