import ast
import autopep8
import subprocess
import bigO
from pylint.lint import Run
from pylint import epylint as lint


def analyze_code_structure(code):
    """
    Analyze the code structure using AST (Abstract Syntax Tree).
    """
    try:
        tree = ast.parse(code)
        num_functions = sum(isinstance(node, ast.FunctionDef) for node in ast.walk(tree))
        num_loops = sum(isinstance(node, (ast.For, ast.While)) for node in ast.walk(tree))
        num_conditions = sum(isinstance(node, ast.If) for node in ast.walk(tree))
        
        return {
            "num_functions": num_functions,
            "num_loops": num_loops,
            "num_conditions": num_conditions
        }
    except SyntaxError as e:
        return {"error": str(e)}


def evaluate_algorithm_complexity(code, example_input):
    """
    Analyze algorithm efficiency (Big-O Complexity).
    """
    exec_namespace = {}
    try:
        exec(code, exec_namespace)
        test_function = [f for f in exec_namespace.values() if callable(f)][-1]  # Get the last defined function
        bigo = bigO.BigO()
        complexity = bigo.test(test_function, "random")
        return str(complexity)
    except Exception as e:
        return str(e)


def run_test_cases(code, test_cases):
    """
    Check correctness by executing test cases.
    """
    results = []
    exec_namespace = {}
    
    try:
        exec(code, exec_namespace)
        test_function = list(exec_namespace.values())[-1]
        
        for inp, expected in test_cases:
            try:
                output = test_function(*inp)
                results.append({
                    "input": inp,
                    "expected": expected,
                    "output": output,
                    "status": "Pass" if output == expected else "Fail"
                })
            except Exception as e:
                results.append({"input": inp, "error": str(e), "status": "Error"})
        
        return results
    except Exception as e:
        return [{"error": str(e), "status": "Error"}]


def analyze_code_quality(code):
    """
    Check code quality using Pylint.
    """
    with open("temp_code.py", "w") as f:
        f.write(code)
    
    pylint_stdout, pylint_stderr = lint.py_run("temp_code.py", return_std=True)
    return pylint_stdout.getvalue()  # Return the text output directly


def generate_static_feedback(code, test_results, complexity):
    """
    Generate predefined rule-based feedback instead of OpenAI.
    """
    feedback = []

    # Check for recursion
    if "def " in code and "return" in code and "(" in code and ")" in code:
        if any(line.strip().startswith("return") and "(" in line for line in code.split("\n")):
            feedback.append("Your function uses recursion. Ensure it handles large inputs efficiently.")

    # Check for loops
    if "for " in code or "while " in code:
        feedback.append("Your code contains loops. Consider optimizing them if they are nested.")

    # Complexity Analysis Feedback
    if "O(N^2)" in complexity:
        feedback.append("Your algorithm has quadratic time complexity (O(N^2)). Try optimizing it.")
    elif "O(N)" in complexity:
        feedback.append("Your algorithm has linear time complexity (O(N)), which is efficient for large inputs.")
    
    # Test case feedback
    if any(res["status"] == "Fail" for res in test_results):
        feedback.append("Some test cases failed. Double-check edge cases and logic errors.")

    if not feedback:
        feedback.append("Code looks good! No major issues detected.")

    return "\n".join(feedback)


def review_code(code, test_cases, example_input):
    """
    Comprehensive rule-based code review (without OpenAI).
    """
    structure = analyze_code_structure(code)
    complexity = evaluate_algorithm_complexity(code, example_input)
    test_results = run_test_cases(code, test_cases)
    quality_report = analyze_code_quality(code)
    static_feedback = generate_static_feedback(code, test_results, complexity)
    
    return {
        "structure_analysis": structure,
        "algorithm_complexity": complexity,
        "test_case_results": test_results,
        "code_quality_report": quality_report,
        "feedback": static_feedback
    }


# Example Usage
if __name__ == "__main__":
    candidate_code = """
def factorial(n):
    if n == 0:
        return 1
    return n * factorial(n-1)
"""

    test_cases = [((5,), 120), ((3,), 6), ((0,), 1)]
    example_input = lambda n: (n,)

    result = review_code(candidate_code, test_cases, example_input)
    print(result)
