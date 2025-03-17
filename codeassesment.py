import ast
import openai
import autopep8
import subprocess
import bigO
from pylint import epylint as lint

# OpenAI API Key (Replace with actual key)
openai.api_key = "sk-...YDoA"

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
        test_function = list(exec_namespace.values())[-1]  # Get the last function defined
        bigo = bigO.BigO()
        complexity = bigo.test(test_function, example_input)
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
    return pylint_stdout.getvalue()

def generate_ai_feedback(code, test_results, complexity):
    """
    Use GPT-4 to provide AI-powered feedback on code logic and improvements.
    """
    prompt = f"""
    Given the following Python code:

    {code}

    Test Results: {test_results}
    Algorithm Complexity: {complexity}

    Analyze the correctness, efficiency, and readability of the code.
    Provide suggestions for improvement and best practices.
    """
    
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "system", "content": "You are an AI coding assistant that reviews code."},
                  {"role": "user", "content": prompt}]
    )
    return response["choices"][0]["message"]["content"]

def review_code(code, test_cases, example_input):
    """
    Comprehensive AI-powered code review.
    """
    structure = analyze_code_structure(code)
    complexity = evaluate_algorithm_complexity(code, example_input)
    test_results = run_test_cases(code, test_cases)
    quality_report = analyze_code_quality(code)
    ai_feedback = generate_ai_feedback(code, test_results, complexity)
    
    return {
        "structure_analysis": structure,
        "algorithm_complexity": complexity,
        "test_case_results": test_results,
        "code_quality_report": quality_report,
        "ai_feedback": ai_feedback
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