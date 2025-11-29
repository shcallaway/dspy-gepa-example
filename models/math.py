"""Math word problem solver using ReAct."""

import dspy


def calculate(expression: str) -> str:
    """
    Evaluate a mathematical expression safely.

    Supports basic arithmetic operations: +, -, *, /, (), and numbers.

    Args:
        expression: A mathematical expression to evaluate (e.g., "25 * 4 + 10")

    Returns:
        A string containing the result or error message

    Examples:
        calculate("10 + 5") returns "15"
        calculate("(100 - 50) / 2") returns "25.0"
        calculate("100 / 8") returns "12.5"
    """
    try:
        # Sanitize input - only allow numbers, operators, parentheses, spaces
        allowed_chars = set('0123456789+-*/.()\t\n ')
        if not all(c in allowed_chars for c in expression):
            return "Error: Invalid characters in expression. Use only numbers and +, -, *, /, (, )"

        # Evaluate the expression safely
        result = eval(expression, {"__builtins__": {}}, {})

        # Return as string (handle both int and float results)
        if isinstance(result, float) and result.is_integer():
            return str(int(result))
        return str(result)

    except ZeroDivisionError:
        return "Error: Division by zero"
    except SyntaxError:
        return "Error: Invalid mathematical syntax"
    except Exception as e:
        return f"Error: {str(e)}"


class MathWordProblem(dspy.Signature):
    """Solve a math word problem by reasoning and using a calculator tool."""

    problem: str = dspy.InputField(desc="A math word problem in natural language")
    answer: str = dspy.OutputField(desc="The numerical answer to the problem")


class MathSolver(dspy.Module):
    """
    Math word problem solver using ReAct with calculator tool.

    This module takes a word problem as input and uses ReAct (Reasoning and Acting)
    to solve it by breaking down the problem, performing calculations using the
    calculator tool, and generating the final numerical answer.
    """

    def __init__(self):
        super().__init__()
        self.react = dspy.ReAct(
            signature=MathWordProblem,
            tools=[calculate],
            max_iters=2  # Reduced to 2 iterations to minimize LLM calls
        )

    def forward(self, problem):
        """
        Solve a math word problem using ReAct.

        Args:
            problem: The word problem to solve

        Returns:
            Prediction with answer field
        """
        return self.react(problem=problem)
