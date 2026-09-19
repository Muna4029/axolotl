"""
Shared test utilities for prompt strategies tests.
"""

# Shared tool definitions to avoid pylint duplicate-code (R0801) errors
TOOL_MULTIPLES = {
    "type": "function",
    "function": {
        "name": "multiples",
        "description": "Generates a list of all the multiples of a number that are less than a given limit.",
        "parameters": {
            "type": "object",
            "properties": {
                "number": {
                    "type": "integer",
                    "description": "The number to find multiples of.",
                },
                "limit": {
                    "type": "integer",
                    "description": "The upper limit for the multiples.",
                },
            },
            "required": ["number", "limit"],
        },
    },
}
