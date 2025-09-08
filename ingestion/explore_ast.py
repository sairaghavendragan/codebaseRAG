import sys
from tree_sitter import Parser
import tree_sitter_language_pack

def print_ast(node, indent=""):
    """Recursively prints the AST of a node."""
    # The node's type (e.g., 'function_declaration', 'identifier')
    node_type = node.type
    
    # The actual text content of the node
    node_text = node.text.decode('utf-8').split('\n')[0] # Show first line only for brevity

    print(f"{indent}- Type: '{node_type}', Text: '{node_text}...'")
    
    for child in node.children:
        print_ast(child, indent + "  ")

def explore(language: str, code_snippet: str):
    """Parses a code snippet for a given language and prints its AST."""
    print(f"\n--- Exploring AST for language: {language} ---")
    try:
        parser = tree_sitter_language_pack.get_parser(language)
        tree = parser.parse(bytes(code_snippet, "utf8"))
        
        print_ast(tree.root_node)
        
    except Exception as e:
        print(f"Could not explore language '{language}'. Error: {e}")

if __name__ == "__main__":
    # --- Example for JavaScript ---
    js_code = """
import { something } from './utils';

const PI = 3.14;

function regularFunction() {
  return 1;
}

const arrowFunction = () => {
  return 2;
};

class MyClass {
  constructor() {
    this.value = 3;
  }

  myMethod() {
    return this.value;
  }
}
"""
    explore("javascript", js_code)

    # --- You can add more examples for other languages ---
    # python_code = "..."
    # explore("python", python_code)

'''   NODE_TYPE_TO_CHUNK_TYPE: Dict[str, str] = {
        'function_definition': 'function', 'method_definition': 'method', 'class_definition': 'class',
        'object_declaration': 'object', 'class_specifier': 'class', 'namespace_declaration': 'module',
        'function_signature': 'function', 'method_signature': 'method', 'interface_declaration': 'interface',
        'function_declaration': 'function', 'class_declaration': 'class', 'method_declaration': 'method',
        "class": "class", "singleton_class": "class", "method": "method", "singleton_method": "method",
        "alias": "method", "module": "module", "function_item": "function",
    }
    NODE_TYPE_TO_NAME_FIELD: Dict[str, str] = {
        'function_definition': 'name', 'method_definition': 'name', 'class_definition': 'name',
        'object_declaration': 'name', 'class_specifier': 'name', 'namespace_declaration': 'name',
        'function_signature': 'name', 'method_signature': 'name', 'interface_declaration': 'name',
        'function_declaration': 'name', 'class_declaration': 'name', 'method_declaration': 'name',
        "class": "name", "singleton_class": "name", "method": "name", "singleton_method": "name",
        "alias": "name", "module": "name", "function_item": "name",
    }
    IDENTIFIER_NODE_TYPES = {'identifier', 'name', 'shorthand_property_identifier_pattern'}'''