from typing import List, Set, Optional, Dict, Tuple
from parse_tree import ParseTreeNode, ParseTreeGenerator
from lexical_analyzer import Token

class SymbolRenamer:

    def __init__(self, parse_tree_root: ParseTreeNode):
        self.root = parse_tree_root
        self.scopes: List[Dict[str, Set[ParseTreeNode]]] = []
        self.identifier_nodes: Dict[str, List[ParseTreeNode]] = {}
        self._analyze_scopes()
    
    def _analyze_scopes(self):
        """Analyze the parse tree to identify scopes and variable declarations/uses."""
        # For the simple expression grammar (grammar 1), we don't have scopes
        # For the function grammar (grammar 2), we need to handle function scopes
        self._collect_identifiers(self.root)
    
    def _collect_identifiers(self, node: ParseTreeNode, current_scope: Optional[Dict] = None):
        """Recursively collect all identifier nodes and their relationships."""
        if node.is_terminal and node.symbol == 'IDENTIFIER':
            identifier_value = node.value
            if identifier_value not in self.identifier_nodes:
                self.identifier_nodes[identifier_value] = []
            self.identifier_nodes[identifier_value].append(node)
        
        # Recursively process children
        for child in node.children:
            self._collect_identifiers(child, current_scope)
    
    def find_related_identifiers(self, target_node: ParseTreeNode) -> List[ParseTreeNode]:
        """
        Find all identifier nodes that refer to the same variable as the target node.
        This is a simplified version - in a real implementation, you'd need to
        consider scoping rules, declarations vs uses, etc.
        """
        if not target_node.is_terminal or target_node.symbol != 'IDENTIFIER':
            raise ValueError("Target node must be an IDENTIFIER terminal")
        
        target_value = target_node.value
        
        # For the simple grammar, all identifiers with the same name are related
        # In a more complex grammar, you'd need to analyze scope and declarations
        return self.identifier_nodes.get(target_value, [])
    
    def rename_symbol(self, target_node: ParseTreeNode, new_name: str) -> str:
        """
        Rename all occurrences of the symbol represented by target_node to new_name.
        Returns the modified source code.
        """
        # Find all related nodes
        related_nodes = self.find_related_identifiers(target_node)
        
        # Mark nodes for renaming
        nodes_to_rename = set(related_nodes)
        
        # Generate the modified text
        return self._generate_renamed_text(nodes_to_rename, new_name)
    
    def _generate_renamed_text(self, nodes_to_rename: Set[ParseTreeNode], new_name: str) -> str:
        """Generate the modified source text with renamed symbols."""
        # Collect all terminal nodes in order
        terminals = []
        self._collect_terminals_in_order(self.root, terminals)
        
        # Build the output text
        output_parts = []
        for term_node in terminals:
            if term_node in nodes_to_rename:
                output_parts.append(new_name)
            elif term_node.symbol != 'eps':  # Skip epsilon nodes
                output_parts.append(term_node.value)
        
        return ' '.join(output_parts)
    
    def _collect_terminals_in_order(self, node: ParseTreeNode, terminals: List[ParseTreeNode]):
        """Collect all terminal nodes in left-to-right order."""
        if node.is_terminal:
            if node.symbol != 'eps':  # Don't include epsilon nodes
                terminals.append(node)
        else:
            for child in node.children:
                self._collect_terminals_in_order(child, terminals)
    
    def get_identifier_by_position(self, position: int) -> Optional[ParseTreeNode]:
        """
        Get an identifier node by its position in the source.
        This is useful for interactive selection.
        """
        terminals = []
        self._collect_terminals_in_order(self.root, terminals)
        
        current_pos = 0
        for term_node in terminals:
            if term_node.symbol == 'IDENTIFIER' and current_pos == position:
                return term_node
            if term_node.symbol == 'IDENTIFIER':
                current_pos += 1
        
        return None
    
    def list_all_identifiers(self) -> List[Tuple[str, int]]:
        """List all identifiers with their positions for user selection."""
        identifiers = []
        terminals = []
        self._collect_terminals_in_order(self.root, terminals)
        
        position = 0
        for term_node in terminals:
            if term_node.symbol == 'IDENTIFIER':
                identifiers.append((term_node.value, position))
                position += 1
        
        return identifiers


def demonstrate_symbol_renaming():
    """Demonstration of the symbol renaming functionality."""
    from lexical_analyzer import LexicalAnalyzer
    import sys
    
    # Use the provided grammar file
    grammar_file = sys.argv[1] if len(sys.argv) > 1 else "grammar.ll1"
    
    try:
        # Initialize components
        lexer = LexicalAnalyzer(grammar_file)
        generator = ParseTreeGenerator(lexer)
        
        # Example input
        test_input = "a * b + a"
        print(f"Original expression: {test_input}")
        
        # Generate parse tree
        success, parse_tree_root = generator.generate_tree(test_input)
        
        if success and parse_tree_root:
            # Create renamer
            renamer = SymbolRenamer(parse_tree_root)
            
            # List all identifiers
            identifiers = renamer.list_all_identifiers()
            print("\nIdentifiers found:")
            for value, pos in identifiers:
                print(f"  Position {pos}: '{value}'")
            
            # Get the first 'a' identifier
            target_node = renamer.get_identifier_by_position(0)
            
            if target_node:
                print(f"\nRenaming all occurrences of '{target_node.value}' to 'x'")
                
                # Find related identifiers
                related = renamer.find_related_identifiers(target_node)
                print(f"Found {len(related)} occurrences of '{target_node.value}'")
                
                # Perform renaming
                modified_text = renamer.rename_symbol(target_node, "x")
                print(f"\nModified expression: {modified_text}")
            
        else:
            print("Failed to generate parse tree")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    demonstrate_symbol_renaming()