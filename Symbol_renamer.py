from typing import List, Set, Optional, Dict, Tuple
from parse_tree import ParseTreeNode, ParseTreeGenerator

class ScopeInfo:
    """Information about a scope in the program"""
    def __init__(self, scope_type: str, name: str = None, start_node: ParseTreeNode = None):
        self.scope_type = scope_type  # 'global', 'function', 'block'
        self.name = name  # Function name for function scopes
        self.start_node = start_node  # The node that starts this scope
        self.identifiers: Dict[str, List[ParseTreeNode]] = {}  # identifier_name -> list of nodes
        self.parent_scope: Optional['ScopeInfo'] = None
        self.child_scopes: List['ScopeInfo'] = []
    
    def add_identifier(self, identifier_value: str, node: ParseTreeNode):
        """Add an identifier node to this scope"""
        if identifier_value not in self.identifiers:
            self.identifiers[identifier_value] = []
        self.identifiers[identifier_value].append(node)
    
    def __repr__(self):
        return f"ScopeInfo(type={self.scope_type}, name={self.name}, identifiers={list(self.identifiers.keys())})"

class SymbolRenamer:

    def __init__(self, parse_tree_root: ParseTreeNode):
        self.root = parse_tree_root
        self.global_scope = ScopeInfo('global', 'global')
        self.current_scope = self.global_scope
        self.all_scopes: List[ScopeInfo] = [self.global_scope]
        self.identifier_nodes: Dict[str, List[ParseTreeNode]] = {}  # Global list for backward compatibility
        self._analyze_scopes()
    
    def _analyze_scopes(self):
        """Analyze the parse tree to identify scopes and variable declarations/uses."""
        self._collect_identifiers_with_scope(self.root, self.global_scope)
        
        # Also populate the global identifier_nodes dict for backward compatibility
        for scope in self.all_scopes:
            for identifier_value, nodes in scope.identifiers.items():
                if identifier_value not in self.identifier_nodes:
                    self.identifier_nodes[identifier_value] = []
                self.identifier_nodes[identifier_value].extend(nodes)
    
    def _collect_identifiers_with_scope(self, node: ParseTreeNode, current_scope: ScopeInfo):
        """Recursively collect all identifier nodes with proper scope handling."""
        
        # Check if this node starts a new scope
        new_scope = self._check_for_new_scope(node, current_scope)
        if new_scope:
            current_scope = new_scope
        
        if node.is_terminal:
            # Check for different possible identifier token types
            if (node.symbol == 'IDENTIFIER' or 
                node.symbol == 'ID' or 
                (hasattr(node, 'token') and node.token and 
                 hasattr(node.token.type, 'name') and 
                 node.token.type.name in ['IDENTIFIER', 'ID'])):
                
                identifier_value = node.value
                
                # Add to current scope
                current_scope.add_identifier(identifier_value, node)
                
                print(f"Found identifier: {identifier_value} at node {node} in scope {current_scope.scope_type}:{current_scope.name}")
        
        # Recursively process children
        for child in node.children:
            self._collect_identifiers_with_scope(child, current_scope)
    
    def _check_for_new_scope(self, node: ParseTreeNode, parent_scope: ScopeInfo) -> Optional[ScopeInfo]:
        """Check if the current node starts a new scope and return the new scope if so."""
        
        # Function scope detection
        if (not node.is_terminal and 
            node.symbol == 'Function' and 
            len(node.children) >= 2 and 
            node.children[1].symbol == 'ID'):
            
            function_name = node.children[1].value
            new_scope = ScopeInfo('function', function_name, node)
            new_scope.parent_scope = parent_scope
            parent_scope.child_scopes.append(new_scope)
            self.all_scopes.append(new_scope)
            print(f"Created new function scope: {function_name}")
            return new_scope
        
        # Block scope detection (for if, while, etc.)
        if (not node.is_terminal and 
            node.symbol == 'Block'):
            
            # Determine the type of block based on parent
            block_type = 'block'
            if node.parent and not node.parent.is_terminal:
                if 'if' in str(node.parent.symbol).lower():
                    block_type = 'if_block'
                elif 'while' in str(node.parent.symbol).lower():
                    block_type = 'while_block'
            
            new_scope = ScopeInfo(block_type, f"{block_type}_{len(self.all_scopes)}", node)
            new_scope.parent_scope = parent_scope
            parent_scope.child_scopes.append(new_scope)
            self.all_scopes.append(new_scope)
            print(f"Created new block scope: {new_scope.name}")
            return new_scope
        
        return None
    
    def find_related_identifiers(self, target_node: ParseTreeNode) -> List[ParseTreeNode]:
        """
        Find all identifier nodes that refer to the same variable as the target node,
        respecting scope rules.
        """
        # Check if target node is an identifier
        is_identifier = (target_node.is_terminal and 
                        (target_node.symbol == 'IDENTIFIER' or 
                         target_node.symbol == 'ID' or
                         (hasattr(target_node, 'token') and target_node.token and 
                          hasattr(target_node.token.type, 'name') and 
                          target_node.token.type.name in ['IDENTIFIER', 'ID'])))
        
        if not is_identifier:
            raise ValueError("Target node must be an IDENTIFIER terminal")
        
        target_value = target_node.value
        
        # Find which scope the target node belongs to
        target_scope = self._find_node_scope(target_node)
        
        if not target_scope:
            # Fallback to old behavior if scope detection fails
            return self.identifier_nodes.get(target_value, [])
        
        # Find all related identifiers in the same scope
        related_nodes = []
        
        # Get identifiers from the target scope
        if target_value in target_scope.identifiers:
            related_nodes.extend(target_scope.identifiers[target_value])
        
        # Also check parent scopes for the same identifier (lexical scoping)
        current_scope = target_scope.parent_scope
        while current_scope:
            if target_value in current_scope.identifiers:
                related_nodes.extend(current_scope.identifiers[target_value])
                break  # Stop at first parent scope that has this identifier
            current_scope = current_scope.parent_scope
        
        return related_nodes
    
    def _find_node_scope(self, target_node: ParseTreeNode) -> Optional[ScopeInfo]:
        """Find which scope a given node belongs to."""
        for scope in self.all_scopes:
            for identifier_value, nodes in scope.identifiers.items():
                if target_node in nodes:
                    return scope
        return None
    
    def rename_symbol(self, target_node: ParseTreeNode, new_name: str) -> str:
        """
        Rename all occurrences of the symbol represented by target_node to new_name,
        respecting scope boundaries.
        Returns the modified source code.
        """
        # Find all related nodes in the same scope
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
            is_identifier = (term_node.symbol == 'IDENTIFIER' or 
                           term_node.symbol == 'ID' or
                           (hasattr(term_node, 'token') and term_node.token and 
                            hasattr(term_node.token.type, 'name') and 
                            term_node.token.type.name in ['IDENTIFIER', 'ID']))
            
            if is_identifier and current_pos == position:
                return term_node
            if is_identifier:
                current_pos += 1
        
        return None
    
    def list_all_identifiers(self) -> List[Tuple[str, int]]:
        """List all identifiers with their positions for user selection."""
        identifiers = []
        terminals = []
        self._collect_terminals_in_order(self.root, terminals)
        
        position = 0
        for term_node in terminals:
            is_identifier = (term_node.symbol == 'IDENTIFIER' or 
                           term_node.symbol == 'ID' or
                           (hasattr(term_node, 'token') and term_node.token and 
                            hasattr(term_node.token.type, 'name') and 
                            term_node.token.type.name in ['IDENTIFIER', 'ID']))
            
            if is_identifier:
                identifiers.append((term_node.value, position))
                position += 1
        
        print(f"Debug: Found {len(identifiers)} identifiers total")
        print(f"Debug: Identifier nodes dict has {len(self.identifier_nodes)} entries")
        for key, nodes in self.identifier_nodes.items():
            print(f"  '{key}': {len(nodes)} occurrences")
        
        # Also print scope information
        print(f"Debug: Found {len(self.all_scopes)} scopes:")
        for scope in self.all_scopes:
            print(f"  {scope}")
        
        return identifiers
    
    def get_identifier_options_by_scope(self, identifier_value: str) -> Dict[str, List[ParseTreeNode]]:
        """Get all occurrences of an identifier grouped by scope."""
        scope_options = {}
        
        for scope in self.all_scopes:
            if identifier_value in scope.identifiers:
                scope_key = f"{scope.scope_type}:{scope.name}"
                scope_options[scope_key] = scope.identifiers[identifier_value]
        
        return scope_options
    
    def _get_scope_description(self, scope: ScopeInfo) -> str:
        """Get a human-readable description of a scope."""
        if scope.scope_type == 'function':
            return f"function '{scope.name}'"
        elif scope.scope_type == 'global':
            return "global scope"
        elif scope.scope_type == 'block':
            # Try to determine what kind of block this is
            if scope.start_node and scope.start_node.parent:
                parent = scope.start_node.parent
                # Traverse up to find the control structure
                while parent and not parent.is_terminal:
                    if parent.symbol in ['Statement'] and len(parent.children) > 0:
                        first_child = parent.children[0]
                        if hasattr(first_child, 'symbol'):
                            if first_child.symbol == 'IF' or (hasattr(first_child, 'value') and first_child.value == 'if'):
                                return "if block"
                            elif first_child.symbol == 'WHILE' or (hasattr(first_child, 'value') and first_child.value == 'while'):
                                return "while block"
                    parent = parent.parent
            
            # Check if it's a function body by looking at parent
            if scope.parent_scope and scope.parent_scope.scope_type == 'function':
                return f"main body of function '{scope.parent_scope.name}'"
            
            return f"block (id: {scope.name})"
        else:
            return f"{scope.scope_type} (id: {scope.name})"

    def select_identifier_by_scope(self, identifier_value: str) -> Optional[ParseTreeNode]:
        """Interactive scope selection for identifier renaming."""
        scope_options = self.get_identifier_options_by_scope(identifier_value)
        
        if not scope_options:
            print(f"No occurrences of '{identifier_value}' found in any scope.")
            return None
        
        if len(scope_options) == 1:
            # Only one scope, no need to ask
            scope_key = list(scope_options.keys())[0]
            nodes = scope_options[scope_key]
            # Find the actual scope object for description
            scope_obj = None
            for scope in self.all_scopes:
                if f"{scope.scope_type}:{scope.name}" == scope_key:
                    scope_obj = scope
                    break
            
            scope_desc = self._get_scope_description(scope_obj) if scope_obj else scope_key
            print(f"Found {len(nodes)} occurrence(s) of '{identifier_value}' in {scope_desc}")
            return nodes[0] if nodes else None
        
        # Multiple scopes - let user choose
        print(f"\nFound '{identifier_value}' in multiple scopes:")
        scope_list = list(scope_options.keys())
        
        for i, scope_key in enumerate(scope_list):
            nodes = scope_options[scope_key]
            # Find the actual scope object for better description
            scope_obj = None
            for scope in self.all_scopes:
                if f"{scope.scope_type}:{scope.name}" == scope_key:
                    scope_obj = scope
                    break
            
            scope_desc = self._get_scope_description(scope_obj) if scope_obj else scope_key
            print(f"  {i+1}. {scope_desc} ({len(nodes)} occurrence(s))")
        
        while True:
            try:
                choice = input(f"\nSelect scope (1-{len(scope_list)}) or 'a' for all scopes: ").strip()
                
                if choice.lower() == 'a':
                    # Return all nodes from all scopes (old behavior)
                    all_nodes = []
                    for nodes in scope_options.values():
                        all_nodes.extend(nodes)
                    print(f"Selected all {len(all_nodes)} occurrences across all scopes")
                    return all_nodes[0] if all_nodes else None
                
                choice_num = int(choice)
                if 1 <= choice_num <= len(scope_list):
                    selected_scope = scope_list[choice_num - 1]
                    nodes = scope_options[selected_scope]
                    
                    # Find scope object for description
                    scope_obj = None
                    for scope in self.all_scopes:
                        if f"{scope.scope_type}:{scope.name}" == selected_scope:
                            scope_obj = scope
                            break
                    
                    scope_desc = self._get_scope_description(scope_obj) if scope_obj else selected_scope
                    print(f"Selected {len(nodes)} occurrence(s) from {scope_desc}")
                    return nodes[0] if nodes else None
                else:
                    print(f"Please enter a number between 1 and {len(scope_list)}")
                    
            except ValueError:
                print("Please enter a valid number or 'a'")
            except KeyboardInterrupt:
                print("\nCancelled.")
                return None
    
    def list_identifiers_by_scope(self) -> Dict[str, List[Tuple[str, int]]]:
        """List identifiers grouped by scope for better user interaction."""
        scope_identifiers = {}
        
        for scope in self.all_scopes:
            scope_key = f"{scope.scope_type}:{scope.name}"
            scope_identifiers[scope_key] = []
            
            # Get positions for identifiers in this scope
            terminals = []
            self._collect_terminals_in_order(self.root, terminals)
            
            position = 0
            for term_node in terminals:
                is_identifier = (term_node.symbol == 'IDENTIFIER' or 
                               term_node.symbol == 'ID' or
                               (hasattr(term_node, 'token') and term_node.token and 
                                hasattr(term_node.token.type, 'name') and 
                                term_node.token.type.name in ['IDENTIFIER', 'ID']))
                
                if is_identifier:
                    # Check if this identifier belongs to current scope
                    if term_node.value in scope.identifiers and term_node in scope.identifiers[term_node.value]:
                        scope_identifiers[scope_key].append((term_node.value, position))
                    position += 1
        
        return scope_identifiers


def demonstrate_symbol_renaming():
    """Demonstration of the symbol renaming functionality."""
    from lexical_analyzer import LexicalAnalyzer
    import sys
    
    # Use the provided grammar file
    grammar_file = "Grammers/" + sys.argv[1] if len(sys.argv) > 1 else "grammar.ll1"
    
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
            
            # Show scope-based identifiers
            scope_identifiers = renamer.list_identifiers_by_scope()
            print("\nIdentifiers by scope:")
            for scope_key, scope_ids in scope_identifiers.items():
                print(f"  {scope_key}: {scope_ids}")
            
            # Get the first 'a' identifier
            target_node = renamer.get_identifier_by_position(0)
            
            if target_node:
                print(f"\nRenaming all occurrences of '{target_node.value}' to 'x'")
                
                # Find related identifiers
                related = renamer.find_related_identifiers(target_node)
                print(f"Found {len(related)} occurrences of '{target_node.value}' in the same scope")
                
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