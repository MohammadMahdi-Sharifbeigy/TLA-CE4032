from typing import List, Optional, Tuple
import graphviz
from lexical_analyzer import Token, LexicalAnalyzer, TokenType
from parse_table import ParserTables

class ParseTreeNode:
    """Node in the parse tree"""
    def __init__(self, symbol: str, value: str = None, is_terminal: bool = False):
        self.symbol = symbol
        self.value = value if value is not None else symbol
        self.is_terminal = is_terminal
        self.children: List['ParseTreeNode'] = []
        self.parent: Optional['ParseTreeNode'] = None
        self.token: Optional[Token] = None  # For terminals, store the token
        self.node_id = None  # Unique ID for graphviz
        
    def add_child(self, child: 'ParseTreeNode'):
        """Add a child node"""
        child.parent = self
        self.children.append(child)
        
    def __repr__(self):
        return f"ParseTreeNode({self.symbol}, terminal={self.is_terminal})"

class ParseTreeGenerator:
    """Generate parse tree and derivation tree with visualization"""
    def __init__(self, lexical_analyzer: LexicalAnalyzer):
        self.lexer = lexical_analyzer
        self.parser_tables = ParserTables(lexical_analyzer)
        self.node_counter = 0
        
    def _get_node_id(self) -> str:
        """Generate unique node ID for graphviz"""
        self.node_counter += 1
        return f"node_{self.node_counter}"
        
    def parse_with_tree(self, input_text: str) -> Tuple[bool, Optional[ParseTreeNode]]:
        """Parse input and generate parse tree"""
        # Tokenize the input
        tokens = self.lexer.tokenize(input_text)
        
        # Initialize the parsing stack with parse tree nodes
        root = ParseTreeNode(self.lexer.start_symbol, is_terminal=False)
        stack = [('$', None), (self.lexer.start_symbol, root)]
        
        # Initialize the input token index
        token_index = 0
        current_token = tokens[token_index]
        
        # Parsing steps for debugging
        print(f"{'Step':<5} {'Stack':<30} {'Input':<20} {'Action':<30}")
        print("-" * 85)
        
        step = 1
        while len(stack) > 1:  # Continue while we have more than just $
            # Get the top of the stack
            top_symbol, top_node = stack[-1]
            
            # Print current state
            stack_str = ' '.join([s[0] for s in stack])
            remaining_tokens = tokens[token_index:] if token_index < len(tokens) else []
            input_str = ' '.join([t.value if t.type != self.lexer.EOF else '$' for t in remaining_tokens])
            
            # If the top is the end marker
            if top_symbol == '$':
                if current_token.type == self.lexer.EOF:
                    print(f"{step:<5} {stack_str:<30} {input_str:<20} Accept")
                    return True, root
                else:
                    print(f"{step:<5} {stack_str:<30} {input_str:<20} Error: Expected EOF")
                    return False, None
            
            # If the top is a terminal
            if top_symbol in self.parser_tables.terminals:
                # If it matches the current input token
                if (current_token.type.name == top_symbol and current_token.type != self.lexer.EOF) or \
                   (current_token.type == self.lexer.EOF and top_symbol == '$'):
                    action = f"Match {top_symbol}"
                    
                    # Store the token in the node
                    if top_node:
                        top_node.token = current_token
                        top_node.value = current_token.value
                    
                    stack.pop()
                    token_index += 1
                    if token_index < len(tokens):
                        current_token = tokens[token_index]
                else:
                    print(f"{step:<5} {stack_str:<30} {input_str:<20} Error: Expected {top_symbol}")
                    return False, None
            
            # If the top is a non-terminal
            elif top_symbol in self.parser_tables.non_terminals:
                # Get the production rule from the parse table
                terminal = current_token.type.name if current_token.type != self.lexer.EOF else '$'
                production = self.parser_tables.get_parse_table_entry(top_symbol, terminal)
                
                if production is None:
                    print(f"{step:<5} {stack_str:<30} {input_str:<20} Error: No production")
                    return False, None
                
                action = f"Expand {top_symbol} -> {production}"
                
                # Pop the non-terminal
                stack.pop()
                
                # Create child nodes for the production
                if production == 'eps':
                    # Add epsilon child
                    eps_node = ParseTreeNode('ε', 'ε', is_terminal=True)
                    top_node.add_child(eps_node)
                else:
                    # Push the production in reverse order
                    symbols = production.split()
                    child_nodes = []
                    
                    for symbol in symbols:
                        is_term = symbol in self.parser_tables.terminals
                        child = ParseTreeNode(symbol, is_terminal=is_term)
                        top_node.add_child(child)
                        child_nodes.append(child)
                    
                    # Push in reverse order
                    for i in range(len(symbols) - 1, -1, -1):
                        stack.append((symbols[i], child_nodes[i]))
            
            else:
                print(f"{step:<5} {stack_str:<30} {input_str:<20} Error: Unknown symbol")
                return False, None
            
            print(f"{step:<5} {stack_str:<30} {input_str:<20} {action}")
            step += 1
        
        return True, root
    
    def visualize_parse_tree(self, root: ParseTreeNode, filename: str = "parse_tree"):
        """Create a visual representation of the parse tree using graphviz"""
        import os
        
        # Ensure Parse_Tree directory exists
        os.makedirs("Parse_Tree", exist_ok=True)
        
        dot = graphviz.Digraph(comment='Parse Tree')
        dot.attr(rankdir='TB')  # Top to bottom
        
        # Configure node and edge styles
        dot.attr('node', shape='ellipse', style='filled', fillcolor='lightblue')
        dot.attr('edge', arrowsize='0.7')
        
        def add_nodes(node: ParseTreeNode):
            """Recursively add nodes to the graph"""
            # Create unique node ID
            node.node_id = self._get_node_id()
            
            # Determine node style
            if node.is_terminal:
                if node.symbol == 'ε':
                    # Epsilon node
                    dot.node(node.node_id, 'ε', shape='plaintext', fillcolor='white')
                else:
                    # Terminal node
                    label = f"{node.value}"
                    dot.node(node.node_id, label, shape='box', fillcolor='lightgreen')
            else:
                # Non-terminal node
                dot.node(node.node_id, node.symbol, fillcolor='lightblue')
            
            # Add edges to children
            for child in node.children:
                add_nodes(child)
                dot.edge(node.node_id, child.node_id)
        
        add_nodes(root)
        
        # Render the graph in Parse_Tree directory
        filepath = os.path.join("Parse_Tree", filename)
        dot.render(filepath, format='png', cleanup=True)
        dot.render(filepath, format='pdf', cleanup=True)
        print(f"Parse tree saved as Parse_Tree/{filename}.png and .pdf")
        
        return dot
    
    def visualize_derivation_steps(self, root: ParseTreeNode, filename: str = "derivation_steps"):
        """Create an animated-style derivation visualization"""
        import os
        
        # Ensure Derivation_tree directory exists
        os.makedirs("Parse_Tree/Derivation_tree", exist_ok=True)
        
        dot = graphviz.Digraph(comment='Derivation Steps')
        dot.attr(rankdir='LR')  # Left to right for timeline effect
        
        # Style for derivation
        dot.attr('node', shape='record', style='filled', fillcolor='lightyellow')
        
        # Collect derivation steps
        steps = []
        self._collect_detailed_derivation_steps(root, steps)
        
        # Create timeline
        timeline = graphviz.Digraph('timeline')
        timeline.attr(rank='same')
        
        prev_id = None
        for i, (production, sentential_form) in enumerate(steps):
            node_id = f"step_{i}"
            
            # Format the step
            if production:
                label = f"Step {i}|{production}|{' '.join(sentential_form)}"
            else:
                label = f"Start|—|{' '.join(sentential_form)}"
            
            dot.node(node_id, label, width='4')
            
            if prev_id:
                dot.edge(prev_id, node_id, label=f"   ", style='bold')
            
            prev_id = node_id
        
        # Render only PDF in Derivation_tree subdirectory
        filepath = os.path.join("Parse_Tree", "Derivation_tree", filename)
        dot.render(filepath, format='pdf', cleanup=True)
        print(f"Derivation steps saved as Parse_Tree/Derivation_tree/{filename}.pdf")
        
        return dot
    
    def _collect_derivation_steps(self, root: ParseTreeNode, steps: List[List[str]]):
        """Collect derivation steps for visualization"""
        # Start with the start symbol
        current = [root.symbol]
        steps.append(current.copy())
        
        # Perform leftmost derivation
        self._leftmost_derivation(root, current, steps, 0)
    
    def _leftmost_derivation(self, node: ParseTreeNode, current: List[str], 
                           steps: List[List[str]], pos: int) -> int:
        """Perform leftmost derivation and collect steps"""
        if node.is_terminal:
            return pos + 1
        
        # Replace the non-terminal with its production
        if node.children:
            # Create new derivation
            new_derivation = current[:pos]
            
            # Add children symbols
            for child in node.children:
                if child.symbol == 'ε':
                    # Skip epsilon in the display
                    continue
                new_derivation.append(child.symbol)
            
            # Add remaining symbols
            new_derivation.extend(current[pos + 1:])
            
            # Only add if it's different from the last step
            if not steps or new_derivation != steps[-1]:
                steps.append(new_derivation)
                current[:] = new_derivation
        
        # Process children left-to-right
        child_pos = pos
        for child in node.children:
            if child.symbol != 'ε':
                child_pos = self._leftmost_derivation(child, current, steps, child_pos)
            
        return child_pos
    
    def visualize_derivation_tree(self, root: ParseTreeNode, filename: str = "derivation_tree"):
        """Backward compatibility - calls visualize_derivation_steps"""
        return self.visualize_derivation_steps(root, filename)
    
    def _collect_detailed_derivation_steps(self, root: ParseTreeNode, 
                                         steps: List[Tuple[str, List[str]]]):
        """Collect derivation steps with production rules"""
        current = [root.symbol]
        steps.append(("", current.copy()))  # Initial state
        
        self._detailed_leftmost_derivation(root, current, steps, 0)
    
    def _detailed_leftmost_derivation(self, node: ParseTreeNode, current: List[str], 
                                    steps: List[Tuple[str, List[str]]], pos: int) -> int:
        """Perform leftmost derivation with production tracking"""
        if node.is_terminal:
            return pos + 1
        
        if node.children:
            # Get production string
            production_parts = []
            for child in node.children:
                production_parts.append(child.symbol)
            production = f"{node.symbol} → {' '.join(production_parts)}"
            
            # Create new sentential form
            new_form = current[:pos]
            for child in node.children:
                if child.symbol != 'ε':
                    new_form.append(child.symbol)
            new_form.extend(current[pos + 1:])
            
            if new_form != current:
                steps.append((production, new_form.copy()))
                current[:] = new_form
        
        # Process children
        child_pos = pos
        for child in node.children:
            if child.symbol != 'ε':
                child_pos = self._detailed_leftmost_derivation(child, current, steps, child_pos)
        
        return child_pos
    
    def print_parse_tree(self, node: ParseTreeNode, indent: int = 0):
        """Print parse tree in text format"""
        prefix = "  " * indent + "|-- " if indent > 0 else ""
        
        if node.is_terminal:
            if node.token:
                print(f"{prefix}{node.symbol}: '{node.value}' @ pos {node.token.position}")
            else:
                print(f"{prefix}{node.symbol}")
        else:
            print(f"{prefix}{node.symbol}")
        
        for child in node.children:
            self.print_parse_tree(child, indent + 1)

# Test the implementation
if __name__ == "__main__":
    import sys
    
    # Check if graphviz is installed
    try:
        import graphviz
    except ImportError:
        print("Please install graphviz: pip install graphviz")
        print("Also make sure Graphviz software is installed on your system")
        sys.exit(1)
    
    if len(sys.argv) > 1:
        grammar_file = sys.argv[1]
    else:
        grammar_file = "grammar.ll1"
    
    print(f"Using grammar file: {grammar_file}")
    
    # Create lexer and parse tree generator
    lexer = LexicalAnalyzer(grammar_file)
    generator = ParseTreeGenerator(lexer)
    
    # Test inputs
    test_inputs = [
        "(a + b) * (c + d)",
        "(123)",
        "a * b * c + d",
        "x + y * z"
    ]
    
    for i, input_text in enumerate(test_inputs):
        print(f"\n{'='*60}")
        print(f"Parsing: '{input_text}'")
        print('='*60)
        
        try:
            success, parse_tree = generator.parse_with_tree(input_text)
            
            if success and parse_tree:
                print(f"\n✓ Parsing successful!")
                
                # Print text representation
                print("\nParse Tree (text format):")
                generator.print_parse_tree(parse_tree)
                
                # Generate visualizations
                print("\nGenerating visualizations...")
                generator.visualize_parse_tree(parse_tree, f"parse_tree_{i}")
                generator.visualize_derivation_steps(parse_tree, f"derivation_steps_{i}")
                
            else:
                print(f"\n✗ Parsing failed!")
                
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()