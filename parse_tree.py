from typing import List, Optional, Tuple
import graphviz
import os
import re
from lexical_analyzer import Token, LexicalAnalyzer, TokenType 
from parse_table import ParserTables
from dpda import DPDA

class ParseTreeNode:
    """Node in the parse tree"""
    def __init__(self, symbol: str, value: str = None, is_terminal: bool = False):
        self.symbol = symbol # The grammar symbol (e.g., 'E', 'IDENTIFIER', 'eps')
        self.value = value if value is not None else symbol # Actual value for terminals (e.g., 'myVar', '+')
        self.is_terminal = is_terminal
        self.children: List['ParseTreeNode'] = []
        self.parent: Optional['ParseTreeNode'] = None
        self.token: Optional[Token] = None  # For terminal nodes, store the original token
        self.node_id = None  # Unique ID for graphviz visualization

    def add_child(self, child: 'ParseTreeNode'):
        """Add a child node"""
        child.parent = self
        self.children.append(child)
        
    def __repr__(self):
        return f"ParseTreeNode({self.symbol}, val='{self.value}', term={self.is_terminal})"

class ParseTreeGenerator:
    """
    Generates a parse tree by processing the log from a DPDA.
    Also handles visualization of the parse tree and derivation steps.
    """
    def __init__(self, lexical_analyzer: LexicalAnalyzer):
        self.lexer = lexical_analyzer
        # ParserTables are needed by the DPDA and for symbol classification here
        self.parser_tables = ParserTables(lexical_analyzer) 
        # Instantiate the DPDA which will perform the core parsing
        self.dpda = DPDA(lexical_analyzer, self.parser_tables)
        self.node_counter = 0 # For generating unique node IDs for graphviz
        
    def _get_node_id(self) -> str:
        """Generate unique node ID for graphviz"""
        self.node_counter += 1
        return f"pt_node_{self.node_counter}" # Prefix to distinguish from DPDA vis nodes
        
    def generate_tree(self, input_text: str) -> Tuple[bool, Optional[ParseTreeNode]]:
        """
        Processes input text using the DPDA and, if accepted, generates a parse tree
        from the DPDA's processing log.
        """
        # Use the DPDA to process the input string
        accepted, log_steps, tokens = self.dpda.process(input_text)

        # Optionally print the DPDA's processing log for debugging/transparency
        print("\n--- DPDA Processing Log (from ParseTreeGenerator) ---")
        for step_line in log_steps:
            print(step_line)
        print("\n--- End of DPDA Log ---")        
        if not accepted:
            print("\nInput not accepted by DPDA. Cannot generate parse tree.")
            return False, None

        # If accepted by DPDA, attempt to build the tree from the log
        print("\nInput accepted by DPDA. Attempting to build parse tree from log...")
        try:
            # Reset node counter for each new tree visualization
            self.node_counter = 0 
            root_node = self._build_tree_from_log_fixed(log_steps, tokens)
            if root_node:
                 print("Parse tree built successfully from DPDA log.")
            return True, root_node
        except Exception as e:
            print(f"Error building tree from DPDA log: {e}")
            import traceback
            traceback.print_exc()
            return False, None

    def _parse_log_line_fixed(self, log_line: str) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
        """
        Parse a DPDA log line more robustly.
        Returns: (step, stack, input, action) or (None, None, None, None) if parsing fails
        """
        # Skip headers and empty lines
        if not log_line.strip() or "---" in log_line or log_line.startswith("Step"):
            return None, None, None, None
        
        # Try to parse using a more flexible approach
        # The format should be: step_num stack input action
        parts = log_line.split()
        if len(parts) < 4:
            return None, None, None, None
            
        # Check if first part is a step number
        try:
            step_num = int(parts[0])
        except ValueError:
            return None, None, None, None
            
        # Find where the action starts by looking for keywords
        action_keywords = ["Expand:", "Match", "Accept:", "Error:"]
        action_start_idx = -1
        
        for i, part in enumerate(parts):
            if any(keyword in part for keyword in action_keywords):
                action_start_idx = i
                break
        
        if action_start_idx == -1:
            return None, None, None, None
            
        # Extract components
        step = str(step_num)
        stack_parts = parts[1:action_start_idx]
        action_parts = parts[action_start_idx:]
        
        stack = " ".join(stack_parts) if stack_parts else ""
        action = " ".join(action_parts) if action_parts else ""
        
        return step, stack, "", action

    def _build_tree_from_log_fixed(self, log_steps: List[str], tokens: List[Token]) -> Optional[ParseTreeNode]:
        """
        Constructs a ParseTreeNode structure based on the log of actions from the DPDA
        and the list of tokens - FIXED VERSION
        """
        if not tokens:
            print("Error: No tokens provided.")
            return None

        start_symbol_name = self.lexer.start_symbol
        root_node = ParseTreeNode(start_symbol_name, is_terminal=False)
        
        # Stack to track nodes that need to be processed
        node_stack: List[ParseTreeNode] = [root_node]
        
        # Token iterator
        token_iter = iter(tokens)
        current_token = next(token_iter, None)
        
        # Process each log step
        for log_entry in log_steps:
            log_line = log_entry.strip()
            
            # Parse the log line
            step, stack, input_remainder, action_part = self._parse_log_line_fixed(log_line)
            
            if action_part is None:
                continue
            
            if action_part.startswith("Expand:"):
                if not node_stack:
                    continue
                    
                current_parent_node = node_stack.pop()
                
                # Extract the production from the action
                production_match = re.search(r'Expand:\s*(\w+)\s*->\s*(.+)', action_part)
                if not production_match:
                    continue
                
                expanded_symbol = production_match.group(1).strip()
                production_symbols_str = production_match.group(2).strip()
                
                if production_symbols_str == 'eps':
                    eps_node = ParseTreeNode('eps', value='ε', is_terminal=True)
                    current_parent_node.add_child(eps_node)
                else:
                    symbols_in_production = production_symbols_str.split()
                    children_nodes_created = []
                    for sym_str in symbols_in_production:
                        # Check if symbol is terminal
                        is_sym_terminal = sym_str in self.parser_tables.terminals
                        child_node = ParseTreeNode(sym_str, is_terminal=is_sym_terminal)
                        current_parent_node.add_child(child_node)
                        children_nodes_created.append(child_node)
                    
                    # Push new child nodes onto the stack in reverse order
                    for child_node in reversed(children_nodes_created):
                        node_stack.append(child_node)
            
            elif action_part.startswith("Match terminal:"):
                if not node_stack:
                    continue
                    
                terminal_node = node_stack.pop()
                
                # Get the next token and update the node
                if current_token and current_token.type != self.lexer.EOF:
                    terminal_node.value = current_token.value
                    terminal_node.token = current_token
                    terminal_node.is_terminal = True
                    
                    # Move to next token
                    current_token = next(token_iter, None)
            
            elif "Accept:" in action_part:
                break
            
            elif "Error:" in action_part:
                return None
        
        return root_node

    def visualize_parse_tree(self, root: Optional[ParseTreeNode], filename: str = "parse_tree"):
        """Create a visual representation of the parse tree using graphviz"""
        if not root:
            print("Cannot visualize an empty or invalid parse tree.")
            return

        # Ensure Parse_Tree directory exists
        output_dir = "Parse_Tree_From_DPDA_Log"
        os.makedirs(output_dir, exist_ok=True)
        
        dot = graphviz.Digraph(comment='Parse Tree (from DPDA Log)')
        dot.attr(rankdir='TB')  # Top to bottom
        
        dot.attr('node', shape='ellipse', style='filled', fontsize='10')
        dot.attr('edge', arrowsize='0.7')

        def add_nodes_to_graph(node: ParseTreeNode):
            """Recursively add nodes and edges to the graphviz Digraph."""
            node.node_id = self._get_node_id() # Assign unique ID for graphviz
            
            node_label = node.symbol
            fill_color = 'lightblue' # Default for non-terminals

            if node.is_terminal:
                if node.symbol == 'eps' or node.value == 'ε':
                    node_label = 'ε'
                    fill_color = 'whitesmoke'
                    dot.node(node.node_id, node_label, shape='plaintext', fillcolor=fill_color)
                else:
                    # For other terminals, show symbol and value (from token)
                    node_label = f"{node.symbol}\n('{node.value}')" 
                    fill_color = 'lightgreen'
                    dot.node(node.node_id, node_label, shape='box', fillcolor=fill_color)
            else: # Non-terminal
                dot.node(node.node_id, node_label, fillcolor=fill_color)
            
            for child in node.children:
                add_nodes_to_graph(child)
                dot.edge(node.node_id, child.node_id)
        
        add_nodes_to_graph(root)
        
        filepath = os.path.join(output_dir, filename)
        try:
            dot.render(filepath, format='png', cleanup=True, view=False)
            dot.render(filepath, format='pdf', cleanup=True, view=False)
            print(f"Parse tree (from DPDA log) saved to '{filepath}.png' and '{filepath}.pdf'")
        except Exception as e:
            print(f"❌ Error rendering parse tree graph: {e}")
            print("   Please ensure Graphviz is installed and in your system's PATH.")
        
        return dot
    
    def visualize_derivation_steps(self, root: Optional[ParseTreeNode], filename: str = "derivation_steps_from_log_tree"):
        if not root:
            print("Cannot visualize derivation for an empty tree.")
            return
        output_dir = "Parse_Tree_From_DPDA_Log/Derivations"
        os.makedirs(output_dir, exist_ok=True)
        
        dot = graphviz.Digraph(comment='Derivation Steps (from Log-Built Tree)')
        dot.attr(rankdir='LR') 
        dot.attr('node', shape='record', style='filled', fillcolor='lightyellow', fontsize='10')
        
        steps_data: List[Tuple[str, List[str]]] = []
        self.node_counter = 0 
        self._collect_detailed_derivation_steps(root, steps_data)
        
        prev_node_id_str = None
        for i, (production_rule, sentential_form_list) in enumerate(steps_data):
            current_node_id_str = f"deriv_step_{i}"
            
            label_production = production_rule if production_rule else "Start Symbol"
            label_sentential = ' '.join(sentential_form_list)
            if not label_sentential and production_rule.endswith("eps"):
                label_sentential = "ε"

            node_label_str = f"{{Step {i} | {label_production.replace('|', '\\|')} | {label_sentential.replace('|', '\\|')}}}"
            dot.node(current_node_id_str, node_label_str)
            
            if prev_node_id_str:
                dot.edge(prev_node_id_str, current_node_id_str, style='bold')
            prev_node_id_str = current_node_id_str
        
        filepath = os.path.join(output_dir, filename)
        try:
            dot.render(filepath, format='pdf', cleanup=True, view=False)
            print(f"Derivation steps (from log-built tree) saved to '{filepath}.pdf'")
        except Exception as e:
            print(f"❌ Error rendering derivation steps graph: {e}")
        return dot

    def _collect_detailed_derivation_steps(self, node: Optional[ParseTreeNode], 
                                         steps_list: List[Tuple[str, List[str]]], 
                                         current_sentential_form: Optional[List[str]] = None, 
                                         position_in_form: int = 0) -> int:
        """
        Performs a leftmost derivation traversal of the tree to collect sentential forms.
        This is a recursive helper.
        """
        if node is None:
            return position_in_form

        if current_sentential_form is None:
            current_sentential_form = [node.symbol]
            steps_list.append(("", current_sentential_form[:]))

        if node.is_terminal:
            if node.symbol == 'eps':
                return position_in_form 
            return position_in_form + 1

        # Non-terminal node - apply its production
        production_rhs_symbols = [child.symbol for child in node.children]
        production_rule_str = f"{node.symbol} → {' '.join(production_rhs_symbols) if production_rhs_symbols else 'eps'}"

        # Construct new sentential form
        new_sentential_form = current_sentential_form[:position_in_form]
        if production_rhs_symbols and production_rhs_symbols != ['eps']:
            new_sentential_form.extend(production_rhs_symbols)
        new_sentential_form.extend(current_sentential_form[position_in_form + 1:])
        
        # Avoid duplicates
        if not steps_list or steps_list[-1][1] != new_sentential_form:
             if new_sentential_form != current_sentential_form:
                  steps_list.append((production_rule_str, new_sentential_form[:]))
        
        # Process children recursively
        current_pos_for_children = position_in_form
        for child_node in node.children:
            if child_node.symbol == 'eps':
                continue
            current_pos_for_children = self._collect_detailed_derivation_steps(
                child_node, steps_list, new_sentential_form, current_pos_for_children
            )
        
        return current_pos_for_children

    def print_parse_tree(self, node: Optional[ParseTreeNode], indent: int = 0):
        """Print parse tree in text format"""
        if not node: return
        prefix = "  " * indent + "|-- " if indent > 0 else ""
        
        val_str = f": '{node.value}'" if node.value != node.symbol and node.is_terminal else ""
        token_pos_str = f" @ pos {node.token.position}" if node.token else ""

        print(f"{prefix}{node.symbol}{val_str}{token_pos_str}")
        
        for child in node.children:
            self.print_parse_tree(child, indent + 1)


if __name__ == "__main__":
    import sys
    
    try:
        import graphviz
    except ImportError:
        print("Please install graphviz: pip install graphviz")
        print("Also make sure Graphviz software is installed on your system (see graphviz.org/download/).")
        sys.exit(1)
    
    if len(sys.argv) > 1:
        grammar_file = "Grammers/" + sys.argv[1]
    else:
        grammar_file = "Grammers/grammar.ll1"
    
    print(f"Using grammar file: {grammar_file}")
    
    try:
        lexer = LexicalAnalyzer(grammar_file)
        generator = ParseTreeGenerator(lexer)
    except FileNotFoundError:
        print(f"Error: Grammar file '{grammar_file}' not found.")
        sys.exit(1)
    except ValueError as ve:
        print(f"Error during ParserTables initialization (grammar might not be LL(1)): {ve}")
        sys.exit(1)
    except Exception as e:
        print(f"Error initializing components: {e}")
        sys.exit(1)

    test_inputs = [
        "a * b + c",
        "( a + b ) * c",
        "id * id + id",
        "( 123 + 45 ) * num",
        "a + b ) * c", # Syntax error example
        "123 +",       # Premature EOF example
        "a b c"        # Lexical error / no rule example
    ]
    
    for i, input_text in enumerate(test_inputs):
        print(f"\n{'='*70}")
        print(f"Processing Input for Tree Generation: '{input_text}' (Test Case {i})")
        print(f"{'='*70}")
        
        try:
            success, parse_tree_root = generator.generate_tree(input_text)
            
            if success and parse_tree_root:
                print(f"\n✓ Parse tree generation successful for '{input_text}'!")
                
                print("\nParse Tree (text format):")
                generator.print_parse_tree(parse_tree_root)
                
                print("\nGenerating visualizations...")
                tree_vis_filename = f"parse_tree_log_{i}"
                generator.visualize_parse_tree(parse_tree_root, tree_vis_filename)
                
                deriv_vis_filename = f"derivation_log_tree_{i}"
                generator.visualize_derivation_steps(parse_tree_root, deriv_vis_filename)
                
            else:
                print(f"\n✗ Parse tree generation failed or input rejected for '{input_text}'.")
                
        except Exception as e:
            print(f"An unexpected error occurred processing '{input_text}': {e}")
            import traceback
            traceback.print_exc()