# dpda.py
from lexical_analyzer import LexicalAnalyzer, Token, TokenType
from parse_table import ParserTables
from typing import Tuple, List
import graphviz
import os

class DPDA:
    def __init__(self, lexical_analyzer: LexicalAnalyzer, parser_tables: ParserTables):
        """
        Initializes the DPDA.
        Args:
            lexical_analyzer: An instance of LexicalAnalyzer.
            parser_tables: An instance of ParserTables containing the LL(1) parse table.
        """
        self.lexer = lexical_analyzer
        self.parser_tables = parser_tables
        self.start_symbol = self.lexer.start_symbol
        self.eof_symbol = '$' 
        self._vis_node_id_counter = 0 # For unique node IDs in visualization

    def _get_vis_node_id(self) -> str:
        """Generates a unique node ID for graphviz visualizations."""
        self._vis_node_id_counter += 1
        return f"dpda_step_{self._vis_node_id_counter}"

    def process(self, input_text: str) -> Tuple[bool, List[str], List[Token]]:
        """
        Processes the input string using the DPDA logic based on the LL(1) parse table.

        Args:
            input_text: The string to be parsed.

        Returns:
            A tuple containing:
            - bool: True if the string is accepted, False otherwise.
            - List[str]: A list of strings describing each step of the parsing process.
            - List[Token]: The list of tokens generated from the input text by the lexer.
        """
        parsing_steps_log = []
        
        # 1. Tokenize the input string
        # The lexer.tokenize method should append an EOF token automatically.
        try:
            tokens = self.lexer.tokenize(input_text)
        except SyntaxError as e:
            parsing_steps_log.append(f"Lexical Error: {e}")
            return False, parsing_steps_log, []
        
        # 2. Initialize the parsing stack
        # The stack will store grammar symbols (strings).
        # It's initialized with the EOF marker and the grammar's start symbol.
        # The start symbol is at the top initially.
        stack = [self.eof_symbol, self.start_symbol]
        
        token_index = 0 # Points to the current input token
        current_token_obj = tokens[token_index]

        accepted = False # Flag to indicate if parsing was successful
        step_num = 1 # Counter for logging steps

        # Log headers
        header = f"{'Step':<5} {'Stack (Top Left)':<40} {'Input Remainder':<30} {'Action':<50}"
        parsing_steps_log.append(header)
        parsing_steps_log.append("-" * len(header))

        # 3. Main parsing loop
        # The loop continues as long as the stack is not empty.
        # The loop should ideally terminate when stack top is '$' and input is also '$' (EOF).
        max_steps = 1000 # Safety break for very long or potentially looping parses

        while stack and step_num <= max_steps:
            top_of_stack = stack[-1] # Peek at the top of the stack
            
            # Determine the current input terminal's type name for table lookup.
            # If current token is EOF, use self.eof_symbol.
            current_input_terminal_type = current_token_obj.type.name if current_token_obj.type != self.lexer.EOF else self.eof_symbol

            # --- Logging current state ---
            # Display stack with top at the left for conventional representation
            stack_display = ' '.join(reversed(stack)) 
            # Display remaining input tokens
            remaining_input_display = ' '.join(
                [t.value if t.type != self.lexer.EOF else self.eof_symbol 
                 for t in tokens[token_index:]]
            )
            # Ensure consistent field widths for parsing later in visualization
            current_step_str_prefix = f"{step_num:<5} {stack_display:<40.40} {remaining_input_display:<30.30} "
            # --- End Logging ---

            action_taken_str = ""

            # A. If top of stack is EOF marker ($)
            if top_of_stack == self.eof_symbol:
                if current_input_terminal_type == self.eof_symbol:
                    action_taken_str = "Accept: Stack and Input both at EOF."
                    accepted = True
                else:
                    # Stack is empty (only $), but input is not yet EOF.
                    action_taken_str = f"Error: Stack empty, but input remains ('{current_token_obj.value}')."
                parsing_steps_log.append(current_step_str_prefix + action_taken_str)
                break # Halt processing (either success or error)

            # B. If top of stack is a terminal symbol
            elif top_of_stack in self.parser_tables.terminals:
                if top_of_stack == current_input_terminal_type:
                    # Match: Pop stack, advance input pointer
                    action_taken_str = f"Match terminal: '{top_of_stack}'"
                    stack.pop()
                    token_index += 1
                    if token_index < len(tokens):
                        current_token_obj = tokens[token_index]
                    else:
                        # This case should ideally not be reached if EOF handling is correct,
                        action_taken_str = "Error: Input exhausted prematurely after a match."
                        parsing_steps_log.append(current_step_str_prefix + action_taken_str)
                        break 
                else:
                    # Mismatch error
                    action_taken_str = f"Error: Terminal mismatch. Expected '{top_of_stack}', got '{current_input_terminal_type}'."
                    parsing_steps_log.append(current_step_str_prefix + action_taken_str)
                    break # Halt on error
            
            # C. If top of stack is a non-terminal symbol
            elif top_of_stack in self.parser_tables.non_terminals:
                # Consult the parse table: M[Non-Terminal, Current_Input_Terminal]
                production_rule = self.parser_tables.get_parse_table_entry(top_of_stack, current_input_terminal_type)
                
                if production_rule is None:
                    # No rule in parse table -> Syntax error
                    action_taken_str = f"Error: No production rule for M['{top_of_stack}', '{current_input_terminal_type}']."
                    parsing_steps_log.append(current_step_str_prefix + action_taken_str)
                    break # Halt on error
                
                # Apply production: Pop Non-Terminal, Push production symbols (reversed)
                action_taken_str = f"Expand: {top_of_stack} -> {production_rule}"
                stack.pop() # Pop the non-terminal
                
                if production_rule != 'eps': # 'eps' means epsilon (empty string)
                    symbols_to_push = production_rule.split()
                    for symbol in reversed(symbols_to_push):
                        stack.append(symbol)
            
            # D. Should not happen with a correct setup
            else:
                action_taken_str = f"Error: Unknown symbol on stack: '{top_of_stack}'."
                parsing_steps_log.append(current_step_str_prefix + action_taken_str)
                break 
            
            parsing_steps_log.append(current_step_str_prefix + action_taken_str)
            step_num += 1

        if step_num > max_steps:
            parsing_steps_log.append(f"Error: Exceeded maximum parsing steps ({max_steps}). Possible loop or very complex input.")

        return accepted, parsing_steps_log, tokens

    def visualize_derivation_steps(self, parsing_steps_log: List[str], filename_prefix: str = "dpda_derivation"):
        if not parsing_steps_log or len(parsing_steps_log) <= 2:
            print("No derivation steps to visualize or log is too short.")
            return

        output_dir = "DPDA_Derivations"
        os.makedirs(output_dir, exist_ok=True)

        dot = graphviz.Digraph(comment='DPDA Derivation Steps')
        dot.attr(rankdir='TB')
        dot.attr('node', shape='record', style='filled', fillcolor='lightyellow', fontsize='10')
        dot.attr('edge', fontsize='10')
        
        self._vis_node_id_counter = 0

        actual_steps_log = parsing_steps_log[2:]
        
        prev_node_id = None

        for log_line in actual_steps_log:
            if "Error: Exceeded maximum parsing steps" in log_line:
                current_node_id = self._get_vis_node_id()
                dot.node(current_node_id, label=log_line.replace("Error: ", ""), color='red', fillcolor='lightpink')
                if prev_node_id:
                    dot.edge(prev_node_id, current_node_id)
                prev_node_id = current_node_id
                continue
            if "Lexical Error:" in log_line:
                current_node_id = self._get_vis_node_id()
                dot.node(current_node_id, label=log_line.replace("Lexical Error: ", ""), color='red', fillcolor='lightpink')
                prev_node_id = current_node_id
                continue
            try:
                step_val = log_line[0:5].strip()
                stack_val = log_line[5:45].strip()
                input_val = log_line[45:75].strip()
                action_val = log_line[75:].strip()
            except IndexError:
                print(f"Warning: Could not parse log line for visualization: '{log_line}'")
                continue

            stack_val_escaped = stack_val.replace('|', '\\|').replace('{', '\\{').replace('}', '\\}').replace('<', '\\<').replace('>', '\\>')
            input_val_escaped = input_val.replace('|', '\\|').replace('{', '\\{').replace('}', '\\}').replace('<', '\\<').replace('>', '\\>')
            action_val_escaped = action_val.replace('|', '\\|').replace('{', '\\{').replace('}', '\\}').replace('<', '\\<').replace('>', '\\>')

            node_label = (f"{{Step: {step_val} |"
                          f"Stack: {stack_val_escaped} |"
                          f"Input: {input_val_escaped} |"
                          f"Action: {action_val_escaped}}}")
            
            current_node_id = self._get_vis_node_id()
            
            node_color = 'lightgreen' if "Accept" in action_val else 'lightyellow'
            if "Error" in action_val:
                node_color = 'lightpink'

            dot.node(current_node_id, label=node_label, fillcolor=node_color)

            if prev_node_id:
                dot.edge(prev_node_id, current_node_id)
            
            prev_node_id = current_node_id

        output_filepath = os.path.join(output_dir, filename_prefix)
        try:
            dot.render(output_filepath, format='pdf', cleanup=True, view=False)
            print(f"✅ DPDA derivation visualization saved to '{output_filepath}.pdf'")
        except Exception as e:
            print(f"❌ Error rendering DPDA derivation graph: {e}")
            print("   Please ensure Graphviz is installed and in your system's PATH.")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        grammar_file_path = "Grammers/" + sys.argv[1]
    else:
        grammar_file_path = "Grammers/grammar.ll1" 
        print(f"No grammar file provided. Using default: '{grammar_file_path}'")

    try:
        lexer = LexicalAnalyzer(grammar_file_path)
        print(f"\n--- Lexer and Grammar Info ({grammar_file_path}) ---")
        lexer.print_grammar_info()

        print("\n--- Computing Parser Tables ---")
        parser_tables = ParserTables(lexer) # This might raise ValueError if grammar not LL(1)
        parser_tables.print_first_sets()
        parser_tables.print_follow_sets()
        parser_tables.print_parse_table() 

        dpda_parser = DPDA(lexer, parser_tables)
        print("\n--- DPDA Initialized ---")

        sample_inputs = [
            "a * b + c",
            "( a + b ) * c",
            "id * id + id",
            "a + b * c + d",
            "( 123 + 45 ) * num",
            "a + b ) * c", # Example of a syntax error
            "123 +" # Example of premature EOF
        ]
        
        for i, test_input_str in enumerate(sample_inputs):
            print(f"\n--- Processing Input: '{test_input_str}' ---")
            is_accepted, steps_log, _ = dpda_parser.process(test_input_str)
            
            for log_entry in steps_log:
                print(log_entry)
            
            if is_accepted:
                print(f"✅ Result: Input '{test_input_str}' is ACCEPTED.")
            else:
                print(f"❌ Result: Input '{test_input_str}' is REJECTED.")
            
            # Visualize the derivation steps
            # Create a unique filename for each input's derivation
            vis_filename = f"dpda_derivation_{i}_{test_input_str.replace(' ', '_').replace('(', '').replace(')', '').replace('*','mul').replace('+','plus')[:20]}"
            dpda_parser.visualize_derivation_steps(steps_log, filename_prefix=vis_filename)

    except FileNotFoundError:
        print(f"Error: Grammar file '{grammar_file_path}' not found.")
    except ValueError as ve: # Catch conflicts from ParserTables
        print(f"Setup Error (likely grammar not LL(1)): {ve}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
