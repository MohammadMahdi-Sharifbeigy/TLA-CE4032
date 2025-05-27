from typing import Dict, List, Set, Tuple
import re
from dataclasses import dataclass
from lexical_analyzer import TokenType, Token, LexicalAnalyzer
import graphviz
import os

class ParserTables:
    def __init__(self, lexical_analyzer):
        self.lexer = lexical_analyzer
        self.terminals = set(self.lexer.terminals)
        self.terminals.add('$')
        self.non_terminals = set(self.lexer.non_terminals)
        self.grammar = {nt: [] for nt in self.non_terminals}
        self.start_symbol = self.lexer.start_symbol
        
        for nt, productions in self.lexer.productions.items():
            self.grammar[nt] = productions
        
        self.parse_table = {}
        self.first_sets = {}
        self.follow_sets = {}
        
        self.compute_first_sets()
        self.compute_follow_sets()
        self.build_parse_table()
    
    def _is_terminal(self, symbol: str) -> bool:
        return symbol in self.terminals or symbol == 'eps'
    
    def _is_non_terminal(self, symbol: str) -> bool:
        return symbol in self.non_terminals
    
    def compute_first_sets(self):
        self.first_sets = {nt: set() for nt in self.non_terminals}
        
        # For terminals, FIRST is the terminal itself
        for terminal in self.terminals:
            self.first_sets[terminal] = {terminal}
        
        # Special case for epsilon
        self.first_sets['eps'] = {'eps'}
        
        # Compute FIRST sets for non-terminals
        changed = True
        while changed:
            changed = False
            
            # For each non-terminal
            for nt, productions in self.grammar.items():
                # For each production rule of the non-terminal
                for production in productions:
                    # Split production into symbols
                    symbols = production.split()
                    
                    # Empty production (epsilon)
                    if not symbols or symbols[0] == 'eps':
                        if 'eps' not in self.first_sets[nt]:
                            self.first_sets[nt].add('eps')
                            changed = True
                        continue
                    
                    # Process each symbol in the production
                    all_can_derive_epsilon = True
                    for symbol in symbols:
                        # If terminal, add it to FIRST and break
                        if self._is_terminal(symbol):
                            if symbol not in self.first_sets[nt]:
                                self.first_sets[nt].add(symbol)
                                changed = True
                            all_can_derive_epsilon = False
                            break
                        
                        # Add all terminals from FIRST(symbol) to FIRST(nt)
                        for term in self.first_sets[symbol] - {'eps'}:
                            if term not in self.first_sets[nt]:
                                self.first_sets[nt].add(term)
                                changed = True
                        
                        # If symbol cannot derive epsilon, we're done
                        if 'eps' not in self.first_sets[symbol]:
                            all_can_derive_epsilon = False
                            break
                    
                    # If all symbols can derive epsilon, add epsilon to FIRST(nt)
                    if all_can_derive_epsilon and 'eps' not in self.first_sets[nt]:
                        self.first_sets[nt].add('eps')
                        changed = True
    
    def compute_follow_sets(self):
        self.follow_sets = {nt: set() for nt in self.non_terminals}
        
        # Add $ to FOLLOW of start symbol
        self.follow_sets[self.start_symbol].add('$')
        
        # Compute FOLLOW sets
        changed = True
        while changed:
            changed = False
            
            # For each non-terminal A
            for nt_a, productions in self.grammar.items():
                # For each production rule
                for production in productions:
                    symbols = production.split()
                    
                    # For each symbol B in the production
                    for i, symbol_b in enumerate(symbols):
                        # Only care about non-terminals
                        if not self._is_non_terminal(symbol_b):
                            continue
                        
                        # Get the symbols following B in the production
                        beta = symbols[i+1:] if i+1 < len(symbols) else []
                        
                        # If there are symbols after B
                        if beta:
                            # Calculate FIRST of the following symbols
                            first_of_beta = self._compute_first_of_string(beta)
                            
                            # Add everything except epsilon from FIRST(beta) to FOLLOW(B)
                            for terminal in first_of_beta - {'eps'}:
                                if terminal not in self.follow_sets[symbol_b]:
                                    self.follow_sets[symbol_b].add(terminal)
                                    changed = True
                            
                            # If beta can derive epsilon, add FOLLOW(A) to FOLLOW(B)
                            if 'eps' in first_of_beta:
                                for terminal in self.follow_sets[nt_a]:
                                    if terminal not in self.follow_sets[symbol_b]:
                                        self.follow_sets[symbol_b].add(terminal)
                                        changed = True
                        
                        # If B is the last symbol, add FOLLOW(A) to FOLLOW(B)
                        else:
                            for terminal in self.follow_sets[nt_a]:
                                if terminal not in self.follow_sets[symbol_b]:
                                    self.follow_sets[symbol_b].add(terminal)
                                    changed = True
    
    def _compute_first_of_string(self, symbols: List[str]) -> Set[str]:
        if not symbols:
            return {'eps'}
        
        result = set()
        all_can_derive_epsilon = True
        
        for symbol in symbols:
            if self._is_terminal(symbol):
                result.add(symbol)
                all_can_derive_epsilon = False
                break
            
            # If symbol is a non-terminal
            elif self._is_non_terminal(symbol):
                # Add all terminals from FIRST(symbol) except epsilon
                result.update(self.first_sets[symbol] - {'eps'})
                
                # If symbol cannot derive epsilon, we're done
                if 'eps' not in self.first_sets[symbol]:
                    all_can_derive_epsilon = False
                    break
        
        # If all symbols can derive epsilon, add epsilon to result
        if all_can_derive_epsilon:
            result.add('eps')
        
        return result
    
    def build_parse_table(self):
        self.parse_table = {
            nt: {terminal: None for terminal in self.terminals} 
            for nt in self.non_terminals
        }
        
        # For each non-terminal A
        for nt, productions in self.grammar.items():
            # For each production A -> α
            for production in productions:
                symbols = production.split()
                
                # Calculate FIRST(α)
                first_of_production = self._compute_first_of_string(symbols)
                
                # For each terminal a in FIRST(α)
                for terminal in first_of_production - {'eps'}:
                    # Set table[A, a] = α
                    if self.parse_table[nt][terminal] is not None:
                        raise ValueError(f"Grammar is not LL(1): Conflict at {nt}, {terminal}")
                    self.parse_table[nt][terminal] = production
                
                # If epsilon is in FIRST(α)
                if 'eps' in first_of_production:
                    # For each terminal b in FOLLOW(A)
                    for terminal in self.follow_sets[nt]:
                        # Set table[A, b] = ε
                        if self.parse_table[nt][terminal] is not None:
                            raise ValueError(f"Grammar is not LL(1): Conflict at {nt}, {terminal}")
                        self.parse_table[nt][terminal] = 'eps'
    
    def get_parse_table_entry(self, non_terminal: str, terminal: str) -> str:
        if non_terminal not in self.parse_table:
            raise ValueError(f"Unknown non-terminal: {non_terminal}")
        if terminal not in self.parse_table[non_terminal]:
            raise ValueError(f"Unknown terminal: {terminal}")
        return self.parse_table[non_terminal][terminal]
    
    def print_parse_table(self):
        col_widths = {terminal: max(len(terminal), 6) for terminal in self.terminals}
        for nt in self.non_terminals:
            for terminal in self.terminals:
                entry = self.parse_table[nt][terminal]
                if entry:
                    col_widths[terminal] = max(col_widths[terminal], len(entry))
        
        print("+" + "-" * 15 + "+", end="")
        for terminal in sorted(self.terminals):
            print("-" * (col_widths[terminal] + 2) + "+", end="")
        print()
        
        print("| Non-Terminal |", end="")
        for terminal in sorted(self.terminals):
            print(f" {terminal:{col_widths[terminal]}} |", end="")
        print()
        
        print("+" + "-" * 15 + "+", end="")
        for terminal in sorted(self.terminals):
            print("-" * (col_widths[terminal] + 2) + "+", end="")
        print()
        
        for nt in sorted(self.non_terminals):
            print(f"| {nt:13} |", end="")
            for terminal in sorted(self.terminals):
                entry = self.parse_table[nt][terminal] or ""
                print(f" {entry:{col_widths[terminal]}} |", end="")
            print()
        
        print("+" + "-" * 15 + "+", end="")
        for terminal in sorted(self.terminals):
            print("-" * (col_widths[terminal] + 2) + "+", end="")
        print()
    
    def export_parse_table_html(self, filename: str = "parse_table.html"):
        """Export parse table as a nicely formatted HTML file."""
        os.makedirs("Parse_Tables", exist_ok=True)
        filepath = os.path.join("Parse_Tables", filename)
        
        html_content = """<!DOCTYPE html>
<html>
<head>
    <title>LL(1) Parse Table</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }
        h1, h2 {
            color: #333;
        }
        table {
            border-collapse: collapse;
            width: auto;
            margin: 20px 0;
            background-color: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        th, td {
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }
        th {
            background-color: #4CAF50;
            color: white;
            font-weight: bold;
        }
        .non-terminal-header {
            background-color: #2196F3;
            color: white;
            font-weight: bold;
        }
        tr:nth-child(even) {
            background-color: #f2f2f2;
        }
        tr:hover {
            background-color: #e8f5e9;
        }
        .production {
            font-family: 'Courier New', monospace;
            color: #d32f2f;
        }
        .epsilon {
            color: #9c27b0;
            font-style: italic;
        }
        .set-display {
            font-family: 'Courier New', monospace;
            background-color: #f5f5f5;
            padding: 10px;
            border-radius: 4px;
            margin: 10px 0;
        }
    </style>
</head>
<body>
    <h1>LL(1) Parse Table</h1>
"""
        
        # Add grammar information
        html_content += f"<h2>Grammar Information</h2>"
        html_content += f"<p><strong>Start Symbol:</strong> {self.start_symbol}</p>"
        html_content += f"<p><strong>Non-Terminals:</strong> {', '.join(sorted(self.non_terminals))}</p>"
        html_content += f"<p><strong>Terminals:</strong> {', '.join(sorted(self.terminals))}</p>"
        
        # Add FIRST sets
        html_content += "<h2>FIRST Sets</h2>"
        html_content += "<div class='set-display'>"
        for nt in sorted(self.non_terminals):
            first_set = self.first_sets[nt]
            html_content += f"FIRST({nt}) = {{{', '.join(sorted(first_set))}}}<br>"
        html_content += "</div>"
        
        # Add FOLLOW sets
        html_content += "<h2>FOLLOW Sets</h2>"
        html_content += "<div class='set-display'>"
        for nt in sorted(self.non_terminals):
            follow_set = self.follow_sets[nt]
            html_content += f"FOLLOW({nt}) = {{{', '.join(sorted(follow_set))}}}<br>"
        html_content += "</div>"
        
        # Add parse table
        html_content += "<h2>Parse Table</h2>"
        html_content += "<table>"
        
        # Header row
        html_content += "<tr><th>Non-Terminal</th>"
        for terminal in sorted(self.terminals):
            html_content += f"<th>{terminal}</th>"
        html_content += "</tr>"
        
        # Data rows
        for nt in sorted(self.non_terminals):
            html_content += f"<tr><td class='non-terminal-header'>{nt}</td>"
            for terminal in sorted(self.terminals):
                entry = self.parse_table[nt][terminal]
                if entry:
                    if entry == 'eps':
                        html_content += f"<td class='epsilon'>ε</td>"
                    else:
                        html_content += f"<td class='production'>{entry}</td>"
                else:
                    html_content += "<td></td>"
            html_content += "</tr>"
        
        html_content += "</table></body></html>"
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"✓ Parse table exported to '{filepath}'")
        return filepath
    
    def visualize_parse_table_graphviz(self, filename: str = "parse_table_viz"):
        """Create a visual representation of the parse table using graphviz."""
        os.makedirs("Parse_Tables", exist_ok=True)
        
        dot = graphviz.Digraph(comment='LL(1) Parse Table', engine='fdp')
        dot.attr(rankdir='TB')
        dot.attr('node', shape='record', style='filled', fontsize='10')
        
        # Create a table-like structure
        table_label = "<<TABLE BORDER='0' CELLBORDER='1' CELLSPACING='0'>"
        
        # Header row
        table_label += "<TR><TD BGCOLOR='lightblue'><B>Non-Terminal</B></TD>"
        for terminal in sorted(self.terminals):
            table_label += f"<TD BGCOLOR='lightgreen'><B>{terminal}</B></TD>"
        table_label += "</TR>"
        
        # Data rows
        for nt in sorted(self.non_terminals):
            table_label += f"<TR><TD BGCOLOR='lightcoral'><B>{nt}</B></TD>"
            for terminal in sorted(self.terminals):
                entry = self.parse_table[nt][terminal]
                if entry:
                    if entry == 'eps':
                        table_label += "<TD>ε</TD>"
                    else:
                        # Escape special characters for HTML
                        escaped_entry = entry.replace('<', '&lt;').replace('>', '&gt;')
                        table_label += f"<TD>{escaped_entry}</TD>"
                else:
                    table_label += "<TD></TD>"
            table_label += "</TR>"
        
        table_label += "</TABLE>>"
        
        # Add the table as a single node
        dot.node('parse_table', label=table_label, shape='plaintext')
        
        # Save the visualization
        filepath = os.path.join("Parse_Tables", filename)
        try:
            dot.render(filepath, format='pdf', cleanup=True, view=False)
            print(f"✓ Parse table visualization saved to '{filepath}.pdf'")
        except Exception as e:
            print(f"❌ Error rendering parse table visualization: {e}")
        
        return filepath
    
    def print_first_sets(self):
        print("FIRST Sets:")
        for nt in sorted(self.non_terminals):
            first_set = self.first_sets[nt]
            print(f"  FIRST({nt}) = {{{', '.join(sorted(first_set))}}}")
    
    def print_follow_sets(self):
        print("FOLLOW Sets:")
        for nt in sorted(self.non_terminals):
            follow_set = self.follow_sets[nt]
            print(f"  FOLLOW({nt}) = {{{', '.join(sorted(follow_set))}}}")

class SyntaxAnalyzer:
    def __init__(self, lexical_analyzer):
        self.lexer = lexical_analyzer
        self.parser_tables = ParserTables(lexical_analyzer)
        
    def parse(self, input_text):
        # Tokenize the input
        tokens = self.lexer.tokenize(input_text)
        
        # Initialize the parsing stack with the end marker and start symbol
        stack = ['$', self.lexer.start_symbol]
        
        # Initialize the input token index
        token_index = 0
        current_token = tokens[token_index]
        
        # Parsing steps
        print(f"{'Step':<5} {'Stack':<30} {'Input':<20} {'Action':<30}")
        print("-" * 85)
        
        step = 1
        while stack:
            # Get the top of the stack
            top = stack[-1]
            
            # Print current state
            stack_str = ' '.join(stack)
            input_str = ' '.join([token.value for token in tokens[token_index:]])
            
            # If the top is the end marker
            if top == '$':
                if current_token.type == self.lexer.EOF:
                    print(f"{step:<5} {stack_str:<30} {input_str:<20} Accept")
                    return True
                else:
                    print(f"{step:<5} {stack_str:<30} {input_str:<20} Error: Expected EOF but got {current_token.value}")
                    return False
            
            # If the top is a terminal
            if top in self.parser_tables.terminals:
                # If it matches the current input token
                if (current_token.type.name == top and current_token.type != self.lexer.EOF) or \
                   (current_token.type == self.lexer.EOF and top == '$'):
                    action = f"Match {top}"
                    stack.pop()
                    token_index += 1
                    if token_index < len(tokens):
                        current_token = tokens[token_index]
                else:
                    print(f"{step:<5} {stack_str:<30} {input_str:<20} Error: Expected {top} but got {current_token.value}")
                    return False
            
            # If the top is a non-terminal
            elif top in self.parser_tables.non_terminals:
                # Get the production rule from the parse table
                terminal = current_token.type.name if current_token.type != self.lexer.EOF else '$'
                production = self.parser_tables.get_parse_table_entry(top, terminal)
                
                if production is None:
                    print(f"{step:<5} {stack_str:<30} {input_str:<20} Error: No production for {top} with {terminal}")
                    return False
                
                action = f"Expand {top} -> {production}"
                
                # Pop the non-terminal
                stack.pop()
                
                # Push the production in reverse order (right to left)
                if production != 'eps':
                    symbols = production.split()
                    for symbol in reversed(symbols):
                        stack.append(symbol)
            
            else:
                print(f"{step:<5} {stack_str:<30} {input_str:<20} Error: Unknown symbol {top}")
                return False
            
            print(f"{step:<5} {stack_str:<30} {input_str:<20} {action}")
            step += 1
        
        return False

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        grammar_file = sys.argv[1]
    else:
        grammar_file = "grammar.ll1" 
    
    print(f"Using grammar file: {grammar_file}")
    
    lexer = LexicalAnalyzer(grammar_file)
    
    lexer.print_grammar_info()
    
    syntax_analyzer = SyntaxAnalyzer(lexer)
    
    syntax_analyzer.parser_tables.print_first_sets()
    syntax_analyzer.parser_tables.print_follow_sets()
    syntax_analyzer.parser_tables.print_parse_table()
    
    sample_inputs = [
        "(a + b) * (c + d)",
        "(123)",
        "a * b * c + d",
        "a + b ) * c", # Example of a syntax error
        "123 +" # Example of premature EOF
    ]
    
    for input_text in sample_inputs:
        print(f"\nParsing: '{input_text}'")
        try:
            syntax_analyzer.parse(input_text)
        except Exception as e:
            print(f"Error parsing '{input_text}': {e}")