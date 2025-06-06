import sys
import os
import re
from typing import Optional
from lexical_analyzer import LexicalAnalyzer
from parse_table import ParserTables
from dpda import DPDA
from parse_tree import ParseTreeGenerator  
from Symbol_renamer import SymbolRenamer  


class MiniCompiler:
    """Main class that integrates all components of the mini-compiler."""
    
    def __init__(self, grammar_file: str):
        """Initialize the compiler with a grammar file."""
        print(f"=== Initializing Mini-Compiler ===")
        print(f"Loading grammar from: {grammar_file}")
        
        try:
            self.lexer = LexicalAnalyzer(grammar_file)
            self.parser_tables = ParserTables(self.lexer)
            self.dpda = DPDA(self.lexer, self.parser_tables)
            self.tree_generator = ParseTreeGenerator(self.lexer)
            print("✓ All components initialized successfully")
        except Exception as e:
            print(f"✗ Initialization failed: {e}")
            raise
    
    def _sanitize_filename(self, text: str, max_length: int = 30) -> str:
        """Create a safe filename from input text."""
        # Replace special characters with safe alternatives
        replacements = {
            '*': 'star',
            '+': 'plus',
            '-': 'minus',
            '/': 'div',
            '(': 'lpar',
            ')': 'rpar',
            ' ': '_',
            '=': 'eq',
            '<': 'lt',
            '>': 'gt'
        }
        
        result = text
        for char, replacement in replacements.items():
            result = result.replace(char, replacement)
        
        # Remove any remaining non-alphanumeric characters
        result = re.sub(r'[^\w_]', '', result)
        
        # Truncate to reasonable length
        if len(result) > max_length:
            result = result[:max_length]
        
        # Ensure not empty
        if not result:
            result = "expression"
            
        return result
    
    def print_grammar_info(self):
        """Display information about the loaded grammar."""
        print("\n=== Grammar Information ===")
        self.lexer.print_grammar_info()
    
    def print_parse_table(self):
        """Display the LL(1) parse table."""
        print("\n=== LL(1) Parse Table ===")
        self.parser_tables.print_parse_table()
    
    def process_input(self, input_text: str):
        """Process an input string through all stages of the compiler."""
        print(f"\n=== Processing Input: '{input_text}' ===")
        
        # Stage 1: Lexical Analysis
        print("\n1. Lexical Analysis:")
        try:
            tokens = self.lexer.tokenize(input_text)
            print("   Tokens generated:")
            for i, token in enumerate(tokens):
                if token.type != self.lexer.EOF:
                    print(f"     {i}: {token}")
        except Exception as e:
            print(f"   ✗ Lexical analysis failed: {e}")
            return None
        
        # Stage 2: Syntax Analysis with DPDA
        print("\n2. Syntax Analysis (DPDA):")
        accepted, log_steps, _ = self.dpda.process(input_text)
        
        if accepted:
            print("   ✓ Input accepted by DPDA")
        else:
            print("   ✗ Input rejected by DPDA")
            return None
        
        # Stage 3: Parse Tree Generation (using fixed method)
        print("\n3. Parse Tree Generation:")
        success, parse_tree = self.tree_generator.generate_tree(input_text)
        
        if success and parse_tree:
            print("   ✓ Parse tree generated successfully")
            print("\n   Parse Tree Structure:")
            self.tree_generator.print_parse_tree(parse_tree)
            
            # Generate visualizations with sanitized filename
            safe_filename_base = self._sanitize_filename(input_text)
            
            # Visualize parse tree
            parse_tree_filename = f"parse_tree_{safe_filename_base}"
            self.tree_generator.visualize_parse_tree(parse_tree, parse_tree_filename)
            
            # Visualize derivation steps
            derivation_filename = f"derivation_{safe_filename_base}"
            self.tree_generator.visualize_derivation_steps(parse_tree, derivation_filename)
            
            return parse_tree
        else:
            print("   ✗ Parse tree generation failed")
            return None
    
    def interactive_rename(self, parse_tree, original_input: str):
        """Interactive symbol renaming interface with scope awareness."""
        print("\n=== Symbol Renaming ===")
        
        renamer = SymbolRenamer(parse_tree)
        identifiers = renamer.list_all_identifiers()
        
        if not identifiers:
            print("No identifiers found in the parse tree.")
            print("This might be because:")
            print("  1. The input doesn't contain any identifiers")
            print("  2. The parse tree wasn't built correctly")
            print("  3. The identifier detection logic needs adjustment")
            
            # Debug: Print all terminal nodes found
            print("\nDEBUG: All terminal nodes in parse tree:")
            self._debug_print_terminals(parse_tree)
            return None
        
        # Display available identifiers (group by value)
        identifier_values = {}
        for value, pos in identifiers:
            if value not in identifier_values:
                identifier_values[value] = []
            identifier_values[value].append(pos)
        
        print("\nAvailable identifiers:")
        for value, positions in identifier_values.items():
            print(f"  '{value}' (appears {len(positions)} time(s) at position(s): {positions})")
        
        # Get user selection by value
        while True:
            selection = input("\nEnter identifier value to rename (or 'q' to cancel): ").strip()
            if selection.lower() == 'q':
                print("Renaming cancelled.")
                return None
            
            if selection in identifier_values:
                break
            else:
                print(f"'{selection}' not found. Available identifiers: {list(identifier_values.keys())}")
        
        # Use the new scope-aware selection method
        target_node = renamer.select_identifier_by_scope(selection)
        
        if not target_node:
            print("Error: Could not find the selected identifier.")
            return None
        
        # Find related identifiers (now scope-aware)
        related = renamer.find_related_identifiers(target_node)
        print(f"\nFound {len(related)} occurrence(s) of '{target_node.value}' in the selected scope(s)")
        
        # Get new name
        new_name = input("Enter new name (or press Enter to cancel): ").strip()
        if not new_name:
            print("Renaming cancelled.")
            return None
        
        # Perform renaming
        try:
            modified_text = renamer.rename_symbol(target_node, new_name)
            print(f"\n✓ Renaming successful!")
            print(f"Modified code: {modified_text}")
            
            # Optionally save to file
            save = input("\nSave to file? (y/n): ").lower()
            if save == 'y':
                filename = input("Enter filename (or press Enter to cancel): ").strip()
                if filename:
                    with open(filename, 'w') as f:
                        f.write(modified_text)
                    print(f"✓ Saved to {filename}")
            
            # Ask if user wants to generate new parse tree and derivation
            generate_new = input("\nGenerate parse tree and derivation for modified code? (y/n): ").lower()
            if generate_new == 'y':
                print(f"\n{'='*60}")
                print(f"Processing modified code: '{modified_text}'")
                print(f"{'='*60}")
                
                # Process the modified code
                new_parse_tree = self.process_input(modified_text)
                if new_parse_tree:
                    # Ask if user wants to continue renaming on the new tree
                    continue_rename = input("\nContinue renaming on the modified code? (y/n): ").lower()
                    if continue_rename == 'y':
                        return self.interactive_rename(new_parse_tree, modified_text)
            
            return modified_text
        
        except Exception as e:
            print(f"✗ Renaming failed: {e}")
            return None
    
    def _debug_print_terminals(self, node, indent=0):
        """Debug helper to print all terminal nodes in the tree"""
        if not node:
            return
            
        prefix = "  " * indent
        if node.is_terminal:
            print(f"{prefix}TERMINAL: {node.symbol} = '{node.value}' (token: {node.token})")
        else:
            print(f"{prefix}NON-TERMINAL: {node.symbol}")
        
        for child in node.children:
            self._debug_print_terminals(child, indent + 1)


def main():
    """Main entry point of the program."""
    print("Theory of Languages and Machines - Mini Compiler Project")
    print("Authors: MohammadMahdi SharifBeigy, Mani Zamani")
    print("=" * 60)
    
    # Parse command line arguments
    if len(sys.argv) < 2:
        print("\nUsage: python main.py <grammar_file> [input_file]")
        print("  grammar_file: Path to the LL(1) grammar file")
        print("  input_file: Optional path to input code file (interactive mode if not provided)")
        sys.exit(1)
    
    grammar_file = 'Grammers/' + sys.argv[1]
    input_file = "Grammers/" + sys.argv[2] if len(sys.argv) > 2 else None
    
    # Check if grammar file exists
    if not os.path.exists(grammar_file):
        print(f"Error: Grammar file '{grammar_file}' not found.")
        sys.exit(1)
    
    try:
        # Initialize compiler
        compiler = MiniCompiler(grammar_file)
        compiler.print_grammar_info()
        
        # Show parse table if requested
        show_table = input("\nShow LL(1) parse table? (y/n/exit): ").lower()
        if show_table == 'exit':
            print("\nExiting...")
            sys.exit(0)
        elif show_table == 'y':
            compiler.print_parse_table()
            
            # Ask if user wants to export parse table
            export_table = input("\nExport parse table to files? (y/n): ").lower()
            if export_table == 'y':
                # Export to HTML
                html_file = compiler.parser_tables.export_parse_table_html()
                
                # Export to PDF visualization
                pdf_file = compiler.parser_tables.visualize_parse_table_graphviz()
                
                print(f"\nParse table exported to:")
                print(f"  - HTML: {html_file}")
                print(f"  - PDF:  {pdf_file}.pdf")
        
        # Main processing loop
        while True:
            # Process input
            if input_file:
                # Read from file (only for first iteration)
                with open(input_file, 'r') as f:
                    input_text = f.read().strip()
                print(f"\nProcessing file: {input_file}")
                print(f"Content: {input_text}")
                input_file = None  # Clear so we go to interactive mode after
            else:
                # Interactive mode
                print("\nEnter the code to process (or 'exit' to quit):")
                input_text = input("> ").strip()
                if input_text.lower() == 'exit':
                    print("\nExiting...")
                    break
            
            # Process the input
            parse_tree = compiler.process_input(input_text)
            
            if parse_tree:
                # Ask if user wants to rename symbols
                rename = input("\nPerform symbol renaming? (y/n/exit): ").lower()
                if rename == 'exit':
                    print("\nExiting...")
                    break
                elif rename == 'y':
                    result = compiler.interactive_rename(parse_tree, input_text)
                    # Result will be the modified text if successful, None otherwise
            else:
                print("\nParsing failed. Please check your input and try again.")
            
            # Ask if user wants to continue
            if not input_file:  # Only ask in interactive mode
                continue_choice = input("\nProcess another input? (y/n): ").lower()
                if continue_choice != 'y':
                    print("\nExiting...")
                    break
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()