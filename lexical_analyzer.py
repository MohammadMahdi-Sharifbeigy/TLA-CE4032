import re

class TokenType:
    def __init__(self, name):
        self.name = name
    
    def __repr__(self):
        return f"TokenType.{self.name}"
    
    def __eq__(self, other):
        if isinstance(other, TokenType):
            return self.name == other.name
        return False
    
    def __hash__(self):
        return hash(self.name)

class Token:
    """Class to represent a token in the input text"""
    def __init__(self, token_type, value, position):
        self.type = token_type
        self.value = value
        self.position = position
    
    def __repr__(self):
        return f"Token({self.type}, '{self.value}', {self.position})"

class LexicalAnalyzer:
    """Lexical analyzer for the grammar"""
    def __init__(self, grammar_file=None):
        self.token_types = {}  # Dictionary mapping token names to TokenType objects
        self.token_patterns = {}  # Dictionary mapping TokenType objects to regex patterns
        self.non_terminals = []
        self.terminals = []
        self.start_symbol = None
        self.productions = {}  # Dictionary mapping non-terminals to their productions
        
        # Special token for end of file
        self.EOF = TokenType("EOF")
        self.token_types["EOF"] = self.EOF
        
        if grammar_file:
            self.load_grammar(grammar_file)
            
    def load_grammar(self, grammar_file):
        try:
            with open(grammar_file, 'r') as file:
                grammar_content = file.read()
            
            self._parse_grammar(grammar_content)
                
        except Exception as e:
            print(f"Error loading grammar file: {e}")
            raise
    
    def _parse_grammar(self, grammar_content):
        # Extract start symbol (START = X)
        start_match = re.search(r'^START\s*=\s*(\w+)', grammar_content, re.MULTILINE)
        if start_match:
            self.start_symbol = start_match.group(1).strip()
            print(f"Parsed start symbol: {self.start_symbol}")
        
        # Extract non-terminals (NON_TERMINALS = A, B, C)
        non_terminals_match = re.search(r'^NON_TERMINALS\s*=\s*(.+)', grammar_content, re.MULTILINE)
        if non_terminals_match:
            non_terminals_str = non_terminals_match.group(1)
            self.non_terminals = [nt.strip() for nt in non_terminals_str.split(',')]
            print(f"Parsed non-terminals: {self.non_terminals}")
        
        # Extract terminals (TERMINALS = X, Y, Z)
        terminals_match = re.search(r'^TERMINALS\s*=\s*(.+)', grammar_content, re.MULTILINE)
        if terminals_match:
            terminals_str = terminals_match.group(1)
            self.terminals = [t.strip() for t in terminals_str.split(',')]
            print(f"Parsed terminals: {self.terminals}")
            
            for terminal in self.terminals:
                token_type = TokenType(terminal)
                self.token_types[terminal] = token_type
                print(f"Created token type: {token_type}")
        
        # Extract production rules for non-terminals (A -> B C | D)
        for non_terminal in self.non_terminals:
            pattern = fr'{re.escape(non_terminal)}\s*->\s*(.+)'
            production_match = re.search(pattern, grammar_content)
            if production_match:
                production_str = production_match.group(1)
                productions = [p.strip() for p in production_str.split('|')]
                self.productions[non_terminal] = productions
                print(f"Parsed production for {non_terminal}: {productions}")
        
        # Extract terminal patterns (IDENTIFIER -> [a-zA-Z_][a-zA-Z0-9_]*)
        for terminal in self.terminals:
            pattern = fr'{re.escape(terminal)}\s*->\s*(.+)'
            terminal_match = re.search(pattern, grammar_content)
            if terminal_match:
                regex_pattern = terminal_match.group(1).strip()
                self.token_patterns[self.token_types[terminal]] = regex_pattern
                print(f"Parsed token pattern for {terminal}: {regex_pattern}")
    
    def tokenize(self, text):
        if not self.token_patterns:
            raise ValueError("No token patterns loaded. Please load a grammar file first.")
            
        tokens = []
        position = 0
        
        while position < len(text):
            # Skip whitespace
            match = re.match(r'\s+', text[position:])
            if match:
                position += match.end()
                continue
                
            token_found = False
            
            # Try to match each token pattern
            for token_type, pattern in self.token_patterns.items():
                try:
                    match = re.match(f"^{pattern}", text[position:])
                    if match:
                        value = match.group(0)
                        tokens.append(Token(token_type, value, position))
                        position += match.end()
                        token_found = True
                        break
                except re.error as e:
                    print(f"Error in regex pattern for {token_type}: {pattern}")
                    print(f"Error message: {e}")
            
            if not token_found:
                print(f"Available patterns: {self.token_patterns}")
                raise SyntaxError(f"Invalid token at position {position}: '{text[position]}'")
        
        # Add EOF token at the end
        tokens.append(Token(self.EOF, "", position))
        return tokens
    
    def print_grammar_info(self):
        """Print information about the loaded grammar"""
        print(f"Start Symbol: {self.start_symbol}")
        print(f"Non-Terminals: {', '.join(self.non_terminals)}")
        print(f"Terminals: {', '.join(self.terminals)}")
        print("Productions:")
        for nt, prods in self.productions.items():
            print(f"  {nt} -> {' | '.join(prods)}")
        print("Token Patterns:")
        for token_type, pattern in self.token_patterns.items():
            print(f"  {token_type} -> {pattern}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        grammar_file = sys.argv[1]
    else:
        grammar_file = "grammar.ll1"
    
    print(f"Using grammar file: {grammar_file}")
    
    lexer = LexicalAnalyzer(grammar_file)
    
    lexer.print_grammar_info()
    
    sample_inputs = [
        "(a + b) * (c + d)",
        "(123)",
        "a * b * c + d"
    ] 
    
    for input_text in sample_inputs:
        print(f"\nTokenizing: '{input_text}'")
        try:
            tokens = lexer.tokenize(input_text)
            for token in tokens:
                print(token)
        except Exception as e:
            print(f"Error tokenizing '{input_text}': {e}")