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
        self.token_types = {}
        self.token_patterns = {}
        self.non_terminals = []
        self.terminals = []
        self.start_symbol = None
        self.productions = {}
        
        self.EOF = TokenType("EOF")
        self.token_types["EOF"] = self.EOF
        
        self.keywords = {}
        self.id_token_type = None
        self.id_pattern = None
        self.other_patterns = []
        
        if grammar_file:
            self.load_grammar(grammar_file)
            self._prepare_dynamic_patterns()

    def _prepare_dynamic_patterns(self):
        # Heuristic: The ID pattern is complex and contains character sets for letters.
        # Keyword patterns are simple literal words.
        id_regex_heuristic = re.compile(r'.*\[[a-zA-Z].*') # Matches patterns with `[a-zA-Z]`
        
        temp_other_patterns = {}

        # First, find the generic ID pattern and separate it
        for token_type, pattern in self.token_patterns.items():
            if token_type.name == 'ID' and id_regex_heuristic.match(pattern):
                self.id_token_type = token_type
                self.id_pattern = pattern
            else:
                temp_other_patterns[token_type] = pattern
        
        # Second, from the remaining patterns, separate keywords from other symbols
        for token_type, pattern in temp_other_patterns.items():
            # A pattern is a keyword if it's a simple word and could be matched by the ID pattern
            is_potential_keyword = self.id_pattern and re.fullmatch(self.id_pattern, pattern)
            
            if is_potential_keyword:
                self.keywords[pattern] = token_type
            else:
                self.other_patterns.append((token_type, pattern))

    def load_grammar(self, grammar_file):
        try:
            with open(grammar_file, 'r') as file:
                grammar_content = file.read()
            self._parse_grammar(grammar_content)
        except Exception as e:
            print(f"Error loading grammar file: {e}")
            raise
    
    def _parse_grammar(self, grammar_content):
        # Extract start symbol
        start_match = re.search(r'^START\s*=\s*(\w+)', grammar_content, re.MULTILINE)
        if start_match:
            self.start_symbol = start_match.group(1).strip()
        
        # Extract non-terminals
        non_terminals_match = re.search(r'^NON_TERMINALS\s*=\s*(.+)', grammar_content, re.MULTILINE)
        if non_terminals_match:
            self.non_terminals = [nt.strip() for nt in non_terminals_match.group(1).split(',')]
        
        # Extract terminals
        terminals_match = re.search(r'^TERMINALS\s*=\s*(.+)', grammar_content, re.MULTILINE)
        if terminals_match:
            self.terminals = [t.strip() for t in terminals_match.group(1).split(',')]
            for terminal in self.terminals:
                self.token_types[terminal] = TokenType(terminal)
        
        # Extract production rules
        for non_terminal in self.non_terminals:
            pattern = fr'^{re.escape(non_terminal)}\s*->\s*(.+)'
            production_match = re.search(pattern, grammar_content, re.MULTILINE)
            if production_match:
                production_str = production_match.group(1)
                self.productions[non_terminal] = [p.strip() for p in production_str.split('|')]

        # Extract terminal patterns
        for terminal in self.terminals:
            pattern = fr'^{re.escape(terminal)}\s*->\s*(.+)'
            terminal_match = re.search(pattern, grammar_content, re.MULTILINE)
            if terminal_match:
                regex_pattern = terminal_match.group(1).strip()
                self.token_patterns[self.token_types[terminal]] = regex_pattern

    def tokenize(self, text):
        """
        Tokenizes text using the dynamically prepared patterns. It checks for
        symbols, then for identifiers/keywords.
        """
        tokens = []
        position = 0
        
        while position < len(text):
            # Skip whitespace
            match = re.match(r'\s+', text[position:])
            if match:
                position += match.end()
                continue
            
            token_found = False
            
            # 1. Try to match other patterns first (e.g., numbers, operators)
            for token_type, pattern in self.other_patterns:
                try:
                    match = re.match(f"^{pattern}", text[position:])
                    if match:
                        value = match.group(0)
                        tokens.append(Token(token_type, value, position))
                        position += match.end()
                        token_found = True
                        break
                except re.error as e:
                    print(f"Error in regex for {token_type}: {e}")

            if token_found:
                continue

            # 2. If no other pattern matched, try to match an identifier or keyword
            if self.id_pattern:
                match = re.match(f"^{self.id_pattern}", text[position:])
                if match:
                    value = match.group(0)
                    # Check if the matched value is a reserved keyword
                    token_type = self.keywords.get(value, self.id_token_type)
                    tokens.append(Token(token_type, value, position))
                    position += match.end()
                    token_found = True
            
            if not token_found:
                raise SyntaxError(f"Invalid token at position {position}: '{text[position:]}'")
        
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