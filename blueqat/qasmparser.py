import re
from itertools import takewhile

_regex_tokens = re.compile(r'OPENQASM 2.0|"[^"]+"|[\w_]+|//|[0-9.]+|[=;*+-/(/)\[\]]')


def split_tokens(qasmstr):
    for i, line in enumerate(qasmstr.split('\n')):
        toks = _regex_tokens.findall(line)
        yield from takewhile(
                lambda x: not x[1].startswith('//'),
                ((i, tok) for tok in toks))


class TokenGetter:
    def __init__(self, it):
        self.it = it
        self.buf = []
        self.lineno = 1


    def get(self):
        if self.buf:
            tok = self.buf.pop()
        else:
            try:
                tok = next(self.it)
            except StopIteration:
                return self.lineno, None
        self.lineno = tok[0]
        return tok


    def unget(self, tok):
        self.buf.append(tok)


    def _fail(self, action):
        if action is None:
            return None
        elif isinstance(action, str):
            raise ValueError(f'Line {self.lineno}: {action}')


    def get_if(self, cond, or_else=None):
        tok = self.get()
        if tok is None:
            return self._fail(or_else)
        if isinstance(cond, str):
            if tok[1] == cond:
                return tok
            self.unget(tok)
            return self._fail(or_else)
        if hasattr(cond, '__call__'):
            if cond(tok[1]):
                return tok
            self.unget(tok)
            return self._fail(or_else)
        raise ValueError('Unknown conditions')


def parse_qasm(qasmstr):
    tokens = TokenGetter(split_tokens(qasmstr))
    errmsg = 'Program shall be start with "OPENQASM 2.0;".'
    tokens.get_if('OPENQASM 2.0', errmsg)
    tokens.get_if(';', errmsg)
    args = _parse_statements(tokens)
    return QasmProgram(*args)


def parse_qasmf(qasmfile, *args, **kwargs):
    if isinstance(qasmfile, str):
        with open(qasmfile) as f:
            return parse_qasmf(f, *args, **kwargs)
    return parse_qasm(qasmfile.read(), *args, **kwargs)


class QasmNode:
    pass


class QasmProgram(QasmNode):
    def __init__(self, statements, gates, qregs, cregs):
        self.statements = statements
        self.gates = gates
        self.qregs = qregs
        self.cregs = cregs


    def __repr__(self):
        return f'QasmProgram({repr(self.statements)}, ' + \
               f'{repr(self.gates)}, ' + \
               f'{repr(self.qregs)}, {repr(self.cregs)})'


_re_symbol = re.compile(r'^[a-zA-Z_][a-zA-Z0-9_]*$')

def _is_symbol(s):
    return _re_symbol.match(s)

_re_uint = re.compile(r'^[1-9][0-9]*$')

def _is_uint(s):
    return _re_uint.match(s)

def _parse_statements(tokens):
    stmts = []
    qregs = {}
    cregs = {}
    gates = {}
    lineno, tok = tokens.get()
    while tok:
        if tok == 'qreg':
            sym, num = _parse_reg(tokens)
            if sym in qregs or sym in cregs:
                raise ValueError(f'Register "{sym}" is already defined.')
            qregs[sym] = num
        elif tok == 'creg':
            sym, num = _parse_reg(tokens)
            if sym in qregs or sym in cregs:
                raise ValueError(f'Register "{sym}" is already defined.')
            cregs[sym] = num
        lineno, tok = tokens.get()
    return [], {}, qregs, cregs


def _parse_reg(tokens):
    sym = tokens.get_if(_is_symbol, 'After "qreg", symbol is expected.')
    tokens.get_if('[', f'Unexpected token after "qreg {sym}".')
    num = tokens.get_if(_is_uint, f'After "qreg {sym}[", unsigned integer is expected.')
    tokens.get_if(']', 'Unclosed bracket "[".')
    tokens.get_if(';', '";" not found.')
    return sym[1], int(num[1])


if __name__ == '__main__':
    # This QFT code is copied from IBM, OpenQASM project.
    # https://github.com/Qiskit/openqasm/blob/master/examples/generic/qft.qasm
    qftstr = '''
// quantum Fourier transform
OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
creg c[4];
x q[0]; 
x q[2];
barrier q;
h q[0];
cu1(pi/2) q[1],q[0];
h q[1];
cu1(pi/4) q[2],q[0];
cu1(pi/2) q[2],q[1];
h q[2];
cu1(pi/8) q[3],q[0];
cu1(pi/4) q[3],q[1];
cu1(pi/2) q[3],q[2];
h q[3];
measure q -> c;'''

    print(list(split_tokens("""
test; te st; te,s///// // //// /t [test1]""")))
    print(list(split_tokens(qftstr)))
    print(parse_qasm(qftstr))
