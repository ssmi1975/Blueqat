from abc import abstractmethod
from enum import Enum
import re
from typing import Any, Callable, Dict, Iterable, List, Set, TextIO, Tuple, Union, NoReturn, Match
from itertools import takewhile

_regex_tokens = re.compile(r'OPENQASM 2.0|"[^"]+"|[a-zA-Z_][a-zA-Z_0-9]+|//|[0-9.]+|\S')
def split_tokens(qasmstr: str) -> Iterable[Tuple[int, str]]:
    for i, line in enumerate(qasmstr.split('\n')):
        toks = _regex_tokens.findall(line)
        yield from takewhile(
            lambda x: not x[1].startswith('//'),
            ((i, tok) for tok in toks))


def _err_with_lineno(lineno: int, msg: str) -> NoReturn:
    raise ValueError(f"Line {lineno}: {msg}")


class TokenGetter:
    def __init__(self, it: Iterable[Tuple[int, str]]) -> None:
        self.it = it
        self.buf = []
        self.lineno = 1

    def get(self) -> Tuple[int, str]:
        if self.buf:
            tok = self.buf.pop()
        else:
            try:
                tok = next(self.it)
            except StopIteration:
                return self.lineno, None
        self.lineno = tok[0]
        return tok

    def unget(self, tok: Tuple[int, str]) -> None:
        self.buf.append(tok)

    def _fail(self, action: Any) -> Any:
        if action is None:
            return None
        elif isinstance(action, str):
            _err_with_lineno(self.lineno, action)

    def get_if(self, cond: Any, or_else: Any = None) -> Union[Tuple[int, str], None]:
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

    def assert_semicolon(self, msg: str = '";" is expected.') -> None:
        self.get_if(';', msg)


def parse_qasm(qasmstr: str) -> 'QasmProgram':
    tokens = TokenGetter(split_tokens(qasmstr))
    errmsg = 'Program shall be start with "OPENQASM 2.0;".'
    tokens.get_if('OPENQASM 2.0', errmsg)
    tokens.assert_semicolon()
    stmts = []
    gates = {}
    qregs = {}
    cregs = {}
    included = set()
    _parse_statements(tokens, stmts, gates, qregs, cregs, included)
    return QasmProgram(stmts, gates, qregs, cregs, included)


def parse_qasmf(qasmfile: Union[str, TextIO], *args, **kwargs) -> 'QasmProgram':
    if isinstance(qasmfile, str):
        with open(qasmfile) as f:
            return parse_qasmf(f, *args, **kwargs)
    return parse_qasm(qasmfile.read(), *args, **kwargs)


class QasmNode:
    pass


class QasmProgram(QasmNode):
    def __init__(self,
                 statements: List[Any],
                 qregs: Dict[str, int],
                 cregs: Dict[str, int],
                 gates: Dict[str, Any],
                 included: Set[str]) -> None:
        self.statements = statements
        self.qregs = qregs
        self.cregs = cregs
        self.gates = gates
        self.included = included


    def __repr__(self) -> str:
        return f'QasmProgram({repr(self.statements)}, ' + \
               f'{repr(self.qregs)}, {repr(self.cregs)}, ' + \
               f'{repr(self.gates)}, {repr(self.included)})'


class QasmFloatExpr(QasmNode):
    pass


class QasmGateDef(QasmNode):
    pass


class QasmBarrier(QasmNode):
    pass


class QasmMeasure(QasmNode):
    pass


class QasmIf(QasmNode):
    pass


class QasmReset(QasmNode):
    pass


class QasmGateApply(QasmNode):
    def __init__(self,
                 gate: 'QasmAbstractGate',
                 params: List[QasmFloatExpr],
                 qregs: List[Tuple[str, int]]) -> None:
        self.gate = gate
        self.params = params
        self.qregs = qregs

    def __repr__(self) -> str:
        return f'QasmGateApply({repr(self.gate)}, {self.params}, {self.qregs})'


QasmGateType = Enum('QasmGateType', 'Gate Opaque Builtin')


class QasmAbstractGate:
    def __init__(self, name: str, params: List[str], qargs: List[str]):
        self.name = name
        self.params = params
        self.n_params = len(params)
        self.qargs = qargs
        self.n_qargs = len(qargs)

    @classmethod
    @abstractmethod
    def gatetype(cls) -> QasmGateType:
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}('{self.name}', {self.params}, {self.qargs})"


class QasmGate(QasmAbstractGate):
    @classmethod
    def gatetype(cls):
        return QasmGateType.Gate

    def __init__(self, gatedef: QasmGateDef):
        self.gatedef = gatedef
        name = ''; params = []; qargs = [] # TODO: Impl.
        super().__init__(name, params, qargs)


    def __repr__(self) -> str:
        return f'QasmGate({repr(self.gatedef)})'


class QasmOpaque(QasmAbstractGate):
    @classmethod
    def gatetype(cls):
        return QasmGateType.Opaque

    def __repr__(self) -> str:
        return f"QasmOpaque('{self.name}')"


class QasmBuiltinGate(QasmAbstractGate):
    @classmethod
    def gatetype(cls):
        return QasmGateType.Builtin

    def __init__(self, name: str):
        params = []; qargs = [] # TODO: Impl.
        super().__init__(name, params, qargs)


def _get_matcher(regex: str) -> Callable[[str], Match]:
    _re = re.compile(regex)
    def matcher(s: str) -> Match:
        return _re.match(s)
    return matcher


_is_symbol = _get_matcher(r'^[a-zA-Z_][a-zA-Z0-9_]*$')
_is_quoted_str = _get_matcher(r'^"[^"]*"$')
_is_uint = _get_matcher(r'^[1-9][0-9]*$')


def _parse_statements(tokens,
                      stmts: List[Any],
                      qregs: Dict[str, int],
                      cregs: Dict[str, int],
                      gates: Dict[str, Any],
                      included: Set[str]):
    lineno, tok = tokens.get()
    while tok:
        if tok == 'qreg':
            sym, num = _parse_reg(tokens)
            if sym in qregs or sym in cregs:
                _err_with_lineno(lineno, f'Register "{sym}" is already defined.')
            qregs[sym] = num
        elif tok == 'creg':
            sym, num = _parse_reg(tokens)
            if sym in qregs or sym in cregs:
                _err_with_lineno(lineno, f'Register "{sym}" is already defined.')
            cregs[sym] = num
        elif tok == 'include':
            incfile = _parse_include_stmt(tokens)
            if incfile in included:
                _err_with_lineno(lineno, f'File "{incfile}" is already included.')
            included.add(incfile)
            if incfile == "qelib1.inc":
                load_qelib1(gates)
            else:
                try:
                    with open(incfile) as f:
                        _parse_statements(f.read(), stmts, qregs, cregs, gates, included)
                except FileNotFoundError:
                    _err_with_lineno(lineno, f'Included file "{incfile}" is not exists.')
                except IsADirectoryError:
                    _err_with_lineno(lineno, f'Included file "{incfile}" is a directory.')
                except PermissionError:
                    _err_with_lineno(lineno, f'Cannot access to "{incfile}". Permission denied.')
                except OSError as e:
                    _err_with_lineno(lineno, f'During reading file {incfile}, Error occured. {e}')
        elif tok in ('gate', 'opaque'):
            if tok == 'gate':
                gate = _parse_def_gate(tokens)
            else:
                gate = _parse_opaque(tokens)
            if gate in gates:
                _err_with_lineno(lineno, f'Gate {gate} is already defined.')
            gates[gate] = gate
        elif tok == 'barrier':
            tokens.assert_semicolon()
            stmts.append(QasmBarrier())
        elif tok == 'if':
            stmts.append(_parse_if_stmt(tokens))
        elif tok == 'reset':
            stmts.append(_parse_reset_stmt(tokens))
        elif tok in gates:
            stmts.append(_parse_apply_gate(tokens))
        else:
            print(f"?{lineno}: {tok}")
        lineno, tok = tokens.get()


def _parse_params(tokens, allow_no_params: bool, allow_empty: bool) -> List[Any]:
    if allow_no_params:
        has_params = tokens.get_if('(')
        if not has_params:
            return []
    else:
        tokens.get_if('(', 'No parameter found.')
    params = []
    if tokens.get_if(')') is None:
        if allow_empty:
            return []
        _err_with_lineno(params[0], 'Empty parameter "()" is not allowed.')
    while 1:
        param = tokens.get()
        if param is None:
            _err_with_lineno(params[0], 'Unexpected end of file.')
        params.append(param[1])
        delim = tokens.get()
        if delim is None:
            _err_with_lineno(params[0], 'Unexpected end of file.')
        if delim[1] == ')':
            return params
        if delim[1] != ',':
            _err_with_lineno(params[0], f'Unexpected token "{delim[1]}".')


def _parse_reg(tokens):
    sym = tokens.get_if(_is_symbol, 'After "qreg", symbol is expected.')
    tokens.get_if('[', f'Unexpected token after "qreg {sym}".')
    num = tokens.get_if(_is_uint, f'After "qreg {sym}[", unsigned integer is expected.')
    tokens.get_if(']', 'Unclosed bracket "[".')
    tokens.assert_semicolon()
    return sym[1], int(num[1])


def _parse_include_stmt(tokens):
    incfile = tokens.get_if(_is_quoted_str, 'After "include", file path is expected.')
    tokens.assert_semicolon()
    return incfile[1][1:-1]


def _parse_if_stmt(tokens):
    # TODO: Impl.
    return QasmIf()


def _parse_reset_stmt(tokens):
    # TODO: Impl.
    return QasmReset()


def _parse_apply_gate(tokens):
    # TODO: Impl.
    return QasmReset()


def _parse_def_gate(tokens):
    # TODO: Impl.
    return QasmGateDef()


def _parse_opaque(tokens):
    name = tokens.get_if(_is_symbol, 'After "opaque", name is expected.')
    params = _parse_params(tokens, allow_no_params=True, allow_empty=False)
    return QasmOpaque(name, params)


def load_qelib1(gates: Dict[str, QasmAbstractGate]) -> None:
    from blueqat.circuit import GATE_SET
    for gate in GATE_SET:
        gates[gate] = QasmBuiltinGate(GATE_SET[gate].lowername)
    # TODO: These gates are not defined in qelib1, but defined in language specifications.
    gates['U'] = QasmBuiltinGate('u3')
    gates['CX'] = QasmBuiltinGate('cx')


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
