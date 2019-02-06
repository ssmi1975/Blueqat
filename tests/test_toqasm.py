from blueqat import Circuit

QASM = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
creg c[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(1.23) q[2];
x q[2];
y q[2];
cz q[2],q[1];
z q[1];
ry(4.56) q[0];
u1(1.32) q[1];
u2(1.23,4.56) q[2];
u3(1.0,2.0,3.0) q[0];
cu1(1.0) q[0],q[1];
cu2(2.0,1.0) q[1],q[2];
cu3(2.0,3.0,1.0) q[2],q[0];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];"""

def test_qasm1():
    c = Circuit()
    c.h[0, 1].cx[0, 1].rz(1.23)[2].x[2].y[2].cz[2, 1].z[1].ry(4.56)[0]
    c.u1(1.32)[1].u2(1.23, 4.56)[2].u3(1.0, 2.0, 3.0)[0]
    c.cu1(1.0)[0, 1].cu2(2.0, 1.0)[1, 2].cu3(2.0, 3.0, 1.0)[2, 0]
    qasm = c.m[:].to_qasm()
    assert QASM == qasm

def qasm_prologue(n_qubits):
    return "\n".join([
        "OPENQASM 2.0;",
        'include "qelib1.inc";',
        "qreg q[" + str(n_qubits) + "];",
        "creg c[" + str(n_qubits) + "];"
    ])

def test_qasm_nocache():
    correct_qasm = qasm_prologue(1) + "\nx q[0];\ny q[0];\nz q[0];"
    c = Circuit().x[0].y[0].z[0]
    c.run()
    c.to_qasm()
    qasm = c.to_qasm()
    assert qasm == correct_qasm

def test_qasm_noprologue():
    correct_qasm = "x q[0];\ny q[0];\nz q[0];"
    c = Circuit().x[0].y[0].z[0]
    qasm = c.to_qasm(output_prologue=False)
    assert qasm == correct_qasm

def test_qasm_noprologue2():
    correct_qasm = "x q[0];\ny q[0];\nz q[0];"
    c = Circuit().x[0].y[0].z[0]
    qasm = c.to_qasm(False)
    assert qasm == correct_qasm
