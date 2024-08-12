import pennylane as qml

ops = [
    qml.RX(0.531, wires=0),
    qml.RY(0.9, wires=1),
    qml.RX(0.3, wires=2),

    qml.CZ(wires=(0,1)),
    qml.RY(-0.4, wires=0),

    qml.WireCut(wires=1),

    qml.CZ(wires=[1, 2]),
]
measurements = [qml.expval(qml.pauli.string_to_pauli_word("ZZZ"))]
tape = qml.tape.QuantumTape(ops, measurements)
print(qml.drawer.tape_text(tape))

graph = qml.qcut.tape_to_graph(tape)

qml.qcut.replace_wire_cut_nodes(graph)

fragments, communication_graph = qml.qcut.fragment_graph(graph)

fragment_tapes = [qml.qcut.graph_to_tape(f) for f in fragments]

print(30 * "-")
print(fragment_tapes[0].draw(decimals=2))
print(30 * "-")
print(fragment_tapes[1].draw(decimals=1))

dev = qml.device("default.qubit", wires=2)
fragment_tapes = [qml.map_wires(t, dict(zip(t.wires, dev.wires)))[0][0] for t in fragment_tapes]

expanded = [qml.qcut.expand_fragment_tape(t) for t in fragment_tapes]

configurations = []
prepare_nodes = []
measure_nodes = []
for tapes, p, m in expanded:
    configurations.append(tapes)
    prepare_nodes.append(p)
    measure_nodes.append(m)

tapes = tuple(tape for c in configurations for tape in c)
print(30 * "-")
for t in tapes:
    print(qml.drawer.tape_text(t))
    print()

# HIC SUNT LEONES
results = [qml.execute([t], qml.device("default.qubit", wires=dev.wires), gradient_fn=None)[0] for t in tapes]
res = qml.qcut.qcut_processing_fn(
    results,
    communication_graph,
    prepare_nodes,
    measure_nodes,
)
print("RES =", res)  # 0.47165198882111165
print("REF =", 0.47165198882111165)