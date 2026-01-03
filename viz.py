from graphviz import Digraph

def trace(root):
    nodes, edges = set(), set()   # nodes = ids, edges = (id,id)

    def _build(v):
        if v in nodes:
            return
        nodes.add(v)
        if v._ctx:
            for child in v._ctx.parents:
                edges.add((child, v))
                _build(child)

    _build(root)
    return nodes, edges

from graphviz import Digraph

def draw_dot(root, format='svg', rankdir='LR'):
    assert rankdir in ['LR', 'TB']
    nodes, edges = trace(root)

    dot = Digraph(format=format, graph_attr={'rankdir': rankdir})

    # tensor nodes
    for node in nodes:
        t = node
        data = t
        grad = t.grad if t.grad is not None else None
        if grad is not None:
            grad = f"{grad.lazydata if grad.lazydata.realized is None else (grad.shape ,grad.lazydata.op_type.__name__)}"
        label = label = (
                    f"Tensor {data.lazydata if data.lazydata.realized is None else (data.shape, data.lazydata.op_type.__name__)}\n"
                    f"Grad: {grad}"
                )
        dot.node(name=str(id(t)), label=label, shape="box")

        # op node
        if t._ctx:
            op_name = t._ctx.__class__.__name__
            op_id = f"{id(t)}_{op_name}"

            dot.node(name=op_id, label=op_name, shape='ellipse')
            dot.edge(op_id, str(id(t)))

    # edges (parent tensor → op)
    for p, c in edges:
        if c._ctx:
            op_name = c._ctx.__class__.__name__
            dot.edge(str(id(p)), f"{id(c)}_{op_name}")

    return dot
