"""Specialize the static K3 component IR. Reject unrecognized driver shapes.

The packed driver supplies the slot offsets and the subgraph operand ordering.
Make the actual subgraph the private MLIR entry, so no full-stage return struct
or temporary contiguous weights/rope bucket is needed.
"""

import re


def one(pattern, text):
    matches = re.findall(pattern, text, re.M)
    if len(matches) != 1:
        raise ValueError(
            f"expected one match for {pattern}, got {len(matches)}"
        )
    return matches[0]


def driver_arguments(ir, role):
    roots = {
        "%arg0": "w->f32",
        "%arg1": "w->q4",
        "%arg2": "input",
        "%arg3": "&pos",
        "%arg4": "k",
        "%arg5": "v",
    }
    if role == "embed":
        roots = {"%arg0": "table", "%arg1": "ids"}
    for line in ir.splitlines():
        m = re.match(
            r"\s*(%\w+) = memref\.(subview|expand_shape|cast) (%\w+)(.*)", line
        )
        if not m:
            continue
        result, op, src, tail = m.groups()
        expr = roots[src]
        if op == "subview":
            off = int(one(r"^\[(\d+)\]", tail))
            expr += f" + {off}"
        roots[result] = expr
    args = one(r"= call @subgraph0_\w+\(([^)]*)\)", ir).split(", ")
    return [roots[a] for a in args]


def subgraph(ir, driver, role, phase, rows, dim):
    symbol = "subgraph0_" + phase
    signature = one(r"^  func.func @" + symbol + r"\((.*?)\) -> .*? \{$", ir)
    bindings = driver_arguments(driver, role)
    shapes = re.findall(r"(%arg\d+): tensor<([\dx]+)(f32|i8|i64)>", signature)
    if len(shapes) != len(bindings):
        raise ValueError("driver vs tensor signature mismatch")
    result, ty = one(r"^    return ([^:]+) : (.*)$", ir)
    result = result.split(", ")[-1]
    ty = ty.split(", ")[-1]
    output = ty.replace("tensor", "memref")
    args = list(zip(bindings, [s[1].rstrip("x") for s in shapes], strict=True))
    if role == "decoder_layer":
        # K3 attention takes inv_freq after cache_position. Qwen3 then
        # passes its two per-head RMSNorm weights.
        operands = one(r"call @k3_attn_\w+\(([^)]*)\)", ir).split(", ")
        if len(operands) not in (7, 9):
            raise ValueError("unsupported K3 attention signature")
        inv = operands[6]
        idx = [s[0] for s in shapes].index(inv)
        old = "tensor<" + shapes[idx][1] + "f32>"
        row_ty = f"tensor<{rows * dim}xf32>"
        signature = signature.replace(f"{inv}: {old}", f"{inv}: {row_ty}")
        signature += ", %sines: " + row_ty
        args[idx] = ("frame->cos_rows", str(rows * dim))
        args.append(("frame->sin_rows", str(rows * dim)))
        lines = []
        for line in ir.splitlines():
            if "call @k3_attn_" in line:
                call, types = line.split(" : (")
                inputs, outputs = types.split(") -> ", 1)
                operands.insert(7, "%sines")
                inputs = inputs.split(", ")
                inputs[6:7] = [row_ty, row_ty]
                line = (
                    call[: call.index("(") + 1]
                    + ", ".join(operands)
                    + ") : ("
                    + ", ".join(inputs)
                    + ") -> "
                    + outputs
                )
            elif "private @k3_attn_" in line:
                head, rest = line.split("(", 1)
                params, outputs = rest.split(") -> ", 1)
                params = params.split(", ")
                if not params[6].startswith(old + " "):
                    raise ValueError("expected inv_freq at attention operand 6")
                params[6:7] = [params[6].replace(old, row_ty)] * 2
                line = head + "(" + ", ".join(params) + ") -> " + outputs
            lines.append(line)
        ir = "\n".join(lines)
    signature += ", %dest: " + output
    args.append(("out", ty[7:-1].removesuffix("xf32")))
    ir = re.sub(
        r"^  func.func @" + symbol + r"\(.*?\) -> .*? \{$",
        f"  func.func @{symbol}({signature}) attributes {{llvm.emit_c_interface}} {{",
        ir,
        flags=re.M,
    )
    ir = re.sub(
        r"^    return [^\n]+$",
        f"    bufferization.materialize_in_destination {result} in restrict writable %dest : ({ty}, {output}) -> ()\n    return",
        ir,
        flags=re.M,
    )
    return ir, args


ROW = "memref<?xf32, strided<[1], offset: ?>>"


def prepare_ir(rows, dim, phase):
    half = dim // 2
    vec = f"vector<{half}xf32>"
    buf = f"memref<{rows}x{dim}xf32>"
    return f"""module {{
  func.func @rb_prepare_{phase}(%inv: memref<{half}xf32>, %pos: memref<1xi64>, %c: {buf}, %s: {buf}) attributes {{llvm.emit_c_interface}} {{
    %z = arith.constant 0 : index
    %step = arith.constant 1 : index
    %end = arith.constant {rows} : index
    %half = arith.constant {half} : index
    %start = memref.load %pos[%z] : memref<1xi64>
    %freq = vector.load %inv[%z] : memref<{half}xf32>, {vec}
    scf.for %i = %z to %end step %step {{
      %i64 = arith.index_cast %i : index to i64
      %p = arith.addi %start, %i64 : i64
      %fp = arith.sitofp %p : i64 to f32
      %row = vector.broadcast %fp : f32 to {vec}
      %angle = arith.mulf %freq, %row : {vec}
      %cv = math.cos %angle : {vec}
      %sv = math.sin %angle : {vec}
      vector.store %cv, %c[%i,%z] : {buf}, {vec}
      vector.store %cv, %c[%i,%half] : {buf}, {vec}
      vector.store %sv, %s[%i,%z] : {buf}, {vec}
      vector.store %sv, %s[%i,%half] : {buf}, {vec}
    }}
    return
  }}
}}"""


def prepared_attention(ir, chunk, dim):
    """Lift the K3 vector sin/cos prologue; preserve its rotated KV stores."""

    def rewrite(match):
        body = match[0]
        rows = int(one(r"@k3_attn_m(\d+)_", body.splitlines()[0]))
        if rows not in (1, chunk):
            raise ValueError("unsupported attention rows")
        body = body.replace(
            "%arg6: " + ROW, "%arg6: " + ROW + ", %sines: " + ROW
        )
        for op, src in (("cos", "%arg6"), ("sin", "%sines")):
            value, vec = one(
                r"^ +(%\w+) = math\." + op + r" %\w+ : (vector<\d+xf32>)$", body
            )
            stores = re.findall(
                r"^ +vector.store "
                + value
                + r", (%\w+)\[([^\]]+)\] : (memref<[^>]+>),",
                body,
                re.M,
            )
            mem, indices, old = stores[0]
            shape = [int(x) for x in old[7:-1].split("x")[:-1]]
            strides = [dim, 1] if len(shape) == 2 else [dim, dim // 2, 1]
            new = (
                old if len(shape) == 3 else old[:-1] + f", strided<[{dim}, 1]>>"
            )
            cast = f"{mem} = memref.reinterpret_cast {src} to offset: [0], sizes: {shape}, strides: {strides} : {ROW} to {new}"
            body = re.sub(
                re.escape(mem)
                + r" = memref.alloc\(\) alignment = \d+ : "
                + re.escape(old),
                cast,
                body,
                count=1,
            )
            body = re.sub(
                r"^ +vector.store " + value + r", " + mem + r"\[.*\n",
                "",
                body,
                flags=re.M,
            )
            body = re.sub(
                r"^ +memref.dealloc " + mem + r" : .*\n", "", body, flags=re.M
            )
            body = "\n".join(
                line.replace(old, new)
                if re.search(re.escape(mem) + r"\[", line)
                else line
                for line in body.split("\n")
            )
            body = re.sub(
                r"math\." + op + r" %\w+ : " + vec,
                f"vector.load {mem}[{indices}] : {new}, {vec}",
                body,
            )
        return body

    ir, n = re.subn(
        r"^  func.func @k3_attn_m\d+_[^\n]+\{\n.*?^  \}",
        rewrite,
        ir,
        flags=re.S | re.M,
    )
    if n != 2:
        raise ValueError(f"expected two K3 attention kernels, got {n}")
    return ir


def final_projection(graph, kernels):
    """Bind the last external matmul allocation to the caller's single-row output."""
    calls = list(
        re.finditer(
            r"^    (%\w+) = call @(k3_(?:q4_plain|moe)_m1_\w+)\(([^)]*)\) : \(([^)]*)\) -> tensor<1x(\d+)xf32>$",
            graph,
            re.M,
        )
    )
    last = calls[-1]
    value, sym, args, types, width = last.groups()
    dest = f"memref<1x1x{width}xf32>"
    # Only reshapes of this projection may follow; all math must be preserved.
    tail = graph[last.end() :].split("    return\n")[0]
    ops = [l.strip() for l in tail.splitlines() if l.strip()]
    if not all(
        " = tosa.const_shape " in l
        or " = tosa.reshape " in l
        or "bufferization.materialize_in_destination " in l
        for l in ops
    ):
        raise ValueError("unexpected ops after " + sym)
    if not any(" = tosa.reshape " + value + "," in l for l in ops):
        raise ValueError("final projection is not the returned row")
    graph = (
        graph[: last.start()]
        + f"    call @{sym}({args}, %dest) : ({types}, {dest}) -> ()\n"
        + graph[last.end() + len(tail) :]
    )
    graph, n = re.subn(
        r"(func.func private @" + sym + r"\([^\n]+?)\) -> tensor<1x\d+xf32>",
        r"\1, " + dest + ")",
        graph,
    )
    if n != 1:
        raise ValueError("missing projection declaration: " + sym)

    def bind(m):
        body = m[0]
        mem = one(r"^    return (%\w+) : memref<1x\d+xf32>$", body)
        ty = f"memref<1x{width}xf32>"
        body = body.replace(
            ") -> " + ty + " attributes", ", %dest: " + dest + ") attributes", 1
        )
        body, n = re.subn(
            re.escape(mem) + r" = memref.alloc\(\) alignment = \d+ : " + ty,
            mem
            + " = memref.collapse_shape %dest [[0, 1], [2]] : "
            + dest
            + " into "
            + ty,
            body,
            count=1,
        )
        if n != 1:
            raise ValueError("projection result ownership changed")
        return body.replace("return " + mem + " : " + ty, "return")

    kernels, n = re.subn(
        r"^  func.func @" + sym + r"\([^\n]+\{\n.*?^  \}",
        bind,
        kernels,
        flags=re.S | re.M,
    )
    if n != 1:
        raise ValueError("missing projection definition: " + sym)
    return graph, kernels
