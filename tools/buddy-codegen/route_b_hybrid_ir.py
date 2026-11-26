"""Destination-passing MLIR kernels for the Route B hybrid templates."""

from textwrap import dedent


def projection(output, width):
    weights = f"memref<{output}x{width}xf32>"
    x = f"memref<{width}xf32>"
    y = f"memref<{output}xf32>"
    return dedent(f"""\
      func.func @project_{output}_{width}(%w: {weights}, %x: {x}, %y: {y})
          attributes {{llvm.emit_c_interface}} {{
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c8 = arith.constant 8 : index
        %width = arith.constant {width} : index
        %output = arith.constant {output} : index
        %zero = arith.constant dense<0.0> : vector<8xf32>
        scf.for %i = %c0 to %output step %c1 {{
          %partial = scf.for %j = %c0 to %width step %c8
              iter_args(%acc = %zero) -> (vector<8xf32>) {{
            %a = vector.load %w[%i, %j] : {weights}, vector<8xf32>
            %b = vector.load %x[%j] : {x}, vector<8xf32>
            %product = arith.mulf %a, %b : vector<8xf32>
            %sum = arith.addf %acc, %product : vector<8xf32>
            scf.yield %sum : vector<8xf32>
          }}
          %dot = vector.reduction <add>, %partial : vector<8xf32> into f32
          memref.store %dot, %y[%i] : {y}
        }}
        return
      }}
    """)


def embedding(rows, width, vocab, dtype):
    weights = f"memref<{vocab}x{width}x{dtype}>"
    ids = f"memref<{rows}xi64>"
    out = f"memref<{rows}x{width}xf32>"
    value = "%element"
    widen = ""
    if dtype == "f16":
        value = "%widened"
        widen = "      %widened = arith.extf %element : f16 to f32\n"
    return dedent(f"""\
      func.func @gather_{rows}(%w: {weights}, %ids: {ids}, %out: {out})
          attributes {{llvm.emit_c_interface}} {{
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %rows = arith.constant {rows} : index
        %width = arith.constant {width} : index
        scf.for %i = %c0 to %rows step %c1 {{
          %id = memref.load %ids[%i] : {ids}
          %row = arith.index_cast %id : i64 to index
          scf.for %j = %c0 to %width step %c1 {{
            %element = memref.load %w[%row, %j] : {weights}
      {widen.rstrip()}
            memref.store {value}, %out[%i, %j] : {out}
          }}
        }}
        return
      }}
    """)


def delta_step(key_dim, value_dim):
    """HF recurrent rule, with lanes along the independent value axis."""
    s = f"memref<{key_dim}x{value_dim}xf32>"
    k = f"memref<{key_dim}xf32>"
    v = f"memref<{value_dim}xf32>"
    vec = "vector<8xf32>"
    return dedent(f"""\
      func.func @delta_step(%state: {s}, %q: {k}, %k: {k}, %v: {v}, %out: {v},
                            %beta: f32, %alpha: f32)
          attributes {{llvm.emit_c_interface}} {{
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c8 = arith.constant 8 : index
        %dk = arith.constant {key_dim} : index
        %dv = arith.constant {value_dim} : index
        %zero = arith.constant dense<0.0> : {vec}
        %decay = vector.broadcast %alpha : f32 to {vec}
        %rate = vector.broadcast %beta : f32 to {vec}
        scf.for %j = %c0 to %dv step %c8 {{
          %old = scf.for %i = %c0 to %dk step %c1
                    iter_args(%sum = %zero) -> ({vec}) {{
            %prior = vector.load %state[%i, %j] : {s}, {vec}
            %decayed = arith.mulf %prior, %decay : {vec}
            vector.store %decayed, %state[%i, %j] : {s}, {vec}
            %ki = memref.load %k[%i] : {k}
            %key = vector.broadcast %ki : f32 to {vec}
            %read = arith.mulf %decayed, %key : {vec}
            %next = arith.addf %sum, %read : {vec}
            scf.yield %next : {vec}
          }}
          %actual = vector.load %v[%j] : {v}, {vec}
          %error = arith.subf %actual, %old : {vec}
          %delta = arith.mulf %error, %rate : {vec}
          %result = scf.for %i = %c0 to %dk step %c1
                        iter_args(%sum = %zero) -> ({vec}) {{
            %ki = memref.load %k[%i] : {k}
            %key = vector.broadcast %ki : f32 to {vec}
            %decayed = vector.load %state[%i, %j] : {s}, {vec}
            %write = arith.mulf %key, %delta : {vec}
            %updated = arith.addf %decayed, %write : {vec}
            vector.store %updated, %state[%i, %j] : {s}, {vec}
            %qi = memref.load %q[%i] : {k}
            %query = vector.broadcast %qi : f32 to {vec}
            %read = arith.mulf %updated, %query : {vec}
            %next = arith.addf %sum, %read : {vec}
            scf.yield %next : {vec}
          }}
          vector.store %result, %out[%j] : {v}, {vec}
        }}
        return
      }}
    """)
