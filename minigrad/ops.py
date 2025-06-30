from __future__ import annotations
from enum import Enum
from typing import NamedTuple,Tuple,Union,Any,Type

#TODO load op, unary ops, binary ops, movement ops, reduce ops, 
BinaryOps = Enum("BinaryOps",["MUL","ADD"])
UnaryOps = Enum("UnaryOps",["RELU","EXP"])
LoadOps = Enum("LoadOps",["FROMCPU"])
MovementOps = Enum("MovementOps",["EXPAND","RESHAPE"])
ReduceOps = Enum("ReduceOps",["SUM"])

Op = Union[LoadOps,UnaryOps,BinaryOps]
OpType = Union[Type[BinaryOps],Type[UnaryOps],Type[LoadOps]]
class LazyOp(NamedTuple):
    op: Op
    src: Tuple[Any,...] # LazyBuffer
    arg: Any = None


class GenericExecAST:

    @classmethod
    def exec_ast(cls, ast: LazyOp):
        # recursively execute the tree. rel_src = realized sources
        rel_srcs = [cls.exec_ast(x) if isinstance(x,LazyOp) else cls.exec_ast(x.op) for x in ast.src]
        if ast.op in LoadOps:
            ret = ast.arg
        elif ast.op in UnaryOps:
            ret = cls.unary_op(ast.op)
        elif ast.op in BinaryOps:
            ret = cls.binary_op(rel_srcs[0], ast.op, rel_srcs[1])
        return ret
