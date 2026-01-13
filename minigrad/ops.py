from __future__ import annotations
from enum import Enum
from typing import NamedTuple,Tuple,Union,Any,Type
import sys

#TODO load op, unary ops, binary ops, movement ops, reduce ops, 
BinaryOps = Enum("BinaryOps",["MUL","ADD","SUB","POW","DIV","CMPEQ"])
UnaryOps = Enum("UnaryOps",["RELU","EXP","RECIPROCAL","LOG","NEG","SIGN"])
LoadOps = Enum("LoadOps",["FROMCPU"])
MovementOps = Enum("MovementOps",["EXPAND","RESHAPE","PERMUTE","STRIDED","PAD","SHRINK","MASKED_FILL"])
ReduceOps = Enum("ReduceOps",["SUM","MAX"])
ProcessingOps = Enum("ProcessingOps",["CONV1D","CONV"])

sys.setrecursionlimit(1000)

Op = Union[LoadOps,UnaryOps,BinaryOps]
OpType = Union[Type[BinaryOps],Type[UnaryOps],Type[LoadOps]]

# stores lazy sources (parents), operation type, other arguments for output lazy buffer
class LazyOp(NamedTuple):
    op: Op
    src: Tuple[Any,...] # LazyBuffer
    arg: Any = None

#TODO visualiztion of tree execution
class GenericExecAST:
    @classmethod
    def exec_ast(cls: GenericExecAST, ast: LazyOp, preprocess= lambda x: x):
        # recursively execute the tree from srcs to the final node.
        # rel_srcs = realized sources
        # rel_srcs = [cls.exec_ast(x) if isinstance(x,LazyOp) else x.realize() for x in ast.src]
        #TODO minimize the recursion using below statement, explict call of exe_ast on lazybuffer
        # rel_srcs = [cls.exec_ast(x,preprocess) if isinstance(x,LazyOp) else preprocess(x) for x in ast.src]

        # If we got a LazyBuffer, realize it
        if hasattr(ast, "realize"):
            return ast.realize()

        # Otherwise must be LazyOp
        assert isinstance(ast, LazyOp)

        rel_srcs = [cls.exec_ast(x) for x in ast.src]

        if ast.op in LoadOps:
            ret = cls.fromCPU(ast.arg)
        elif ast.op in UnaryOps:
            # TODO shape assert
            ret = cls.unary_op(rel_srcs[0],ast.op)
        elif ast.op in BinaryOps:
            assert rel_srcs[0].shape==rel_srcs[1].shape
            ret = cls.binary_op(rel_srcs[0], ast.op, rel_srcs[1])
        elif ast.op in ReduceOps:
            ret = cls.reduce_op(rel_srcs[0],ast.op,*ast.arg)
        elif ast.op in MovementOps:
            ret = cls.movement_op(rel_srcs[0],ast.op,ast.arg)
        elif ast.op in ProcessingOps:
            ret = cls.processing_op(rel_srcs[0],ast.op,rel_srcs[1],ast.arg)
        else:
            raise Exception("Unknown Op")
        return ret
