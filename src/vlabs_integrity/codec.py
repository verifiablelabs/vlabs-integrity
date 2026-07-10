"""Bounded tagged JSON codec that preserves verifier argument/result types."""
from __future__ import annotations

import math

_DEPTH = 64
_NODES = 100_000
_STRING = 1_000_000
_INT_BITS = 4096


def _take(budget: list[int], depth: int) -> None:
    budget[0] += 1
    if depth > _DEPTH or budget[0] > _NODES:
        raise ValueError("protocol value exceeds bounds")


def _encode(value: object, depth: int, budget: list[int]) -> dict:
    _take(budget, depth)
    kind = type(value)
    if value is None:
        return {"t": "null"}
    if kind is bool:
        return {"t": "bool", "v": value}
    if kind is int and value.bit_length() <= _INT_BITS:
        return {"t": "int", "v": value}
    if kind is float and math.isfinite(value):
        return {"t": "float", "v": value}
    if kind is str and len(value) <= _STRING:
        return {"t": "str", "v": value}
    if kind in (list, tuple):
        return {
            "t": "list" if kind is list else "tuple",
            "v": [_encode(item, depth + 1, budget) for item in value],
        }
    if kind is dict:
        if any(type(key) is not str or len(key) > _STRING for key in value):
            raise TypeError("only bounded string dictionary keys are supported")
        return {
            "t": "dict",
            "v": [[key, _encode(value[key], depth + 1, budget)] for key in sorted(value)],
        }
    raise TypeError("unsupported protocol value")


def encode_value(value: object) -> dict:
    return _encode(value, 0, [0])


def _decode(value: object, depth: int, budget: list[int]) -> object:
    _take(budget, depth)
    if type(value) is not dict or type(value.get("t")) is not str:
        raise ValueError("invalid tagged value")
    tag = value["t"]
    if tag == "null":
        if set(value) != {"t"}:
            raise ValueError("invalid null")
        return None
    if set(value) != {"t", "v"}:
        raise ValueError("invalid tagged shape")
    payload = value["v"]
    if tag == "bool" and type(payload) is bool:
        return payload
    if tag == "int" and type(payload) is int and payload.bit_length() <= _INT_BITS:
        return payload
    if tag == "float" and type(payload) is float and math.isfinite(payload):
        return payload
    if tag == "str" and type(payload) is str and len(payload) <= _STRING:
        return payload
    if tag in {"list", "tuple"} and type(payload) is list:
        items = [_decode(item, depth + 1, budget) for item in payload]
        return items if tag == "list" else tuple(items)
    if tag == "dict" and type(payload) is list:
        result = {}
        for pair in payload:
            if (
                type(pair) is not list
                or len(pair) != 2
                or type(pair[0]) is not str
                or len(pair[0]) > _STRING
                or pair[0] in result
            ):
                raise ValueError("invalid dictionary entry")
            result[pair[0]] = _decode(pair[1], depth + 1, budget)
        return result
    raise ValueError("invalid tagged payload")


def decode_value(value: object) -> object:
    return _decode(value, 0, [0])


WORKER_CODEC_SOURCE = r'''
import math as _cm
_CD=64
_CN=100000
_CS=1000000
_CI=4096
def _ct(_b,_d):
    _b[0]+=1
    if _d>_CD or _b[0]>_CN: raise ValueError("protocol value exceeds bounds")
def _cei(_x,_d,_b):
    _ct(_b,_d)
    _k=type(_x)
    if _x is None: return {"t":"null"}
    if _k is bool: return {"t":"bool","v":_x}
    if _k is int and _x.bit_length()<=_CI: return {"t":"int","v":_x}
    if _k is float and _cm.isfinite(_x): return {"t":"float","v":_x}
    if _k is str and len(_x)<=_CS: return {"t":"str","v":_x}
    if _k in (list,tuple): return {"t":"list" if _k is list else "tuple","v":[_cei(_i,_d+1,_b) for _i in _x]}
    if _k is dict:
        if any(type(_q) is not str or len(_q)>_CS for _q in _x): raise TypeError("only bounded string dictionary keys are supported")
        return {"t":"dict","v":[[_q,_cei(_x[_q],_d+1,_b)] for _q in sorted(_x)]}
    raise TypeError("unsupported protocol value")
def _encode_value(_x): return _cei(_x,0,[0])
def _cdi(_x,_d,_b):
    _ct(_b,_d)
    if type(_x) is not dict or type(_x.get("t")) is not str: raise ValueError("invalid tagged value")
    _t=_x["t"]
    if _t=="null":
        if set(_x)!={"t"}: raise ValueError("invalid null")
        return None
    if set(_x)!={"t","v"}: raise ValueError("invalid tagged shape")
    _v=_x["v"]
    if _t=="bool" and type(_v) is bool: return _v
    if _t=="int" and type(_v) is int and _v.bit_length()<=_CI: return _v
    if _t=="float" and type(_v) is float and _cm.isfinite(_v): return _v
    if _t=="str" and type(_v) is str and len(_v)<=_CS: return _v
    if _t in {"list","tuple"} and type(_v) is list:
        _a=[_cdi(_i,_d+1,_b) for _i in _v]
        return _a if _t=="list" else tuple(_a)
    if _t=="dict" and type(_v) is list:
        _r={}
        for _p in _v:
            if type(_p) is not list or len(_p)!=2 or type(_p[0]) is not str or len(_p[0])>_CS or _p[0] in _r: raise ValueError("invalid dictionary entry")
            _r[_p[0]]=_cdi(_p[1],_d+1,_b)
        return _r
    raise ValueError("invalid tagged payload")
def _decode_value(_x): return _cdi(_x,0,[0])
'''

__all__ = ["decode_value", "encode_value", "WORKER_CODEC_SOURCE"]
