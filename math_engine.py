import hashlib
import re
import numpy as np
from config import HAS_TORCH, DEVICE

if HAS_TORCH:
    import torch

# Константы парсера
VAR_Z, VAR_C, CONST = 0, 1, 2
OP_ADD, OP_SUB, OP_MUL, OP_DIV, OP_POW = 3, 4, 5, 6, 7
OP_SIN, OP_COS, OP_EXP, OP_LN, OP_ABS, OP_CONJ, OP_INV, OP_SIGM = 8, 9, 10, 11, 12, 13, 14, 15

EPS_REG = 1e-20

# Декодер энтропии для генерации формул
class EntropyDecoder:
    def __init__(self, seed_int: int):
        self.state = seed_int.to_bytes(64, 'big')
        self.buffer = []
        self.pointer = 0

    def _refresh_entropy(self):
        self.state = hashlib.sha512(self.state).digest()
        self.buffer.extend(list(self.state))
        self.pointer = 0

    def get_next_byte(self) -> int:
        if not self.buffer or self.pointer >= len(self.buffer):
            self._refresh_entropy()
        val = self.buffer[self.pointer]
        self.pointer += 1
        return val

    def get_float(self) -> float:
        b0 = self.get_next_byte()
        b1 = self.get_next_byte()
        b2 = self.get_next_byte()
        b3 = self.get_next_byte()
        val = (b0 << 24) | (b1 << 16) | (b2 << 8) | b3
        return val / 4294967296.0

def generate_ast(decoder, depth, max_depth):
    p_terminal = min(1.0, (depth - 1) / (max_depth - 1)) if depth > 1 else 0.0
    if decoder.get_float() < p_terminal:
        r = decoder.get_float()
        if r < 0.45: return {"type": "terminal", "opcode": VAR_Z, "val": None}
        elif r < 0.85: return {"type": "terminal", "opcode": VAR_C, "val": None}
        else:
            real = -1.5 + 3.0 * decoder.get_float()
            imag = -1.5 + 3.0 * decoder.get_float()
            return {"type": "terminal", "opcode": CONST, "val": complex(real, imag)}
    else:
        r_op = decoder.get_float()
        if r_op < 0.35:
            op_list = [OP_SIN, OP_COS, OP_EXP, OP_LN, OP_ABS, OP_CONJ, OP_INV, OP_SIGM]
            op = op_list[decoder.get_next_byte() % len(op_list)]
            child = generate_ast(decoder, depth + 1, max_depth)
            return {"type": "unary", "opcode": op, "child": child}
        else:
            op_list = [OP_ADD, OP_SUB, OP_MUL, OP_MUL, OP_DIV, OP_POW]
            op = op_list[decoder.get_next_byte() % len(op_list)]
            left = generate_ast(decoder, depth + 1, max_depth)
            right = generate_ast(decoder, depth + 1, max_depth)
            return {"type": "binary", "opcode": op, "left": left, "right": right}

def ast_to_rpn(node, rpn):
    if node["type"] == "terminal":
        rpn.append((0, node["opcode"], node["val"]))
    elif node["type"] == "unary":
        ast_to_rpn(node["child"], rpn)
        rpn.append((1, node["opcode"], None))
    elif node["type"] == "binary":
        ast_to_rpn(node["left"], rpn)
        ast_to_rpn(node["right"], rpn)
        rpn.append((2, node["opcode"], None))

def validate_rpn(rpn):
    has_z = any(t == 0 and op == VAR_Z for t, op, _ in rpn)
    has_c = any(t == 0 and op == VAR_C for t, op, _ in rpn)
    num_ops = sum(1 for t, _, _ in rpn if t in (1, 2))
    return has_z and has_c and (num_ops >= 1)

def rpn_to_str(rpn):
    stack = []
    op_syms = {OP_ADD: "+", OP_SUB: "-", OP_MUL: "*", OP_DIV: "/", OP_POW: "^"}
    unary_names = {
        OP_SIN: "sin", OP_COS: "cos", OP_EXP: "exp", OP_LN: "ln",
        OP_ABS: "abs_rect", OP_CONJ: "conj", OP_INV: "inv", OP_SIGM: "sigmoid"
    }
    for t_type, op, val in rpn:
        if t_type == 0:
            if op == VAR_Z: stack.append("Z")
            elif op == VAR_C: stack.append("C")
            elif op == CONST: stack.append(f"({val.real:.2f}+{val.imag:.2f}j)")
        elif t_type == 1:
            arg = stack.pop()
            stack.append(f"{unary_names[op]}({arg})")
        elif t_type == 2:
            arg2 = stack.pop()
            arg1 = stack.pop()
            stack.append(f"({arg1}{op_syms[op]}{arg2})")
    return stack[0] if stack else "Z"

def parse_infix_to_rpn(expr_str):
    expr_str = expr_str.replace(" ", "").replace("abs_rect", "abs")
    token_specification = [
        ('COMPLEX', r'\d+(?:\.\d*)?[jJ]|\d+(?:\.\d*)?[\+\-]\d+(?:\.\d*)?[jJ]'), 
        ('NUMBER',  r'\d+(?:\.\d*)?'),                                       
        ('IDENT',   r'[a-zA-Z_][a-zA-Z0-9_]*'),                              
        ('OP',      r'[\+\-\*\/\^\(\),]')                                    
    ]
    tok_regex = '|'.join(f'(?P<{name}>{pattern})' for name, pattern in token_specification)
    tokens = []
    for mo in re.finditer(tok_regex, expr_str):
        tokens.append((mo.lastgroup, mo.group()))
    
    output = []
    op_stack = []
    prec = {'+': 2, '-': 2, '*': 3, '/': 3, '^': 4}
    assoc = {'+': 'L', '-': 'L', '*': 'L', '/': 'L', '^': 'R'}
    funcs = {'sin', 'cos', 'exp', 'ln', 'abs', 'conj', 'inv', 'sigmoid'}
    prev_token_type = 'START'
    
    for kind, val in tokens:
        if kind == 'NUMBER' or kind == 'COMPLEX':
            c_val = complex(val.replace('j', 'j').replace('J', 'j')) if 'j' in val.lower() else complex(float(val))
            output.append((0, CONST, c_val))
            prev_token_type = 'NUMBER'
        elif kind == 'IDENT':
            val_lower = val.lower()
            if val_lower == 'z':
                output.append((0, VAR_Z, None))
                prev_token_type = 'IDENT'
            elif val_lower == 'c':
                output.append((0, VAR_C, None))
                prev_token_type = 'IDENT'
            elif val_lower in funcs:
                op_stack.append(('FUNC', val_lower))
                prev_token_type = 'IDENT'
            else:
                raise ValueError(f"Неизвестная функция: {val}")
        elif val == '(':
            op_stack.append(('PAREN', '('))
            prev_token_type = 'LPAREN'
        elif val == ')':
            while op_stack and op_stack[-1][1] != '(':
                output.append(op_stack.pop())
            if not op_stack:
                raise ValueError("Несогласованные скобки")
            op_stack.pop() 
            if op_stack and op_stack[-1][0] == 'FUNC':
                output.append(op_stack.pop())
            prev_token_type = 'RPAREN'
        elif val == ',':
            while op_stack and op_stack[-1][1] != '(':
                output.append(op_stack.pop())
            prev_token_type = 'COMMA'
        elif kind == 'OP':
            is_unary = False
            if val in ('+', '-'):
                if prev_token_type in ('START', 'LPAREN', 'COMMA', 'OP'):
                    is_unary = True
            if is_unary:
                if val == '-':
                    output.append((0, CONST, 0j))
                    while (op_stack and op_stack[-1][0] == 'OP' and 
                           (assoc['-'] == 'L' and prec['-'] <= prec.get(op_stack[-1][1], 0) or
                            assoc['-'] == 'R' and prec['-'] < prec.get(op_stack[-1][1], 0))):
                        output.append(op_stack.pop())
                    op_stack.append(('OP', '-'))
            else:
                while (op_stack and op_stack[-1][0] == 'OP' and 
                       (assoc[val] == 'L' and prec[val] <= prec.get(op_stack[-1][1], 0) or
                        assoc[val] == 'R' and prec[val] < prec.get(op_stack[-1][1], 0))):
                    output.append(op_stack.pop())
                op_stack.append(('OP', val))
            prev_token_type = 'OP'
            
    while op_stack:
        top_type, top_val = op_stack.pop()
        if top_val in ('(', ')'):
            raise ValueError("Несогласованные скобки")
        output.append((top_type, top_val))
        
    rpn_tokens = []
    for token in output:
        item_type = token[0]
        if item_type == 0:
            # token уже является валидным RPN-кортежем (0, OPCODE, VAL)
            rpn_tokens.append(token)
        elif item_type == 'OP':
            item_val = token[1]
            op_code = {'+': OP_ADD, '-': OP_SUB, '*': OP_MUL, '/': OP_DIV, '^': OP_POW}[item_val]
            rpn_tokens.append((2, op_code, None))
        elif item_type == 'FUNC':
            item_val = token[1]
            op_code = {
                'sin': OP_SIN, 'cos': OP_COS, 'exp': OP_EXP, 'ln': OP_LN,
                'abs': OP_ABS, 'conj': OP_CONJ, 'inv': OP_INV, 'sigmoid': OP_SIGM
            }[item_val]
            rpn_tokens.append((1, op_code, None))
    return rpn_tokens

def evaluate_rpn_pytorch(rpn, Z, C, device, use_double=False):
    stack = []
    torch_complex = torch.complex128 if use_double else torch.complex64
    for t_type, op, val in rpn:
        if t_type == 0:
            if op == VAR_Z: stack.append(Z)
            elif op == VAR_C: stack.append(C)
            elif op == CONST: stack.append(torch.tensor(val, dtype=torch_complex, device=device))
        elif t_type == 1:
            A = stack.pop()
            if op == OP_SIN:
                stack.append(torch.sin(torch.complex(A.real, torch.clamp(A.imag, -15.0, 15.0))))
            elif op == OP_COS:
                stack.append(torch.cos(torch.complex(A.real, torch.clamp(A.imag, -15.0, 15.0))))
            elif op == OP_EXP:
                stack.append(torch.exp(torch.complex(torch.clamp(A.real, -15.0, 15.0), A.imag)))
            elif op == OP_LN:
                mag_sq = A.real**2 + A.imag**2 + EPS_REG
                stack.append(torch.complex(0.5 * torch.log(mag_sq), torch.atan2(A.imag, A.real)))
            elif op == OP_ABS:
                stack.append(torch.complex(torch.abs(A.real), torch.abs(A.imag)))
            elif op == OP_CONJ:
                stack.append(torch.conj(A))
            elif op == OP_INV:
                denom = A.real**2 + A.imag**2 + EPS_REG
                stack.append(torch.complex(A.real / denom, -A.imag / denom))
            elif op == OP_SIGM:
                real_clamped = torch.clamp(A.real, -15.0, 15.0)
                denom = 1.0 + torch.exp(-real_clamped) * torch.complex(torch.cos(-A.imag), torch.sin(-A.imag))
                d_mag_sq = denom.real**2 + denom.imag**2 + EPS_REG
                stack.append(torch.complex(denom.real / d_mag_sq, -denom.imag / d_mag_sq))
        elif t_type == 2:
            B = stack.pop()
            A = stack.pop()
            if op == OP_ADD: stack.append(A + B)
            elif op == OP_SUB: stack.append(A - B)
            elif op == OP_MUL: stack.append(A * B)
            elif op == OP_DIV:
                denom = B.real**2 + B.imag**2 + EPS_REG
                stack.append(torch.complex((A.real * B.real + A.imag * B.imag) / denom, (A.imag * B.real - A.real * B.imag) / denom))
            elif op == OP_POW:
                mag_sq = A.real**2 + A.imag**2 + EPS_REG
                ln_A = torch.complex(0.5 * torch.log(mag_sq), torch.atan2(A.imag, A.real))
                prod = B * ln_A
                stack.append(torch.exp(torch.complex(torch.clamp(prod.real, -15.0, 15.0), prod.imag)))
                
    Z_next = stack[0]
    anomalies = torch.isnan(Z_next) | torch.isinf(Z_next)
    if torch.any(anomalies):
        Z_next = torch.where(anomalies, torch.tensor(1e5 + 0j, dtype=torch_complex, device=device), Z_next)
    return Z_next

def evaluate_rpn_numpy(rpn, Z, C, use_double=False):
    stack = []
    complex_dtype = np.complex128 if use_double else np.complex64
    for t_type, op, val in rpn:
        if t_type == 0:
            if op == VAR_Z: stack.append(Z)
            elif op == VAR_C: stack.append(C)
            elif op == CONST: stack.append(complex_dtype(val))
        elif t_type == 1:
            A = stack.pop()
            with np.errstate(invalid='ignore', over='ignore'):
                if op == OP_SIN:
                    stack.append(np.sin(np.real(A) + 1j * np.clip(np.imag(A), -15.0, 15.0)))
                elif op == OP_COS:
                    stack.append(np.cos(np.real(A) + 1j * np.clip(np.imag(A), -15.0, 15.0)))
                elif op == OP_EXP:
                    stack.append(np.exp(np.clip(np.real(A), -15.0, 15.0) + 1j * np.imag(A)))
                elif op == OP_LN:
                    mag_sq = np.real(A)**2 + np.imag(A)**2 + EPS_REG
                    stack.append(0.5 * np.log(mag_sq) + 1j * np.arctan2(np.imag(A), np.real(A)))
                elif op == OP_ABS:
                    stack.append(np.abs(np.real(A)) + 1j * np.abs(np.imag(A)))
                elif op == OP_CONJ:
                    stack.append(np.conj(A))
                elif op == OP_INV:
                    denom = np.real(A)**2 + np.imag(A)**2 + EPS_REG
                    stack.append(np.real(A)/denom - 1j*np.imag(A)/denom)
                elif op == OP_SIGM:
                    real_clamped = np.clip(np.real(A), -15.0, 15.0)
                    denom = 1.0 + np.exp(-real_clamped) * (np.cos(-np.imag(A)) + 1j * np.sin(-np.imag(A)))
                    d_mag_sq = np.real(denom)**2 + np.imag(denom)**2 + EPS_REG
                    stack.append(np.real(denom)/d_mag_sq - 1j*np.imag(denom)/d_mag_sq)
        elif t_type == 2:
            B = stack.pop()
            A = stack.pop()
            with np.errstate(invalid='ignore', over='ignore'):
                if op == OP_ADD: stack.append(A + B)
                elif op == OP_SUB: stack.append(A - B)
                elif op == OP_MUL: stack.append(A * B)
                elif op == OP_DIV:
                    denom = np.real(B)**2 + np.imag(B)**2 + EPS_REG
                    stack.append((np.real(A)*np.real(B) + np.imag(A)*np.imag(B))/denom + 1j*(np.imag(A)*np.real(B) - np.real(A)*np.imag(B))/denom)
                elif op == OP_POW:
                    mag_sq = np.real(A)**2 + np.imag(A)**2 + EPS_REG
                    ln_A = 0.5 * np.log(mag_sq) + 1j * np.arctan2(np.imag(A), np.real(A))
                    prod = B * ln_A
                    stack.append(np.exp(np.clip(np.real(prod), -15.0, 15.0) + 1j * np.imag(prod)))
                    
    Z_next = stack[0]
    anomalies = np.isnan(Z_next) | np.isinf(Z_next)
    if np.any(anomalies):
        Z_next = np.where(anomalies, complex_dtype(1e5 + 0j), Z_next)
    return Z_next