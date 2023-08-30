import Crypto.Util.number as numb
import random
import math

"""Shamir标准加法/乘法"""
"""Usage
    两个恢复：ShamirRecover
    多个恢复：MultiShamirRecover
    底层没有区别，只是为了区别用法
    
    加密见函数名即为使用方式
"""


# 求逆函数
def oj(a, n):
    a = a % n
    s = [0, 1]
    while a != 1:
        if a == 0:
            return 0
        q = n // a
        t = n % a
        n = a
        a = t
        s += [s[-2] - q * s[-1]]
    return s[-1]


# max_length 为p的长度，同时也是秘密的最大长度
# secret_is_text =0 默认输入时文本， 非0时认为是数字
# p 默认为0， 会根据max_length 自动生成，不为0时直接使用，需要保证p为素数， 函数内没有素性检验
def create(w_in, t_in, secret, max_length=513, secret_is_text=0, p=0):
    if not p:
        p = numb.getPrime(max_length)

    w = w_in
    t = t_in
    s = secret

    if secret_is_text:
        s = numb.bytes_to_long(s.encode("utf-8"))
    else:
        try:
            s = int(s)
        except Exception as e:
            s = numb.bytes_to_long(s.encode("utf-8"))

    x_list = list()
    a_list = list()
    i = w
    while i > 0:
        x = random.randint(p // 2, p)  # 该范围没有特定限制，如果想让xi,yi取小一点儿的话可把范围写小点儿，但是要大于w
        if x not in x_list:
            x_list.append(x)
            i -= 1
    for a in range(t):
        a_list.append(random.randint(p // 2, p))  # 同上

    result = list()
    for x in x_list:
        y = s
        for a_n in range(t):
            a = a_list[i]
            y += a * pow(x, i + 1, p)
        result.append((x, y))
    return t, p, result


# 0-num 1-text
def restore(p, information, get_text=0):
    x_list = list()
    y_list = list()
    for x, y in information:
        x_list.append(x)
        y_list.append(y)
    s = 0
    for x_i in range(len(x_list)):
        tmp_num = y_list[x_i]
        x_i_j = 1
        for x_j in range(len(x_list)):
            if x_i != x_j:
                tmp_num = tmp_num * (0 - x_list[x_j]) % p
                x_i_j *= x_list[x_i] - x_list[x_j]
        tmp_num = tmp_num * oj(x_i_j, p) % p
        s += tmp_num
    s = s % p
    return s


def ShamirAdd(a, b):
    Sum = float(a) + float(b)
    t, p, secret_set = create(2, 2, Sum)
    return p, secret_set


def ShamirRecover(p, secret_set):
    return restore(p, secret_set)


def MultiShamirAdd(rec, l=[]):
    n = len(l)
    Sum = 0
    for x in l:
        Sum += int(x)
    t, p, secret_set = create(n, rec, Sum)
    return p, secret_set


def MultiShamirRecover(p, secret_set):
    return restore(p, secret_set)


def ShamirMult(a, b):
    Sum = int(a) * int(b)
    t, p, secret_set = create(2, 2, Sum)
    return p, secret_set


def MultiShamirMult(rec, l=[]):
    n = len(l)
    Mult = 1
    for x in l:
        Mult *= int(x)
    t, p, secret_set = create(n, rec, Mult)
    return p, secret_set


def ShamirExp(base, index):
    if base == 0 and index == 0:
        print("input error -- base and index should not equal to 0 at the same time")
        return -1, []
    exp = math.pow(base, index)
    t, p, secret_set = create(2, 2, exp)
    return p, secret_set


def ShamirSqrt(x, precision=1e-8):
    left = 0
    right = x if x > 1 else 1  # 需要注意输入值是不是小于1
    mid = left + (right - left) / 2

    while mid * mid > x + precision or mid * mid < x - precision:

        if mid * mid > x + precision:  # 大于 目标值
            right = mid
        elif mid * mid < x - precision:
            left = mid
        mid = left + (right - left) / 2
    t, p, secret_set = create(2, 2, mid)
    return p, secret_set


if __name__ == '__main__':
    # bi_add（两个数相加）
    a = input("输入第一个数 : ")
    b = input("输入第二个数 : ")
    p, secret_set = ShamirAdd(a, b)
    res = ShamirRecover(p, secret_set)
    print(res)

    # mul_add（多个数相加）
    # memberList = [1, 2, 3]
    # p, secret_set = MultiShamirAdd(3, memberList)
    # res = MultiShamirRecover(p, secret_set)
    # print(res)

    # bi_mult（两个数相乘）
    # a = input("输入第一个数 : ")
    # b = input("输入第二个数 : ")
    # p, secret_set = ShamirMult(a, b)
    # res = ShamirRecover(p ,secret_set)
    # print(res)

    # mul_mult（多个数相乘）
    # memberList = [2, 3, 5]
    # p, secret_set = MultiShamirMult(3, memberList)
    # res = MultiShamirRecover(p, secret_set)
    # print(res)

    # exp(指数)
    # base = 3
    # index = 3
    # p, secret_set = ShamirExp(base, index)
    # res = ShamirRecover(p, secret_set)
    # print(res)

    # # sqrt(square root)
    # base = 36
    # p, secret_set = ShamirSqrt(base)
    # res = ShamirRecover(p, secret_set)
    # print(res)
