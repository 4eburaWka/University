class HammingBasic:
    def __init__(self, M, r, status):
        self.M = M
        self.r = r
        self.code = []
        self.encode(status)

    def encode(self, status):
        k = self.calculate()
        n = k + self.r

        if status:
            self.code = [0] * 8
        else:
            self.code = [0] * n

        j = 0
        for i in range(n):
            if self.is_powered(i + 1):
                continue
            self.code[i] = (self.M >> j) & 1
            j += 1

        for i in range(self.r):
            pos = (1 << i) - 1
            sum_bits = 0
            for j in range(n):
                if (j + 1) & (1 << i):
                    sum_bits ^= self.code[j]
            self.code[pos] = sum_bits

    def correct(self):
        n = len(self.code) - 1
        err = 0

        for i in range(self.r):
            pos = (1 << i)
            sum_bits = 0

            for j in range(n):
                if (j + 1) & pos:
                    sum_bits ^= self.code[j]

            if sum_bits != 0:
                err += pos

        if err != 0:
            print(f"Err pos: {err - 1}")
            self.code[err - 1] ^= 1
        else:
            print("No err")

    def decode(self):
        decoded = 0
        j = 0

        for i in range(len(self.code)):
            if not self.is_powered(i + 1):
                decoded |= (self.code[i] << j)
                j += 1

        return decoded

    def print_code(self):
        print("".join(str(bit) for bit in reversed(self.code)))

    def set_code(self, i):
        if i >= len(self.code) or i < 0:
            print("wrong index")
            return
        self.code[i] ^= 1

    @staticmethod
    def is_powered(x):
        return x and not (x & (x - 1))

    def calculate(self):
        k = 0
        m = self.M
        while m > 0:
            m >>= 1
            k += 1
        return k


class HammingExtension(HammingBasic):
    def correct2(self):
        n = len(self.code)
        err = 0
        parity_check = 0

        for i in range(self.r):
            pos = (1 << i)
            sum_bits = 0

            for j in range(n):
                if (j + 1) & pos:
                    sum_bits ^= self.code[j]

            if sum_bits != 0:
                err += pos
                parity_check = 1

        if err == 0:
            print("No error detected.")
        elif parity_check and self.detect2(err):
            print(
                f"Double error detected at positions: {err} and {parity_check}. Correction not possible.")
        else:
            print(
                f"Single error detected at position: {err - 1}. Correcting...")
            self.code[err - 1] ^= 1

    def detect2(self, error1):
        parity_check = 0
        for bit in self.code:
            parity_check ^= bit

        if parity_check:
            error2 = error1 ^ parity_check
            return True
        return False


# Пример использования
M = 0b1011011  # Пример исходного сообщения (7 бит)
r = 4  # Количество проверочных битов

# Создаем расширенный код Хэмминга
hamming = HammingExtension(M, r, status=False)
hamming.print_code()  # Печать закодированного сообщения

# Допускаем ошибку для теста
hamming.set_code(3)

# Корректируем ошибки
hamming.correct2()
hamming.print_code()  # Печать исправленного кода

# Декодируем сообщение
decoded = hamming.decode()
print(f"Декодированное сообщение: {bin(decoded)}")
