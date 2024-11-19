class HammingExtension:
    def __init__(self, code, r):
        self.code = code
        self.r = r

    def correct2(self):
        n = len(self.code)
        err = 0
        parity_check = 0

        # Определяем позицию ошибки (если она есть)
        for i in range(self.r):
            pos = 1 << i
            sum_bits = 0

            for j in range(n):
                if (j + 1) & pos:
                    sum_bits ^= self.code[j]

            if sum_bits != 0:
                err += pos
                parity_check = 1

        # Проверка на наличие ошибок
        if err == 0:
            print("Ошибок не обнаружено.")
        elif parity_check and self.detect2(err, parity_check):
            print(f"Обнаружена двойная ошибка в позициях: {err} и {parity_check}. Исправление невозможно.")
        else:
            print(f"Обнаружена одиночная ошибка в позиции: {err - 1}. Исправление...")
            self.code[err - 1] ^= 1

    def detect2(self, error1, error2):
        parity_check = 0
        for bit in self.code:
            parity_check ^= bit

        if parity_check:
            error2 = error1 ^ parity_check
            return True
        return False

# Пример использования
code = [1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1]  # Пример закодированного сообщения (допустим, с 2 ошибками)
r = 4  # Количество проверочных битов

hamming = HammingExtension(code, r)
hamming.correct2()
print("Исправленный код:", hamming.code)
