s = '我在这里》》》m here'
print(len(bytes(s.encode('utf-8'))))
codes = []
for code in s:
    codes.append(ord(code))
print(codes)
