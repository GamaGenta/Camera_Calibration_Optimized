import csv
import re
from decimal import Decimal, InvalidOperation

def txt_to_csv(infile='input.txt', outfile='output.csv', delimiter=';'):
    with open(infile, 'r', encoding='utf-8') as fin, \
         open(outfile, 'w', encoding='utf-8', newline='') as fout:
        writer = csv.writer(fout, delimiter=delimiter, quoting=csv.QUOTE_MINIMAL)
        for line in fin:
            if line.strip() == '':
                writer.writerow([])          # leere Zeile erhalten
            else:
                tokens = line.split()        # splittet an beliebiger Whitespace
                writer.writerow(tokens)
def txt_to_csv_multi_space(infile='input.txt', outfile='output.csv', delimiter=';'):
    splitter = re.compile(r' {2,}')  # 2 oder mehr Leerzeichen
    with open(infile, 'r', encoding='utf-8') as fin, \
         open(outfile, 'w', encoding='utf-8', newline='') as fout:
        writer = csv.writer(fout, delimiter=delimiter, quoting=csv.QUOTE_MINIMAL)
        for line in fin:
            s = line.rstrip('\r\n')
            if s.strip() == '':                # leere oder nur-whitespace-Zeilen erhalten
                writer.writerow([])
                continue
            parts = [p.strip() for p in splitter.split(s)]  # Ränder entfernen, aber interne Einzel-Leerzeichen bleiben
            parts = [p for p in parts if p != '']           # leere Tokens am Rand entfernen
            writer.writerow(parts)

def txt_to_csv_multi_space_decimal(infile='input.txt', outfile='output.csv', delimiter=';'):
    splitter = re.compile(r' {2,}')  # 2 oder mehr Leerzeichen
    num_re = re.compile(r'^[+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?$')  # Zahl inkl. Exponent
    with open(infile, 'r', encoding='utf-8') as fin, \
         open(outfile, 'w', encoding='utf-8', newline='') as fout:
        writer = csv.writer(fout, delimiter=delimiter, quoting=csv.QUOTE_MINIMAL)
        for line in fin:
            s = line.rstrip('\r\n')
            if s.strip() == '':
                writer.writerow([])  # leere Zeile erhalten
                continue
            parts = [p.strip() for p in splitter.split(s)]
            out = []
            for p in parts:
                if p == '':
                    continue
                if num_re.match(p):
                    try:
                        d = Decimal(p)
                        out.append(format(d, 'f'))  # schreibt z.B. 3.86e+05 -> 386000
                    except InvalidOperation:
                        out.append(p)
                else:
                    out.append(p)
            writer.writerow(out)

def txt_to_csv_multi_space_decimal_comma(infile='input.txt', outfile='output.csv', delimiter=';'):
    splitter = re.compile(r' {2,}')
    num_re = re.compile(r'^[+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?$')
    with open(infile, 'r', encoding='utf-8') as fin, \
         open(outfile, 'w', encoding='utf-8', newline='') as fout:
        writer = csv.writer(fout, delimiter=delimiter, quoting=csv.QUOTE_MINIMAL)
        for line in fin:
            s = line.rstrip('\r\n')
            if s.strip() == '':
                writer.writerow([])
                continue
            parts = [p.strip() for p in splitter.split(s)]
            out = []
            for p in parts:
                if p == '':
                    continue
                if num_re.match(p):
                    try:
                        d = Decimal(p)
                        sval = format(d, 'f')    # ohne Exponent, z.B. "386000" oder "3.14"
                        sval = sval.replace('.', ',')  # Punkt -> Komma
                        out.append(sval)
                    except InvalidOperation:
                        out.append(p)
                else:
                    out.append(p)
            writer.writerow(out)

def main():
    # Beispielaufruf
    # txt_to_csv('costs 3 cam.txt', 'costs 3 cam.csv', delimiter=';')
    # txt_to_csv_multi_space('costs 3 cam.txt', 'costs 3 cam_multi_space.csv', delimiter=';')
    # txt_to_csv_multi_space('costs 4 cam.txt', 'costs 4 cam_multi_space.csv', delimiter=';')
    txt_to_csv_multi_space_decimal_comma('costs 3 cam.txt', 'costs 3 cam_multi_space_decimal.csv', delimiter=';')
    txt_to_csv_multi_space_decimal_comma('costs 4 cam.txt', 'costs 4 cam_multi_space_decimal.csv', delimiter=';')

if __name__ == "__main__":
    main()