# hello scripts...
import sys
sys.path.append("./")

# testando funcao get_fund

from utilProcessing import get_fund

fundamentais = get_fund()
print(fundamentais)