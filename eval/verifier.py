from math_verify import parse, verify

def score(gold_solution: str, text: str, finish: str) -> float:
  if finish == "length":          
    return 0.0
  gold = parse(gold_solution)
  if not gold:  
    raise ValueError("gold parse failed")
  pred = parse(text)
  return float(verify(gold, pred))