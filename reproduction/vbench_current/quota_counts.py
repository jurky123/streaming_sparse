GRID=[(t,s) for t in (5,8,10,12,15,20) for s in tuple(i/8 for i in range(9))]+[('full',5),('full',10)]
def counts(p):
    hist=p-96
    result=[]
    for total,share in GRID:
        if total=='full':
            a,b=72,hist*share//100
        else:
            n=(p-24)*total//100
            a=min(72,max(1,n-hist,round(n*share)))
            b=n-a
        result.append((a,b))
    return result

