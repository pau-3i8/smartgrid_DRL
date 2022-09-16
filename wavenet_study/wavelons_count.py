
RES_LVL = 4

print('1 input')
wavelons = 1
for j in range(RES_LVL):
    wavelons += 2**j
    print(f'resolution {j}:', wavelons)
    
print('2 input')
wavelons = 1
for j in range(RES_LVL):
    wavelons += 2*2**j+2**(2*j)
    print(f'resolution {j}:', wavelons)
    
print('3 input')
wavelons = 1
for j in range(RES_LVL):
    wavelons += 3*2**j+3*2**(2*j)+2**(3*j)
    print(f'resolution {j}:', wavelons)
    
print('4 input')
wavelons = 1
for j in range(RES_LVL):
    wavelons += 4*2**j+6*2**(2*j)+4*2**(3*j)+2**(4*j)
    print(f'resolution {j}:', wavelons)
    
print('5 input')
wavelons = 1
for j in range(RES_LVL):
    wavelons += 5*2**j+10*2**(2*j)+10*2**(3*j)+5*2**(4*j)+2**(5*j)
    print(f'resolution {j}:', wavelons)
    
print('6 input')
wavelons = 1
for j in range(RES_LVL):
    wavelons += 6*2**j+15*2**(2*j)+20*2**(3*j)+15*2**(4*j)+6*2**(5*j)+2**(6*j)
    print(f'resolution {j}:', wavelons)

