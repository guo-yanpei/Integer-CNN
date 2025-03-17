
w_in=32
padd=1

f1=open("conv.txt","r")
conv_res=f1.readlines()   
y=conv_res[2].split(" ")[:-1]  # output
trace_res=open("dat.txt","r").readlines()
x=trace_res[2].split(" ")[:-1]  # input 
w=trace_res[4].split(" ")[:-1]  # weight

x_=x[:w_in*w_in]
w_=w[:9]

# w_in*w_in conv 3*3 (pad=1) = w_in*w_in
print(len(y),len(x_),len(w_))

padd_w=(w_in+2*padd)

X=[] # padded X
W=[0 for i in range(padd_w*3)] # converted w, size: (w_in+2)*3

for i in range(3):
    for j in range(3):
        W[padd_w*i+j]=int(w[3*i+j])
pad_y=[]
for i in range(padd_w*padd_w):
    xi=i//padd_w
    yi=i%padd_w
    if xi==0 or yi==0 or xi==padd_w-1 or yi==padd_w-1: # in padded area
        X.append(0)
    else:
        X.append(int(x_[(w_in-1-(xi-1))*w_in+w_in-1-(yi-1)]))
# X: degree (w_in+2)*(w_in+2)


Y=[0 for i in range(len(X)+len(W))]
# Y: degree (w_in+2)*(w_in+5)

# conv:
for i in range(len(X)):
    for j in range(len(W)):
        Y[i+j]+=X[i]*W[j]

for i in range(w_in*w_in):
    xi=i//w_in
    yi=i%w_in
    assert(int(y[i])==Y[(padd_w-1-xi)*padd_w+padd_w-1-yi])